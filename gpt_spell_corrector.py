import os
import inspect
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from transformers import GPT2LMHeadModel
from gpt_utils import generate_new_text, generate_new_text_sft
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab_str = "#&+.0123456789'-abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ~"

model_path = "model.pt"

@dataclass
class GPTConfig:
    block_size: int = 48    # context length (word chars sequence)
    n_layer: int = 4        # number of layers
    n_head: int = 4         # number of heads
    n_embd: int = 128       # embedding dimension
    vocab_size: int = len(vocab_str)
    vocab_str: str = vocab_str


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd


    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    # head_layer+1 ​= head_layer + Attn(LN(head_layer)) + MLP(LN(head_layer))

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            # F.cross_entropy(ignore_index=-100) for SFT-mode
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        return logits, loss


    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}

        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model


    def configure_optimizers(self, weight_decay, learning_rate, device_type, master_process):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

######################################################################################################################

def load_txt(file_path: str) -> list:
    items = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(line)
    return items

class WordacyDataset:
    def __init__(self, words, context_size, vocab_str, batch_size):
        """
        context_size : context sequence length
        batch_size : amount words in one batch
        stoi       : symbol -> index
        device     : 'cuda' or 'cpu'
        """
        chars = sorted(list(set(vocab_str)))

        self.vocab_size = len(chars)
        self.stoi = {}
        self.itos = {}
        for i, ch in enumerate(chars, start=0):
            self.stoi[ch] = i
            self.itos[i] = ch

        self.eos_ch = '~'
        self.eos_token_id = self.stoi[self.eos_ch]
        self.sep_token_id = 0   # separator symbol=" " has 0-index into stoi.

        self.context_size = context_size
        self.batch_size = batch_size
        self.batches = [
            words[i: i + self.batch_size]
            for i in range(0, len(words), self.batch_size)
        ]

    def encode(self, word):
        return [self.stoi[c] for c in word]

    def decode(self, tokens):
        # tokens: list or tensor indices
        # remove padding (0)
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        return "".join([self.itos[i] for i in tokens if i != self.eos_token_id])

    def size(self):
        return len(self.batches)

    # implementation is custom_collate_fn(..) by rasbt, 
    def get_batch(self, idx, device):
        if idx >= len(self.batches) or (len(self.batches) == 0):
            return None, None

        batch_words = self.batches[idx]
        
        batch_max_length = min(
            max(len(item)+1 for item in batch_words),   # Find the longest sequence in the batch
            self.context_size                           # split by context length
        )

        inputs_lst = []
        targets_lst = []
        pad_token_id = self.eos_token_id

        for w in batch_words:
            new_item = self.encode(w) + [self.eos_token_id]

            # Pad sequences to max_length
            padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))

            inputs = torch.tensor(padded[:-1])  # Truncate the last token for inputs
            targets = torch.tensor(padded[1:])  # Shift +1 to the right for targets

            # New: Replace all but the first padding tokens in targets by ignore_index
            mask = targets == pad_token_id
            # Removes dimensions of size 1.
            indices = torch.nonzero(mask).squeeze()
            if indices.numel() > 1:
                targets[indices[1:]] = -100

            inputs_lst.append(inputs)
            targets_lst.append(targets)

        # Convert list of inputs and targets to tensors and transfer to target device
        inputs_tensor = torch.stack(inputs_lst).to(device)
        targets_tensor = torch.stack(targets_lst).to(device)
        return inputs_tensor, targets_tensor


    def get_batch_sft(self, idx, device):
        if idx >= len(self.batches) or (len(self.batches) == 0):
            return None, None

        batch_words = self.batches[idx]

        input_ids_list, labels_list = [], []
        max_len = 0

        # Токенизация и маскирование
        for w in batch_words:
            question_ids = self.encode(w) + [self.sep_token_id]
            answer_ids = self.encode(w) + [self.eos_token_id]
            # добавляю в конец принудительно endoftext, он не будет маскироваться на -100, потомучто является частью ответа

            input_ids = question_ids + answer_ids
            labels = [-100] * len(question_ids) + answer_ids  # маскируем вопрос

            input_ids_list.append(input_ids)
            labels_list.append(labels)
            max_len = max(max_len, len(input_ids))

        # Padding
        padded_inputs = []
        padded_labels = []
        for inp, lbl in zip(input_ids_list, labels_list):
            padded_inputs.append(torch.tensor(inp + [self.eos_token_id] * (max_len - len(inp)), dtype=torch.long))
            padded_labels.append(torch.tensor(lbl + [-100] * (max_len - len(lbl)), dtype=torch.long))

        inputs_tensor = torch.stack(padded_inputs).to(device)
        labels_tensor = torch.stack(padded_labels).to(device)
        return inputs_tensor, labels_tensor


def train(model: GPT, train_ds: WordacyDataset, learning_rate, max_epochs = 20):

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    max_batches = train_ds.size()

    progress = tqdm(range(max_epochs), desc="Training", position=0)

    for epoch in progress:
        losses = torch.zeros(max_batches)
        for id in range(max_batches):
            xb, yb = train_ds.get_batch_sft(id, device)

            # evaluate the loss
            logits, loss = model(xb, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses[id] = loss.item()

        # mean loss per epoch
        avg_loss = losses.mean().item()
        progress.set_postfix(avg_loss=f"{avg_loss:.4f}")


if __name__ == "__main__":

    train_words = load_txt("datasets/dictionary.txt")

    config = GPTConfig()
    train_ds = None
    model = None

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)

        config = GPTConfig(**checkpoint["config"])
        state_dict = checkpoint["state_dict"]

        train_ds = WordacyDataset(train_words, context_size=config.block_size, vocab_str=config.vocab_str, batch_size=1)
        print(f"::loaded, dataset: items={len(train_words)}, batches={train_ds.size()}, vocab_words={train_ds.vocab_size}")

        model = GPT(config)
        model.to(device)
        model.load_state_dict(state_dict)
    else:

        train_ds = WordacyDataset(train_words, context_size=config.block_size, vocab_str=config.vocab_str, batch_size=1)
        print(f"::create, dataset: items={len(train_words)}, batches={train_ds.size()}, vocab_words={train_ds.vocab_size}")

        model = GPT(config)
        model.to(device)

        train(
            model,
            train_ds,
            1e-5,
            max_epochs=50,
        )
        torch.save({
            "state_dict": model.state_dict(),
            "config": config.__dict__,
        }, model_path)
    ####################################################################################################################

    word_test = "rasterize"

    corrected = generate_new_text_sft(
        word_test,
        model,
        train_ds,   # as encoder
        device,
        device_type = str(device.type),
        #max_length = config.block_size-1
        )

    print(f"### input: {word_test}\n### corrected: {corrected}")
