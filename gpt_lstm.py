
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class GPTConfig:
    block_size: int = 64    # max sequence length
    n_layer: int = 4        # number of layers
    n_head: int = 4         # number of heads
    n_embd: int = 128       # embedding dimension


class CausalSelfAttentionLSTM(nn.Module):

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

        lstm_hidden = config.n_embd
        # LSTM after attention
        self.lstm = nn.LSTM(config.n_embd, lstm_hidden, batch_first=True, bidirectional=False)
        self.lstm_proj = nn.Linear(lstm_hidden, config.n_embd)


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

        # --- LSTM after attention ---
        lstm_out, _ = self.lstm(y)
        lstm_out = self.lstm_proj(lstm_out)
        # TODO: can be added the residual with y into block above
        return lstm_out


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


class BlockLSTM(nn.Module):
    # head_layer+1 ​= head_layer + Attn(LN(head_layer)) + MLP(LN(head_layer))

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttentionLSTM(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # Attention + LSTM + residual
        attn_out = self.attn_lstm(self.ln_1(x))
        x = x + attn_out  # residual

        x = x + self.mlp(self.ln_2(x))
        return x


class GPT_LSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([BlockLSTM(config) for _ in range(config.n_layer)]),
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
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


class WordacyDataset:
    def __init__(self, words, block_size, batch_size, device):
        """
        block_size : maximum sequence length
        batch_size : amount words in one batch
        stoi       : symbol -> index
        device     : 'cuda' or 'cpu'
        """
        text = "#+.0123456789'-abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ" + "~"

        chars = sorted(list(set(text)))
        #vocab_size = len(chars)
        self.stoi = {}
        self.itos = {}
        for i, ch in enumerate(chars, start=0):
        #for i, ch in enumerate(chars, start=1):
            self.stoi[ch] = i
            self.itos[i] = ch

        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.batches = [
            words[i:i + self.batch_size]
            for i in range(0, len(words), self.batch_size)
        ]
        self.reset()

    def encode(self, word):
        return [self.stoi[c] for c in word]

    def decode(self, tokens):
            # tokens: list or tensor indices
            # remove padding (0)
            if isinstance(tokens, torch.Tensor):
                tokens = tokens.tolist()
            return "".join([self.itos[i] for i in tokens if i != 0])

    def reset(self):
        self.idx = -1

    def get_batch(self):
        if self.idx >= len(self.batches) or (len(self.batches) == 0):
            self.reset()
            return None

        self.idx += 1
        batch_words = self.batches[self.idx]

        x_list = []
        y_list = []

        for w in batch_words:
            tokens = self.encode(w)
            if len(tokens) > self.block_size:
                tokens = tokens[:self.block_size]
            else:
                tokens = tokens + [0] * (self.block_size - len(tokens))

            x_list.append(tokens[:-1])
            y_list.append(tokens[1:])

        x = torch.tensor(x_list, dtype=torch.long, device=self.device)
        y = torch.tensor(y_list, dtype=torch.long, device=self.device)
        return x, y


if __name__ == "__main__":

    ###############################################
    # itos = {0: "<PAD>"}
    # stoi = {"<PAD>": 0}
    # itos = {0: "<PAD>"}
    # stoi = {"<PAD>": 0}

    train_words = [
        "ai", "machine", "learning", "large", "language", "model", "train", "transformer",
        "builds", "gpt-2", "fine", "tuning", "steps", "wordacy", "spelling", "correction",
        ]

    train_gen = WordacyDataset(train_words, block_size=32, batch_size=4, stoi=stoi, device=device)


    epochs = 10
    for i in enumerate(epochs):
        while True:
            x, y = train_gen.get_batch()
            if x is None:
                break

