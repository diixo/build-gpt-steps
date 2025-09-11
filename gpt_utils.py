
import torch
from torch.nn import functional as F


def generate_new_text(prompt: str, model, enc, device, device_type, max_length=32):
    model.eval()

    inputs = enc.encode(prompt)
    tokens = torch.tensor(inputs, dtype=torch.long).unsqueeze(0) # (1, T)
    xgen = tokens.to(device)

    while xgen.size(1) < max_length:
        # forward the model to get the logits
        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(xgen) # (B, T, vocab_size)
            # take the logits at the last position
            logits = logits[:, -1, :] # (B, vocab_size)
            # get the probabilities
            probs = F.softmax(logits, dim=-1)

            xcol = torch.argmax(probs, dim=-1, keepdim=True)  # (B, 1)
            # append to the sequence
            xgen = torch.cat((xgen, xcol), dim=1)
    # return generated text
    decoded = xgen[0, :max_length].tolist()
    return enc.decode(decoded[len(inputs):])

