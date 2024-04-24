import time
start_time = time.time()
import torch
import torch.nn as nn
from torch.nn import functional as F
import os

batch_size = 64 # 32
block_size = 256 # 8
max_iters = 5000 # 5000
max_new_tokens = 5000 # 1000
iters_print_size = max_iters/100 # print the loss every max_iters/10 iterations
learning_rate = 3e-4
torch.manual_seed(1337)
eval_iters = 200 # 200
n_embd = 384
num_heads = 6
n_layer = 6 # specify number of blocks
dropout = 0.2

if n_embd % num_heads > 0:
    raise Exception('n_embd must be divisible by num_heads')

# I don't think cuda is working properly, doesn't seem to give any performance boost
# or maybe even with device = 'cpu' it still runs on the gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} as device.', end='\n\n')

# read in text
print('Reading text...', end='')

with open(r'text\tinymatrix.txt', 'r', encoding='utf-8') as f:
    text = f.read()
print('done.', end='\n\n')

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    # generate randint from 0 to the size of the data with output size (8,)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad
def estimate_loss():
    out = {}
    m.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = m(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out

class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False) # n_embd, head_size
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # B, T, headsize
        q = self.query(x)
        v = self.value(x)

        wei = q @ k.transpose(-2, -1) * C**-0.5 # B, T, T
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # B, T, T
        wei = F.softmax(wei, dim=-1) # B, T, T
        wei = self.dropout(wei)

        out = wei @ v # B, T, T @ B, T, headisize --> B, T, headsize
        return out # B, T, headsize
    

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out
        # dimensions will be B, T, headsize*num_heads, hence usually choose headsize = num_heads/4
        

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), # per Attention is All You Need, dimensionality of ff layer is 4x
            nn. ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
    

class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_embd//head_size, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # self.sa_heads = MultiHeadAttention(num_heads, n_embd//num_heads) # head_size chosen so that output is still B, T, n_embd
        self.lm_head = nn.Linear(n_embd, vocab_size)
        # self.ffwd = FeedForward(n_embd)
        # self.blocks = nn.Sequential(
        #     Block(n_embd, num_heads),
        #     Block(n_embd, num_heads),
        #     Block(n_embd, num_heads),
        #     nn.LayerNorm(n_embd),
        # )
        

        self.blocks = nn.Sequential(*[Block(n_embd, num_heads) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B, T, n_embd)

        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, n_embd)
        x = tok_emb + pos_emb # (B, T, n_embd)
        # x = self.sa_heads(x)  # (B, T, n_embd)
        # x = self.ffwd(x)

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets==None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):

            # crop the size of idx since position_embedding_table can only take idx up to block_size
            idx_cond = idx[:, -block_size:]

            logits, loss = self(idx_cond)
            
            logits = logits[:, -1]
            # print(logits)
            probs = F.softmax(logits, dim=-1)
            # print(probs)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            # print('--')
        return idx

m = BigramLanguageModel().to(device)
optimizer = torch.optim.AdamW(m.parameters(), lr = learning_rate)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

parameter_count = count_parameters(m)

print('Beginning training...')
for steps in range(max_iters):

    if steps % iters_print_size == 0:
        print('    Starting iteration '+ str(steps) +'...', end='')

    xb, yb = get_batch('train')

    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    # print(loss.item())

    if steps % iters_print_size == 0:
        print(f'done.')
        # eval_loss = estimate_loss()
        # print(f"        train loss: {eval_loss['train']:.2f}, val loss: {eval_loss['val']:.2f}")

print(f'...training done.')
eval_loss = estimate_loss()
print(f"\ntrain loss: {eval_loss['train']:.2f}, val loss: {eval_loss['val']:.2f}")

print(f'\nGenerating {max_new_tokens} tokens.\n')

generated_text = decode(m.generate(torch.zeros((1,1), dtype=torch.long, device=device), max_new_tokens=max_new_tokens)[0].tolist())

# print(generated_text)

def get_unique_filename(base_filename):
    counter = 1
    filename = base_filename
    while os.path.exists(filename):
        filename = f"{base_filename.split('.')[0]}{counter}.txt"
        counter += 1
    return filename

filename = get_unique_filename('text\matrix_out.txt')

end_time = time.time()
elapsed_time = end_time - start_time
hours = int(elapsed_time // 3600)
minutes = int((elapsed_time % 3600) // 60)
seconds = int(elapsed_time % 60)

with open(filename, 'w') as file:
    file.write(f"Number of parameters used to train: {parameter_count:,}\n")
    file.write(f"Time taken to train and generate text: {hours} hours, {minutes} minutes, and {seconds} seconds.\n")
    file.write(f"------------------------------------------------------------------\n")
    file.write(generated_text)
    print(f"Data written to {filename}")