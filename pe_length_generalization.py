"""
Experiment 2: Length generalization, extending positional_encoding.py from
arXiv:2605.00662 (Bose, 2026).

Adds RoPE as a fourth condition (missing from the original paper's
comparison, and the de facto modern standard: Llama/Gemma/Qwen/Mistral/Phi
all use it). Trains once at L_TRAIN, then evaluates WITHOUT retraining at
longer lengths to test extrapolation.

NOTE: STPE is implemented as PE / L_TRAIN, a fixed rescaling by the trained
length, per the paper's definition (STPE = (T/L)*PE). This is one
reasonable convention; check it against the paper before trusting the
STPE numbers below, since the original repo only tested STPE at a single
fixed length and never under length mismatch.
"""
import torch, torch.nn as nn, torch.nn.functional as F, math

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

V = 50
D = 64
N_HEAD = 4
N_LAYER = 2
L_TRAIN = 64
EVAL_LENS = [64, 96, 128, 192]   # 64 = in-distribution, rest = extrapolation
STEPS = 600  # reduced from 3000 for sandbox time limits; task is trivial enough to converge
BS = 32

def get_batch(T):
    x = torch.randint(0, V, (BS, T), device=device)
    y = (x + 1) % V
    return x, y

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def rope_cos_sin(T, d_head, device):
    i = torch.arange(0, d_head, 2, device=device).float()
    inv_freq = 1.0 / (10000 ** (i / d_head))
    pos = torch.arange(T, device=device).float()
    freqs = torch.einsum('t,d->td', pos, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos()[None, None], emb.sin()[None, None]

class Block(nn.Module):
    def __init__(self, d, n_head, pe_type):
        super().__init__()
        self.n_head, self.d_head, self.pe_type = n_head, d // n_head, pe_type
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.ReLU(), nn.Linear(4 * d, d))
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)

    def forward(self, x):
        B, T, D = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).view(B, T, 3, self.n_head, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.pe_type == "rope":
            cos, sin = rope_cos_sin(T, self.d_head, x.device)
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        out = (att.softmax(dim=-1) @ v).transpose(1, 2).reshape(B, T, D)
        x = x + self.proj(out)
        x = x + self.ff(self.ln2(x))
        return x

class Model(nn.Module):
    def __init__(self, pe_type):
        super().__init__()
        self.pe_type = pe_type
        self.emb = nn.Embedding(V, D)
        self.pe = nn.Embedding(L_TRAIN, D) if pe_type == "rank" else None
        self.blocks = nn.ModuleList([Block(D, N_HEAD, pe_type) for _ in range(N_LAYER)])
        self.out = nn.Linear(D, V)

    def additive_pe(self, T, device):
        if self.pe_type == "rope":
            return 0.0
        i = torch.arange(0, D // 2, device=device).float()
        w = 1 / 10000 ** (2 * i / D)
        pos = torch.arange(T, device=device).float()
        pe = torch.zeros(T, D, device=device)
        if self.pe_type == "sin":
            pe[:, 0::2] = torch.sin(pos[:, None] * w)
            pe[:, 1::2] = torch.cos(pos[:, None] * w)
        elif self.pe_type == "stpe":
            pe[:, 0::2] = torch.sin(pos[:, None] * w) / L_TRAIN
            pe[:, 1::2] = torch.cos(pos[:, None] * w) / L_TRAIN
        elif self.pe_type == "rank":
            if T > L_TRAIN:
                return None  # cannot extrapolate: lookup table sized to L_TRAIN
            pe = self.pe(pos.long())
        return pe

    def forward(self, x):
        B, T = x.shape
        pe = self.additive_pe(T, x.device)
        if pe is None:
            return None
        x = self.emb(x) + pe
        for blk in self.blocks:
            x = blk(x)
        return self.out(x)

def train(pe_type):
    model = Model(pe_type).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(STEPS):
        x, y = get_batch(L_TRAIN)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, V), y.view(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    return model

@torch.no_grad()
def eval_at(model, T, n_batches=20):
    if T > L_TRAIN and model.pe_type == "rank":
        return None
    losses = []
    for _ in range(n_batches):
        x, y = get_batch(T)
        logits = model(x)
        if logits is None:
            return None
        losses.append(F.cross_entropy(logits.view(-1, V), y.view(-1)).item())
    return sum(losses) / len(losses) / math.log(2)  # BPC

if __name__ == "__main__":
    import sys, json
    pe_type = sys.argv[1]
    model = train(pe_type)
    row = {}
    for L in EVAL_LENS:
        bpc = eval_at(model, L)
        row[L] = None if bpc is None else round(bpc, 4)
    with open("results.jsonl", "a") as f:
        f.write(json.dumps({"pe_type": pe_type, "bpc": row}) + "\n")
    print(pe_type, row)
