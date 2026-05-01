import torch, torch.nn as nn, torch.nn.functional as F, math

# config
L, d, V = 64, 64, 50
steps, bs = 3000, 32
device = "cuda" if torch.cuda.is_available() else "cpu"

# synthetic dataset (next-token = +1 mod V)
def get_batch():
    x = torch.randint(0, V, (bs, L), device=device)
    y = (x + 1) % V
    return x, y

# model
class Model(nn.Module):
    def __init__(self, pe_type):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.pe_type = pe_type
        self.pe = nn.Embedding(L, d) if pe_type=="rank" else None
        self.attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, 4, 128, batch_first=True), 2)
        self.out = nn.Linear(d, V)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        if self.pe_type=="sin":
            i = torch.arange(0, d//2, device=x.device)
            w = 1/10000**(2*i/d)
            pe = torch.zeros(T, d, device=x.device)
            pe[:,0::2] = torch.sin(pos[:,None]*w)
            pe[:,1::2] = torch.cos(pos[:,None]*w)
        elif self.pe_type=="scaled":
            i = torch.arange(0, d//2, device=x.device)
            w = 1/10000**(2*i/d)
            pe = torch.zeros(T, d, device=x.device)
            pe[:,0::2] = torch.sin((pos[:,None]/L)*w)
            pe[:,1::2] = torch.cos((pos[:,None]/L)*w)
        else:  # rank embedding
            pe = self.pe(pos)
        x = self.emb(x) + pe
        x = self.attn(x)
        return self.out(x)

def train(pe_type):
    model = Model(pe_type).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(steps):
        x,y = get_batch()
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1,V), y.view(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()/math.log(2)

for t in ["sin","scaled","rank"]:
    print(t, "BPC:", train(t))
