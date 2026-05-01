import torch
import matplotlib.pyplot as plt

L, d = 128, 128

pos   = torch.arange(L).unsqueeze(1).float()
i_idx = torch.arange(0, d//2).unsqueeze(0).float()
omega = 1.0 / (10000 ** (2 * i_idx / d))

phase = pos * omega
PE = torch.zeros(L, d)
PE[:, 0::2] = torch.sin(phase)
PE[:, 1::2] = torch.cos(phase)

# frequency-compressed encoding — distinct from amplitude scaling
phase_f = (pos / L) * omega
STPE_freq = torch.zeros(L, d)
STPE_freq[:, 0::2] = torch.sin(phase_f)
STPE_freq[:, 1::2] = torch.cos(phase_f)

pe_dot   = PE @ PE.T
freq_dot = STPE_freq @ STPE_freq.T

mask = ~torch.eye(L, dtype=torch.bool)
dist = torch.abs(pos - pos.T)[mask].numpy()
x  = pe_dot[mask].numpy()
yf = freq_dot[mask].numpy()

# correlations for annotation
xf_t = pe_dot[mask]; yf_t = freq_dot[mask]
pearson  = torch.corrcoef(torch.stack([xf_t, yf_t]))[0,1].item()
rx = xf_t.argsort().argsort().float()
ry = yf_t.argsort().argsort().float()
spearman = torch.corrcoef(torch.stack([rx, ry]))[0,1].item()

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.scatter(dist, x,  s=3, alpha=0.35, color='steelblue',
           label='PE  (sinusoidal)')
ax.scatter(dist, yf, s=3, alpha=0.35, color='tomato',
           label=r'STPE$_{\rm freq}$ = $\sin(pos/L \cdot \omega)$')
ax.set_xlabel(r"Positional distance  $|pos - pos'|$", fontsize=12)
ax.set_ylabel("Dot product", fontsize=12)
ax.set_title(
    "Figure 1c:  Frequency compression — a distinct encoding\n"
    f"Pearson r = {pearson:.3f}   Spearman ρ = {spearman:.3f}",
    fontsize=11)
ax.legend(fontsize=10)
ax.text(0.98, 0.55,
        "Ordering preserved (ρ = 0.965)\nMetric not preserved (r = 0.706)\n\n"
        r"$\sin(pos/L \cdot \omega) \neq (1/L)\cdot\sin(pos \cdot \omega)$",
        transform=ax.transAxes, ha='right', va='center', fontsize=9,
        bbox=dict(boxstyle='round,pad=0.4', fc='#fff7ed', ec='#fdba74', lw=0.8))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig("fig1c_freqcomp.png", dpi=200)
plt.show()
