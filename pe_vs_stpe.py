import torch
import matplotlib.pyplot as plt
import numpy as np

L, d = 128, 128

pos   = torch.arange(L).unsqueeze(1).float()
i_idx = torch.arange(0, d//2).unsqueeze(0).float()
omega = 1.0 / (10000 ** (2 * i_idx / d))

phase = pos * omega
PE = torch.zeros(L, d)
PE[:, 0::2] = torch.sin(phase)
PE[:, 1::2] = torch.cos(phase)

STPE = PE / L  # amplitude scaling: STPE = (T/L)*PE, T=1

pe_dot   = PE @ PE.T
stpe_dot = STPE @ STPE.T

mask = ~torch.eye(L, dtype=torch.bool)
x = pe_dot[mask].numpy()
y = stpe_dot[mask].numpy()

# perfect reference line
xs = np.linspace(x.min(), x.max(), 200)
ys = xs / (L ** 2)

fig, ax = plt.subplots(figsize=(5.5, 5))
ax.scatter(x, y, s=4, alpha=0.4, color='steelblue', label='position pairs')
ax.plot(xs, ys, 'r--', lw=1.5, label=r'$y = x \/ L^2$  (exact)')
ax.set_xlabel(r"$\langle PE(p),\; PE(q) \rangle$", fontsize=12)
ax.set_ylabel(r"$\langle STPE(p),\; STPE(q) \rangle$", fontsize=12)
ax.set_title(
    "Figure 1a:  PE vs STPE dot products\n"
    r"Pearson $r$ = 1.000  —  slope = $1/L^2$ (exact)",
    fontsize=11)
ax.legend(fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig("fig1a_scatter.png", dpi=200)
plt.show()
