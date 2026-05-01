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

STPE = PE / L

pe_dot   = PE @ PE.T
stpe_dot = STPE @ STPE.T

mask = ~torch.eye(L, dtype=torch.bool)
dist = torch.abs(pos - pos.T)[mask].numpy()

x  = pe_dot[mask].numpy()
y  = (stpe_dot[mask] * (L**2)).numpy()  # rescale to PE units

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.scatter(dist, x, s=3, alpha=0.35, color='steelblue',
           label='PE  (sinusoidal)')
ax.scatter(dist, y, s=3, alpha=0.35, color='darkorange',
           label=r'STPE $\times\, L^2$  (rescaled to PE units)')
ax.set_xlabel(r"Positional distance  $|pos - pos'|$", fontsize=12)
ax.set_ylabel("Dot product", fontsize=12)
ax.set_title(
    "Figure 1b:  Similarity structure vs positional distance\n"
    r"STPE rescaled by $L^2$ — curves are identical",
    fontsize=11)
ax.legend(fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig("fig1b_distance.png", dpi=200)
plt.show()
