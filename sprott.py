import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os
import random

# Estetica LaTeX
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})

# Classe Sprott 2D
class Sprott2D:
    def __init__(self, a=1.5, b=-0.5, c=-1.0, d=0.7):
        self.a, self.b, self.c, self.d = a, b, c, d
        self.x0, self.y0 = random.uniform(-1, 1), random.uniform(-1, 1)
        self.tmax = 100

    def system(self, t, X):
        x, y = X
        dxdt = y
        dydt = self.a * x + self.b * y + self.c * x**2 + self.d * x * y
        return [dxdt, dydt]

    def solve(self):
        self.soln = solve_ivp(
            self.system, (0, self.tmax), (self.x0, self.y0),
            dense_output=True, max_step=0.1
        )

# Crea più attrattori
num_attrattori = 1
attractors = [Sprott2D() for _ in range(num_attrattori)]
for a in attractors:
    a.solve()

# Interpolazione temporale
t_interp = np.linspace(0, attractors[0].tmax, 100000)
trajectories = [a.soln.sol(t_interp) for a in attractors]

# Colori
colors = ['#FF5733', '#1f77b4', '#2ECC71']

# Cartella immagini
save_folder = os.path.join(os.getcwd(), 'immagini_sprott2D')
os.makedirs(save_folder, exist_ok=True)

# Plot 2D
plt.style.use(['dark_background'])
fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
fig.set_facecolor('w')
ax.set_facecolor('w')
ax.cla()

for idx, (xt, yt) in enumerate(trajectories):
    ax.plot(xt, yt, color=colors[idx], lw=0.3, alpha=0.7)

ax.set_title(r'Attratttori di tipo Sprott 2D', color='k', fontsize=14)
ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
plt.show()

# Frame-by-frame (opzionale)
num_frames = 100000
for i in range(0, num_frames, 100):  # ogni 100 punti
    fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
    fig.set_facecolor('w')
    ax.set_facecolor('w')

    for idx, (xt, yt) in enumerate(trajectories):
        ax.plot(xt[:i], yt[:i], color=colors[idx], lw=0.3, alpha=0.7)

    ax.set_title(f'Attrattori di tipo Sprott 2D: frame {i//100}', color='k', fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, f'{i//100:04d}.png'),
                dpi=100, bbox_inches='tight', pad_inches=0.1)
    plt.close()

print("✅ Frame salvati in:", save_folder)
