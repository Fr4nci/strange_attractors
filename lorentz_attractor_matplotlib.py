import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np
import os
import random
import numba
# Estetica LaTeX
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})

# Classe Lorenz
class Lorenz:
    def __init__(self, a=10, b=28, c=8/3):
        self.a, self.b, self.c = a, b, c
        self.xs, self.ys, self.zs = random.uniform(-1, 1), 0, 0
        self.tmax = 100

    def lorenz(self, t, X):
        x, y, z = X
        x_dot = self.a * (y - x)
        y_dot = x * (self.b - z) - y
        z_dot = x * y - self.c * z
        return x_dot, y_dot, z_dot

    def solve(self):
        self.soln = solve_ivp(
            self.lorenz, (0, self.tmax), (self.xs, self.ys, self.zs),
            dense_output=True, max_step=0.1
        )

# Crea più attrattori
num_attrattori = 3
attractors = [Lorenz() for _ in range(num_attrattori)]
for a in attractors:
    a.solve()

# Interpolazione
t_interp = np.linspace(0, attractors[0].tmax, 100000)
trajectories = [a.soln.sol(t_interp) for a in attractors]

# Colori diversi per ogni attrattore
colors = ['#FF5733', '#1f77b4', '#2ECC71']  # arancio, blu, verde

# Cartella per immagini
save_folder = os.path.join(os.getcwd(), 'immagini_attrattore')
os.makedirs(save_folder, exist_ok=True)

# Plotting
plt.style.use(['dark_background'])
fig = plt.figure(figsize=(15,9), dpi=100)
ax = fig.add_subplot(projection='3d')
fig.set_facecolor('w')
ax.set_facecolor('w')
ax.cla()
ax.view_init(15, 145)
for idx, (xt, yt, zt) in enumerate(trajectories):
    ax.plot(xt[:-1], yt[:-1], zt[:-1], color=colors[idx], lw=0.15)
ax.set_title(f'$\\textbf{{Attrattori di Lorentz}}$', color='k', fontsize=12)
plt.show()

plt.style.use(['dark_background'])
fig = plt.figure(figsize=(15, 9), dpi=100)
ax = fig.add_subplot(projection='3d')
fig.set_facecolor('w')
ax.set_facecolor('w')
# Frame per frame
num_frames = 100000
for i in numba.prange(int(num_frames/10)):
    ax.cla()
    ax.view_init(15, 145)  # rotazione graduale

    for idx, (xt, yt, zt) in enumerate(trajectories):
        ax.plot(xt[:i*10], yt[:i*10], zt[:i*10], color=colors[idx], lw=0.15)

    ax.set_title(f'$\\textbf{{Attrattori di Lorentz}}$: frame {i}', color='k', fontsize=12)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    # Salva frame
    plt.savefig(os.path.join(save_folder, f'{i:04d}.png'),
                dpi=100, bbox_inches='tight', pad_inches=0.1)

plt.close()
print("✅ Frame colorati salvati in:", save_folder)
