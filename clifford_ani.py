import numpy as np
import matplotlib.pyplot as plt
import numba

# Funzione iterativa con Numba
@numba.njit
def sprott_attractor(x, y, a):
    x_new = np.sin(a[0]*y) + a[2] * np.cos(a[0] * x)
    y_new = np.sin(a[1]*x) + a[3] * np.cos(a[1] * y)
    return x_new, y_new

# Parametri iniziali
base_params = [1.7, 1.7, 0.6, 1.2]

n_points = 10_000_000
n_frames = 600  # numero di frame
bins = 3000

for t in numba.prange(335, n_frames):
    # Variazione dei parametri nel tempo
    delta = 0.2 * np.sin(2 * np.pi * t / n_frames)
    params = [
        base_params[0] + delta,
        base_params[1] + delta,
        base_params[2] + 0.1 * np.cos(2 * np.pi * t / n_frames),
        base_params[3] + 0.1 * np.sin(2 * np.pi * t / n_frames)
    ]

    # Inizializzazione delle orbite
    x, y = 0.1, 0.0
    xs = np.empty(n_points, dtype=np.float32)
    ys = np.empty(n_points, dtype=np.float32)

    for i in range(n_points):
        x, y = sprott_attractor(x, y, params)
        xs[i] = x
        ys[i] = y

    # Istogramma 2D
    hist, xedges, yedges = np.histogram2d(xs, ys, bins=bins)
    hist = np.log1p(hist)

    # Plot
    plt.figure(figsize=(10, 10), dpi=300)
    plt.imshow(
        hist.T, origin='lower',
        cmap='hot',
        interpolation='bilinear',
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]]
    )
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(f"./Clifford/frame_{t:03d}.png", bbox_inches='tight', pad_inches=0)
    plt.close()
