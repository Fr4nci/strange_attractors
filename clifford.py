import numpy as np
import matplotlib.pyplot as plt
import numba
# Parametri dell'attrattore
params = [
    -1.7, 1.8, -1.9, -0.4
]

# Funzione iterativa
@numba.jit
def sprott_attractor(x, y, a):
    x_new = np.sin(a[0]*y) + a[2] * np.cos(a[0] * x)
    y_new = np.sin(a[1]*x) + a[3] * np.cos(a[1] * y)
    return x_new, y_new

# Inizializzazione
n_points = 10_000_0000000
x, y = 0.0, 0.0
xs, ys = np.empty(n_points), np.empty(n_points)

for i in range(n_points):
    x, y = sprott_attractor(x, y, params)
    xs[i] = x
    ys[i] = y

# Costruzione dell'immagine densa
bins = 3000  # più grande = più dettagliata ma più RAM
hist, xedges, yedges = np.histogram2d(xs, ys, bins=bins)

# Trasformazione logaritmica per l'effetto etereo
hist = np.log1p(hist)  # log(1 + x) per evitare log(0)

# Plot
plt.figure(figsize=(10, 10), dpi=300, facecolor='white')
plt.style.use('default')
plt.imshow(
    hist.T, origin='lower',
    cmap='Oranges',  # oppure 'magma', 'inferno', 'plasma'
    interpolation='bilinear',
    
    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]]
)
plt.axis('off')
plt.tight_layout()
plt.show()
"""n_points = 10000000
x, y = 0.6, 0.5
xs, ys = np.empty(n_points), np.empty(n_points)

# Iterazione
for i in range(n_points):
    x, y = sprott_attractor(x, y, params)
    xs[i] = x
    ys[i] = y

# Colori sfumati nel tempo
colors = np.logspace(0, 1, n_points)

# Plot
plt.style.use('default')
fig, ax = plt.subplots(figsize=(10, 10), dpi=1000)
sc = ax.scatter(xs, ys, c=colors, cmap='Blues', s=0.001, edgecolor='none')
ax.set_aspect('equal')
plt.axis('off')
plt.tight_layout()
plt.show()"""
