import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path

# ============================================================
# MODELO ACOPLADO DE 3 POBLACIONES BACTERIANAS
# GIF con barrido en beta0 y doble gráfica:
#   - arriba: mu_i(t)
#   - abajo: R_i(t)
# ============================================================

# Parámetros globales
gamma_default = 0.05
tmax = 90.0
dt = 0.01

# Barrido de beta(0)
beta_iniciales = np.linspace(0.1, 25.0, 50)

# Condiciones iniciales comunes
R0 = np.array([0.2, 0.2, 0.2], dtype=float)
x0 = np.array([1.1, 1.1, 1.1], dtype=float)
N0 = np.array([0.2, 0.2, 0.2], dtype=float)

gamma = np.array([gamma_default, gamma_default, gamma_default], dtype=float)

# Tres poblaciones
poblaciones = [
    {"nombre": r"Población 1: $k=0.001$", "k1": 100.0, "k2": 1.0},
    {"nombre": r"Población 2: $k=1$",     "k1": 1.0,    "k2": 1.0},
    {"nombre": r"Población 3: $k=1000$",  "k1": 1.0,    "k2": 100.0},
]

k1_vals = np.array([p["k1"] for p in poblaciones], dtype=float)
k2_vals = np.array([p["k2"] for p in poblaciones], dtype=float)


def positive(z):
    return np.maximum(z, 1e-12)


def clamp_R(R):
    return np.clip(R, 0.0, 1.0)


def h_of_x(x, k1):
    x = positive(x)
    return k1 / (k1 + x)


def g_of_x(x, k2):
    x = positive(x)
    return x / (k2 + x)


def h_prim_of_x(x, k1):
    x = positive(x)
    return -k1 / (k1 + x) ** 2


def g_prim_of_x(x, k2):
    x = positive(x)
    return k2 / (k2 + x) ** 2


def f_of_x(x, k1, k2):
    x = positive(x)
    h = h_of_x(x, k1)
    g = g_of_x(x, k2)
    hp = h_prim_of_x(x, k1)
    gp = g_prim_of_x(x, k2)
    f = 1.0 / (1.0 - (gp / g) * (h / hp))
    return np.clip(f, 0.0, 1.0)


def rhs(y):
    """
    y = [R1,R2,R3, x1,x2,x3, N1,N2,N3, beta]
    """
    R = clamp_R(y[0:3])
    x = positive(y[3:6])
    N = positive(y[6:9])
    beta = max(y[9], 0.0)

    h = h_of_x(x, k1_vals)
    g = g_of_x(x, k2_vals)
    f = f_of_x(x, k1_vals, k2_vals)

    mu = R * g

    dR = R * g * (f - R)
    dx = beta * (1.0 - R) * h - R * g
    dN = (mu - gamma) * N - mu * N * N
    dbeta = -beta * np.sum(N * h)

    return np.concatenate([dR, dx, dN, np.array([dbeta])])


def rk4(beta0):
    n_steps = int(tmax / dt) + 1
    t = np.linspace(0.0, tmax, n_steps)

    y = np.zeros((n_steps, 10), dtype=float)
    y[0, 0:3] = R0
    y[0, 3:6] = x0
    y[0, 6:9] = N0
    y[0, 9] = beta0

    for n in range(n_steps - 1):
        yn = y[n]
        k1 = rhs(yn)
        k2 = rhs(yn + 0.5 * dt * k1)
        k3 = rhs(yn + 0.5 * dt * k2)
        k4 = rhs(yn + dt * k3)

        y_next = yn + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        y_next[0:3] = clamp_R(y_next[0:3])
        y_next[3:6] = positive(y_next[3:6])
        y_next[6:9] = positive(y_next[6:9])
        y_next[9] = max(y_next[9], 0.0)
        y[n + 1] = y_next

    R = y[:, 0:3]
    x = y[:, 3:6]
    N = y[:, 6:9]
    beta = y[:, 9]
    mu = R * g_of_x(x, k2_vals)

    return t, R, x, N, beta, mu


# ============================================================
# Precalcular todas las simulaciones del barrido
# ============================================================

simulaciones = []
mu_global_max = 0.0
R_global_max = 0.0

for beta0 in beta_iniciales:
    t, R, x, N, beta, mu = rk4(beta0)
    mu_max = np.max(mu, axis=0)
    R_min = np.min(R, axis=0)
    R_max = np.max(R, axis=0)
    R_amp = np.abs(R_max - R_min)

    simulaciones.append({
        "beta0": beta0,
        "t": t,
        "R": R,
        "mu": mu,
        "mu_max": mu_max,
        "R_amp": R_amp,
        "beta_final": beta[-1],
    })

    mu_global_max = max(mu_global_max, np.max(mu))
    R_global_max = max(R_global_max, np.max(R))

mu_global_max = max(mu_global_max * 1.05, 0.1)
R_global_max = max(R_global_max * 1.05, 1.0)

# ============================================================
# Crear figura doble y animación
# ============================================================

fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
ax_mu, ax_R = axes

lineas_mu = []
lineas_R = []

for _ in range(3):
    linea_mu, = ax_mu.plot([], [], linewidth=2.2)
    linea_R, = ax_R.plot([], [], linewidth=2.2)
    lineas_mu.append(linea_mu)
    lineas_R.append(linea_R)

ax_mu.set_ylabel(r"$\widetilde{\mu}_i(t)$")
ax_mu.set_ylim(0, mu_global_max)
ax_mu.grid(alpha=0.3)

ax_R.set_xlabel("Tiempo")
ax_R.set_ylabel(r"$R_i(t)$")
ax_R.set_ylim(0, R_global_max)
ax_R.grid(alpha=0.3)
ax_R.set_xlim(0, tmax)

titulo = fig.suptitle("")

def init():
    for linea in lineas_mu + lineas_R:
        linea.set_data([], [])
    return lineas_mu + lineas_R + [titulo]

def update(frame):
    sim = simulaciones[frame]
    t = sim["t"]
    mu = sim["mu"]
    R = sim["R"]
    mu_max = sim["mu_max"]
    R_amp = sim["R_amp"]
    beta0 = sim["beta0"]
    beta_final = sim["beta_final"]

    for i, pob in enumerate(poblaciones):
        lineas_mu[i].set_data(t, mu[:, i])
        lineas_mu[i].set_label(
            rf"{pob['nombre']}, $\max \widetilde{{\mu}}={mu_max[i]:.3f}$"
        )

        lineas_R[i].set_data(t, R[:, i])
        lineas_R[i].set_label(
            rf"{pob['nombre']}, $|R_{{\max}}-R_{{\min}}|={R_amp[i]:.3f}$"
        )

    titulo.set_text(
        rf"Dinámica temporal para $\widetilde{{\beta}}(0)={beta0:.2f}$"
        + "\n"
        + rf"$\widetilde{{\beta}}(t_{{final}})={beta_final:.3f}$"
    )

    ax_mu.legend(frameon=True, fontsize=8, loc="best")
    ax_R.legend(frameon=True, fontsize=8, loc="best")

    return lineas_mu + lineas_R + [titulo]

fig.tight_layout(rect=[0, 0, 1, 0.95])

anim = FuncAnimation(
    fig,
    update,
    frames=len(simulaciones),
    init_func=init,
    interval=180,
    blit=False,
    repeat=True,
)

# Guardar el GIF en la misma carpeta del script
try:
    carpeta_salida = Path(__file__).resolve().parent
except NameError:
    carpeta_salida = Path.cwd()

ruta_gif = carpeta_salida / "dinamica_mu_R_barrido_beta0_0p1_25.gif"
anim.save(str(ruta_gif), writer=PillowWriter(fps=6), dpi=120)

print(f"GIF guardado en: {ruta_gif.resolve()}")

plt.show()
