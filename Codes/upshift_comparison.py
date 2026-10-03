import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# MODELO ACOPLADO DE 3 POBLACIONES BACTERIANAS
# ============================================================
# Variables:
#   R_i(t)     : fracción ribosomal / recursos de crecimiento
#   x_i(t)     : sustrato interno
#   N_i(t)     : tamaño poblacional
#   beta(t)    : disponibilidad efectiva del sustrato común
#
# Sistema normalizado:
#   dR_i/dt    = R_i g_i(x_i) ( f_i(x_i) - R_i )
#   dx_i/dt    = beta (1 - R_i) h_i(x_i) - R_i g_i(x_i)
#   dN_i/dt    = (mu_i - gamma_i) N_i
#   dbeta/dt   = - alpha beta sum_i N_i (1 - R_i) h_i(x_i)
#   mu_i       = R_i g_i(x_i)
# ============================================================

# Parámetros globales
gamma_default = 0.05  # mortalidad basal

tmax = 200.0
dt = 0.01

# Barrido de upshift: beta0_pre fijo y beta1 variable
beta0_pre = 0.5
beta1_barrido = np.linspace(0.55, 20.0, 70)

# Condiciones iniciales comunes
R0 = np.array([0.2, 0.2, 0.2], dtype=float)
x0 = np.array([0.1, 0.1, 0.1], dtype=float)
N0 = np.array([0.2, 0.2, 0.2], dtype=float)

gamma = np.array([gamma_default, gamma_default, gamma_default], dtype=float)

# Tres poblaciones.
# k efectivo = k2/k1:
#   k = 0.001  -> k1 = 1000, k2 = 1
#   k = 1      -> k1 = 1,    k2 = 1
#   k = 1000   -> k1 = 1,    k2 = 1000
poblaciones = [
    {"nombre": r"$k=0.01$", "k1": 100.0, "k2": 1.0},
    {"nombre": r"$k=1$",     "k1": 1.0,    "k2": 1.0},
    {"nombre": r"$k=100$",  "k1": 1.0,    "k2": 100.0},
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
    return -k1 / (k1 + x)**2


def g_prim_of_x(x, k2):
    x = positive(x)
    return k2 / (k2 + x)**2


def f_of_x(x, k1, k2):
    """
    Ley óptima:
        f(x) = 1 / (1 - (g'/g)(h/h'))
    """
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
        k_1 = rhs(yn)
        k_2 = rhs(yn + 0.5 * dt * k_1)
        k_3 = rhs(yn + 0.5 * dt * k_2)
        k_4 = rhs(yn + dt * k_3)

        y_next = yn + (dt / 6.0) * (k_1 + 2.0 * k_2 + 2.0 * k_3 + k_4)

        # Correcciones físicas
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
# EXPERIMENTO DE UPSHIFT
# ============================================================
# 1) Primero llevamos el sistema a un estado estacionario con beta = beta0_pre = 0.5.
#    Durante esta fase beta se mantiene fija, porque representa el medio pre-upshift.
# 2) Después se impone un salto beta0_pre -> beta1.
# 3) Para cada beta1 se mide:
#       - max_t mu_i(t)
#       - |R_i,max - R_i,min|
# ============================================================

def rhs_beta_fija(y, beta_fija):
    """
    Dinámica con beta mantenida constante.
    Se usa para alcanzar el estado estacionario previo al upshift.

    y = [R1,R2,R3, x1,x2,x3, N1,N2,N3]
    """
    R = clamp_R(y[0:3])
    x = positive(y[3:6])
    N = positive(y[6:9])
    beta = max(beta_fija, 0.0)

    h = h_of_x(x, k1_vals)
    g = g_of_x(x, k2_vals)
    f = f_of_x(x, k1_vals, k2_vals)

    mu = R * g

    dR = R * g * (f - R)
    dx = beta * (1.0 - R) * h - R * g
    dN = (mu - gamma) * N - mu * N * N

    return np.concatenate([dR, dx, dN])


def rk4_beta_fija(beta_fija, tmax_pre=300.0):
    """
    Integra hasta el estado estacionario pre-upshift con beta fija.
    Devuelve el último estado y_pre = [R,x,N].
    """
    n_steps = int(tmax_pre / dt) + 1

    y = np.zeros((n_steps, 9), dtype=float)
    y[0, 0:3] = R0
    y[0, 3:6] = x0
    y[0, 6:9] = N0

    for n in range(n_steps - 1):
        yn = y[n]
        k_1 = rhs_beta_fija(yn, beta_fija)
        k_2 = rhs_beta_fija(yn + 0.5 * dt * k_1, beta_fija)
        k_3 = rhs_beta_fija(yn + 0.5 * dt * k_2, beta_fija)
        k_4 = rhs_beta_fija(yn + dt * k_3, beta_fija)

        y_next = yn + (dt / 6.0) * (k_1 + 2.0 * k_2 + 2.0 * k_3 + k_4)

        y_next[0:3] = clamp_R(y_next[0:3])
        y_next[3:6] = positive(y_next[3:6])
        y_next[6:9] = positive(y_next[6:9])
        y[n + 1] = y_next

    return y[-1]


def rk4_upshift_desde_estado(y_pre, beta1):
    """
    Simulación post-upshift.
    Estado inicial: y_pre = [R_pre, x_pre, N_pre] obtenido con beta0_pre.
    En t=0 se impone beta(0)=beta1 y después beta(t) vuelve a ser dinámica,
    consumida por las poblaciones.
    """
    n_steps = int(tmax / dt) + 1
    t = np.linspace(0.0, tmax, n_steps)

    y = np.zeros((n_steps, 10), dtype=float)
    y[0, 0:3] = y_pre[0:3]
    y[0, 3:6] = y_pre[3:6]
    y[0, 6:9] = y_pre[6:9]
    y[0, 9] = beta1

    for n in range(n_steps - 1):
        yn = y[n]
        k_1 = rhs(yn)
        k_2 = rhs(yn + 0.5 * dt * k_1)
        k_3 = rhs(yn + 0.5 * dt * k_2)
        k_4 = rhs(yn + dt * k_3)

        y_next = yn + (dt / 6.0) * (k_1 + 2.0 * k_2 + 2.0 * k_3 + k_4)

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
# ESTADO ESTACIONARIO PRE-UPSHIFT, beta = 0.5
# ============================================================

y_pre = rk4_beta_fija(beta0_pre, tmax_pre=300.0)
R_pre = y_pre[0:3]
x_pre = y_pre[3:6]
N_pre = y_pre[6:9]
mu_pre = R_pre * g_of_x(x_pre, k2_vals)

print("\nEstado previo aproximado con beta0_pre =", beta0_pre)
print("Población                    R_pre        x_pre        N_pre        mu_pre")
for i, pob in enumerate(poblaciones):
    print(
        f"{pob['nombre']:30s} "
        f"{R_pre[i]:10.5f} {x_pre[i]:10.5f} {N_pre[i]:10.5f} {mu_pre[i]:10.5f}"
    )


# ============================================================
# BARRIDO EN beta1 POST-UPSHIFT
# ============================================================

mu_max = np.zeros((len(beta1_barrido), 3))
R_amplitud = np.zeros((len(beta1_barrido), 3))

for j, beta1_val in enumerate(beta1_barrido):
    t, R, x, N, beta, mu = rk4_upshift_desde_estado(y_pre, beta1_val)
    for i in range(3):
        mu_max[j, i] = np.max(mu[:, i])
        R_amplitud[j, i] = (np.max(R[:, i]) - np.min(R[:, i]))/(np.max(R[:, i]))


# ============================================================
# GRÁFICA ÚNICA: tres poblaciones en la misma figura
# ============================================================

# Tamaños de letra de la gráfica
FS_LABEL = 16
FS_TICKS = 13
FS_LEGEND = 11
FS_TITLE = 16

fig, ax1 = plt.subplots(figsize=(10, 6.2))

colores = ["tab:blue", "tab:orange", "tab:green"]

# Eje izquierdo: máximo de mu_i para las tres poblaciones
for i, pob in enumerate(poblaciones):
    ax1.plot(
        beta1_barrido,
        mu_max[:, i],
        color=colores[i],
        linewidth=2.3,
        label=rf"{pob['nombre']}: $\max_t \widetilde{{\mu}}_i(t)$"
    )

ax1.set_xlabel(r"$\widetilde{\beta}_1$ tras el upshift", fontsize=FS_LABEL)
ax1.set_ylabel(r"$\max_t \widetilde{\mu}_i(t)$", fontsize=FS_LABEL)
ax1.grid(alpha=0.3)
ax1.tick_params(axis="both", labelsize=FS_TICKS)

# Eje derecho: amplitud de R_i para las tres poblaciones
ax2 = ax1.twinx()

for i, pob in enumerate(poblaciones):
    ax2.plot(
        beta1_barrido,
        R_amplitud[:, i],
        color=colores[i],
        linestyle="--",
        linewidth=2.3,
        label=rf"{pob['nombre']}: $|R_{{i,\max}}-R_{{i,\min}}|$"
    )

ax2.set_ylabel(r"$\Delta R$", fontsize=FS_LABEL)
ax2.tick_params(axis="both", labelsize=FS_TICKS)

# Leyenda conjunta de ambos ejes
lineas_1, etiquetas_1 = ax1.get_legend_handles_labels()
lineas_2, etiquetas_2 = ax2.get_legend_handles_labels()

ax1.legend(
    lineas_1 + lineas_2,
    etiquetas_1 + etiquetas_2,
    loc="best",
    fontsize=FS_LEGEND,
    frameon=True
)

plt.tight_layout()
plt.show()


print("\nBarrido de upshift terminado.")
print("Columnas:")
print("beta1, mu_max_pob1, mu_max_pob2, mu_max_pob3, R_amp_pob1, R_amp_pob2, R_amp_pob3")

for idx in [0, len(beta1_barrido)//2, -1]:
    print(
        f"{beta1_barrido[idx]:.3f}, "
        f"{mu_max[idx,0]:.5f}, {mu_max[idx,1]:.5f}, {mu_max[idx,2]:.5f}, "
        f"{R_amplitud[idx,0]:.5f}, {R_amplitud[idx,1]:.5f}, {R_amplitud[idx,2]:.5f}"
    )
