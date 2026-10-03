import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# MODELO ACOPLADO DE 3 POBLACIONES BACTERIANAS
# ============================================================
# Variables:
#   R_i(t)     : fracción ribosomal
#   x_i(t)     : sustrato interno
#   N_i(t)     : tamaño poblacional
#   beta(t)    : disponibilidad efectiva del sustrato común
#
# Sistema NORMALIZADO:
#   dR_i/dt    = R_i g_i(x_i) ( f_i(x_i) - R_i )
#   dx_i/dt    = beta (1 - R_i) h_i(x_i) - R_i g_i(x_i)
#   dN_i/dt    = (mu_i - gamma_i) N_i - mu_i N_i^2
#   dbeta/dt   = - beta sum_i N_i h_i(x_i)
#   mu_i       = R_i g_i(x_i)
# ============================================================


# ============================================================
# 1. PARÁMETROS GENERALES
# ============================================================
#tasas de mortalidad
gamma_default = 0.15

# Condición basal y upshift
beta0_pre = 0.5 
beta1 = 2.5        # cambia este valor para ver otro salto nutricional

# Tiempos
tmax_pre = 200.0   # tiempo mostrado antes del upshift
tmax_relax = 200.0 # tiempo total para alcanzar estado estacionario
tmax_post = 200.0  # tiempo mostrado después del upshift

dt = 0.01

# Condiciones iniciales comunes
R0 = np.array([0.2, 0.2, 0.2], dtype=float)
x0 = np.array([0.1, 0.1, 0.1], dtype=float)
N0 = np.array([0.2, 0.2, 0.2], dtype=float)

#todas las poblaciones tienen la misma tasa de mortalidad porq son la misma bacteria, entonces
#la guardo como vector por comodidad
gamma = np.array([gamma_default, gamma_default, gamma_default], dtype=float)

# Tres poblaciones
poblaciones = [
    {"nombre": r"Saturada, $k=0.01$", "k1": 100.0, "k2": 1.0},
    {"nombre": r"Subsaturada, $k=1$", "k1": 1.0, "k2": 1.0},
    {"nombre": r"Ext. subsaturada, $k=100$", "k1": 1.0, "k2": 100.0},
]

k1_vals = np.array([p["k1"] for p in poblaciones], dtype=float)
k2_vals = np.array([p["k2"] for p in poblaciones], dtype=float)

colores = ["tab:blue", "tab:orange", "tab:green"]


# ============================================================
# 2. FUNCIONES DEL MODELO
# ============================================================

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

# ============================================================
# DINÁMICA PRE-UPSHIFT CON beta FIJA hasta alcanzar est estacionario
# ============================================================

def rhs_beta_fija(y, beta_fija):
    """
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


def integrar_beta_fija(beta_fija, tmax):
    """
    Integra la fase con beta fija.
    """

    n_steps = int(tmax / dt) + 1
    t = np.linspace(0.0, tmax, n_steps)

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

        y_next = yn + (dt / 6.0) * (k_1 + 2.0*k_2 + 2.0*k_3 + k_4)

        y_next[0:3] = clamp_R(y_next[0:3])
        y_next[3:6] = positive(y_next[3:6])
        y_next[6:9] = positive(y_next[6:9])

        y[n + 1] = y_next

    R = y[:, 0:3]
    x = y[:, 3:6]
    N = y[:, 6:9]
    beta = np.full_like(t, beta_fija)

    return t, R, x, N, beta, y


# ============================================================
# DINÁMICA POST-UPSHIFT CON beta DINÁMICA: hay un salto instantaneo beta1, y despues la dinamica de 
#beta prima es la del modelo acoplado (arriba esta)
# ============================================================

def rhs_post_upshift(y):
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


def integrar_post_upshift(y_pre, beta1):
    """
    Integra la fase posterior al upshift.
    """

    n_steps = int(tmax_post / dt) + 1
    t = np.linspace(0.0, tmax_post, n_steps)

    y = np.zeros((n_steps, 10), dtype=float)

    y[0, 0:3] = y_pre[0:3]
    y[0, 3:6] = y_pre[3:6]
    y[0, 6:9] = y_pre[6:9]
    y[0, 9] = beta1

    for n in range(n_steps - 1):
        yn = y[n]

        k_1 = rhs_post_upshift(yn)
        k_2 = rhs_post_upshift(yn + 0.5 * dt * k_1)
        k_3 = rhs_post_upshift(yn + 0.5 * dt * k_2)
        k_4 = rhs_post_upshift(yn + dt * k_3)

        y_next = yn + (dt / 6.0) * (k_1 + 2.0*k_2 + 2.0*k_3 + k_4)

        y_next[0:3] = clamp_R(y_next[0:3])
        y_next[3:6] = positive(y_next[3:6])
        y_next[6:9] = positive(y_next[6:9])
        y_next[9] = max(y_next[9], 0.0)

        y[n + 1] = y_next

    R = y[:, 0:3]
    x = y[:, 3:6]
    N = y[:, 6:9]
    beta = y[:, 9]

    return t, R, x, N, beta, y


# ============================================================
# SIMULACIÓN COMPLETA: ESTACIONARIO + UPSHIFT
# ============================================================

# 1. Integramos mucho tiempo con beta0 para asegurar estacionario
t_relax, R_relax, x_relax, N_relax, beta_relax, y_relax = integrar_beta_fija(
    beta0_pre,
    tmax_relax
)

# Estado estacionario previo
y_pre = y_relax[-1]

# 2. Para mostrar la fase estacionaria, repetimos el estado estacionario
#    durante un intervalo temporal antes del upshift
n_pre_show = int(tmax_pre / dt) + 1
t_pre_show = np.linspace(-tmax_pre, 0.0, n_pre_show)

R_pre_show = np.tile(y_pre[0:3], (n_pre_show, 1))
x_pre_show = np.tile(y_pre[3:6], (n_pre_show, 1))
N_pre_show = np.tile(y_pre[6:9], (n_pre_show, 1))
beta_pre_show = np.full(n_pre_show, beta0_pre)

# 3. Integramos la fase post-upshift
t_post, R_post, x_post, N_post, beta_post, y_post = integrar_post_upshift(
    y_pre,
    beta1
)

# 4. Pegamos ambas fases en una sola serie temporal
t_total = np.concatenate([t_pre_show, t_post[1:]])

R_total = np.vstack([R_pre_show, R_post[1:, :]])
x_total = np.vstack([x_pre_show, x_post[1:, :]])
N_total = np.vstack([N_pre_show, N_post[1:, :]])
beta_total = np.concatenate([beta_pre_show, beta_post[1:]])

print("\nEstado estacionario previo con beta0_pre =", beta0_pre)
print("Población                    R_pre        x_pre        N_pre")
for i, pob in enumerate(poblaciones):
    print(
        f"{pob['nombre']:30s} "
        f"{y_pre[i]:10.5f} "
        f"{y_pre[3+i]:10.5f} "
        f"{y_pre[6+i]:10.5f}"
    )
# ============================================================
# CÁLCULO DE mu_i(t)
# ============================================================

# Tasa de crecimiento de cada población:
# mu_i(t) = R_i(t) g_i(x_i(t))
mu_total = R_total * g_of_x(x_total, k2_vals)


# ============================================================
# GRÁFICAS TEMPORALES: x_i, mu_i, N_i, R_i
# ============================================================

FS_LABEL = 17
FS_TICKS = 15
FS_LEGEND = 14

fig, axes = plt.subplots(
    4, 1,
    figsize=(11, 10),
    sharex=True,
    constrained_layout=True
)

nombres_leyenda = [r"$k=0.01$", r"$k=1$", r"$k=100$"]

# ------------------------------------------------------------
# x_i(t)
# ------------------------------------------------------------

for i in range(3):
    axes[0].plot(
        t_total,
        x_total[:, i],
        color=colores[i],
        linewidth=2.5,
        label=nombres_leyenda[i]
    )

axes[0].axvline(0, color="black", linestyle="--", linewidth=1.8)
axes[0].set_ylabel(r"$x_i(t)$", fontsize=FS_LABEL)
axes[0].tick_params(axis="both", labelsize=FS_TICKS)
axes[0].grid(alpha=0.3)
axes[0].legend(fontsize=FS_LEGEND, frameon=True, loc="best")


# ------------------------------------------------------------
# mu_i(t)
# ------------------------------------------------------------

for i in range(3):
    axes[1].plot(
        t_total,
        mu_total[:, i],
        color=colores[i],
        linewidth=2.5
    )

axes[1].axvline(0, color="black", linestyle="--", linewidth=1.8)
axes[1].set_ylabel(r"$\widetilde{\mu}_i(t)$", fontsize=FS_LABEL)
axes[1].tick_params(axis="both", labelsize=FS_TICKS)
axes[1].grid(alpha=0.3)


# ------------------------------------------------------------
# N_i(t)
# ------------------------------------------------------------

for i in range(3):
    axes[2].plot(
        t_total,
        N_total[:, i],
        color=colores[i],
        linewidth=2.5
    )

axes[2].axvline(0, color="black", linestyle="--", linewidth=1.8)
axes[2].set_ylabel(r"$N_i(t)$", fontsize=FS_LABEL)
axes[2].tick_params(axis="both", labelsize=FS_TICKS)
axes[2].grid(alpha=0.3)


# ------------------------------------------------------------
# R_i(t)
# ------------------------------------------------------------

for i in range(3):
    axes[3].plot(
        t_total,
        R_total[:, i],
        color=colores[i],
        linewidth=2.5
    )

axes[3].axvline(0, color="black", linestyle="--", linewidth=1.8)
axes[3].set_xlabel(r"Tiempo", fontsize=FS_LABEL)
axes[3].set_ylabel(r"$R_i(t)$", fontsize=FS_LABEL)
axes[3].tick_params(axis="both", labelsize=FS_TICKS)
axes[3].grid(alpha=0.3)


# ------------------------------------------------------------
# ACORTAR EJE X
# ------------------------------------------------------------

axes[3].set_xlim(-10, 30)

# Sin título general
plt.show()