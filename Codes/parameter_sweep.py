import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# ============================================================
# MODELO PARA UNA POBLACIÓN CON PARÁMETRO k VARIABLE
# ============================================================
# Variables:
#   R(t)     : fracción ribosomal / recurso interno de crecimiento
#   x(t)     : sustrato interno
#   N(t)     : tamaño poblacional
#   beta(t)  : disponibilidad efectiva del sustrato
#
# Sistema normalizado:
#   dR/dt    = R g(x) ( f(x) - R )
#   dx/dt    = beta (1 - R) h(x) - R g(x)
#   dN/dt    = (mu - gamma) N - mu N^2
#   dbeta/dt = - beta N h(x)
#   mu       = R g(x)
# ============================================================


# ============================================================
# 1. PARÁMETROS GENERALES
# ============================================================

gamma = 0.05

# Condición basal pre-upshift
beta0_pre = 0.5

# Barrido de beta1
beta1_vals = np.linspace(0.55, 10.0, 35)

# Barrido de estrategias k
k_vals = np.logspace(np.log10(0.01), np.log10(100), 45)

# Condiciones iniciales comunes
R0 = 0.2
x0 = 0.1
N0 = 0.2

# Tiempos y paso temporal
# Valores moderados para que no tarde demasiado
dt = 0.05
tmax_pre = 120.0
tmax_post = 60.0

# Tamaños de letra grandes
FS_TITLE = 22
FS_LABEL = 20
FS_TICKS = 16
FS_CBAR = 18


# ============================================================
# 2. FUNCIONES DEL MODELO
# ============================================================

def positive(z):
    return np.maximum(z, 1e-12)


def clamp_R(R):
    return np.clip(R, 0.0, 1.0)


def k_to_k1_k2(k):
    """
    Traduce el parámetro efectivo k = k2/k1 a k1 y k2.

    Convención:
        k = 0.01  -> k1 = 100, k2 = 1
        k = 1     -> k1 = 1,   k2 = 1
        k = 100   -> k1 = 1,   k2 = 100
    """
    if k < 1.0:
        k1 = 1.0 / k
        k2 = 1.0
    else:
        k1 = 1.0
        k2 = k

    return k1, k2


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
# 3. DINÁMICA CON beta FIJA
#    Se usa para alcanzar el estado estacionario pre-upshift
# ============================================================

def rhs_beta_fija(y, beta_fija, k1, k2):
    """
    y = [R, x, N]
    """

    R = clamp_R(y[0])
    x = positive(y[1])
    N = positive(y[2])
    beta = max(beta_fija, 0.0)

    h = h_of_x(x, k1)
    g = g_of_x(x, k2)
    f = f_of_x(x, k1, k2)

    mu = R * g

    dR = R * g * (f - R)
    dx = beta * (1.0 - R) * h - R * g
    dN = (mu - gamma) * N - mu * N * N

    return np.array([dR, dx, dN], dtype=float)


def rk4_beta_fija(beta_fija, k1, k2, tmax, dt):
    """
    Integra con beta fija para alcanzar el estado estacionario.
    """

    n_steps = int(tmax / dt) + 1

    y = np.zeros((n_steps, 3), dtype=float)
    y[0] = np.array([R0, x0, N0], dtype=float)

    for n in range(n_steps - 1):
        yn = y[n]

        k_1 = rhs_beta_fija(yn, beta_fija, k1, k2)
        k_2 = rhs_beta_fija(yn + 0.5 * dt * k_1, beta_fija, k1, k2)
        k_3 = rhs_beta_fija(yn + 0.5 * dt * k_2, beta_fija, k1, k2)
        k_4 = rhs_beta_fija(yn + dt * k_3, beta_fija, k1, k2)

        y_next = yn + (dt / 6.0) * (k_1 + 2*k_2 + 2*k_3 + k_4)

        y_next[0] = clamp_R(y_next[0])
        y_next[1] = positive(y_next[1])
        y_next[2] = positive(y_next[2])

        y[n + 1] = y_next

    return y[-1]


# ============================================================
# 4. DINÁMICA POST-UPSHIFT
# ============================================================

def rhs_upshift(y, k1, k2):
    """
    y = [R, x, N, beta]
    """

    R = clamp_R(y[0])
    x = positive(y[1])
    N = positive(y[2])
    beta = max(y[3], 0.0)

    h = h_of_x(x, k1)
    g = g_of_x(x, k2)
    f = f_of_x(x, k1, k2)

    mu = R * g

    dR = R * g * (f - R)
    dx = beta * (1.0 - R) * h - R * g
    dN = (mu - gamma) * N - mu * N * N
    dbeta = -beta * N * h

    return np.array([dR, dx, dN, dbeta], dtype=float)


def rk4_upshift_desde_estado(y_pre, beta1, k1, k2, tmax, dt):
    """
    Integra la dinámica tras imponer beta(0)=beta1.

    y_pre = [R_pre, x_pre, N_pre]
    """

    n_steps = int(tmax / dt) + 1
    t = np.linspace(0.0, tmax, n_steps)

    y = np.zeros((n_steps, 4), dtype=float)

    y[0, 0] = y_pre[0]
    y[0, 1] = y_pre[1]
    y[0, 2] = y_pre[2]
    y[0, 3] = beta1

    for n in range(n_steps - 1):
        yn = y[n]

        k_1 = rhs_upshift(yn, k1, k2)
        k_2 = rhs_upshift(yn + 0.5 * dt * k_1, k1, k2)
        k_3 = rhs_upshift(yn + 0.5 * dt * k_2, k1, k2)
        k_4 = rhs_upshift(yn + dt * k_3, k1, k2)

        y_next = yn + (dt / 6.0) * (k_1 + 2*k_2 + 2*k_3 + k_4)

        y_next[0] = clamp_R(y_next[0])
        y_next[1] = positive(y_next[1])
        y_next[2] = positive(y_next[2])
        y_next[3] = max(y_next[3], 0.0)

        y[n + 1] = y_next

    R = y[:, 0]
    x = y[:, 1]
    N = y[:, 2]
    beta = y[:, 3]
    mu = R * g_of_x(x, k2)

    return t, R, x, N, beta, mu


# ============================================================
# 5. BARRIDO DOBLE EN k Y beta1
# ============================================================

mu_max = np.full((len(k_vals), len(beta1_vals)), np.nan)
R_variabilidad = np.full((len(k_vals), len(beta1_vals)), np.nan)

try:
    for i, k in enumerate(k_vals):

        print(f"Procesando estrategia {i+1}/{len(k_vals)} | k = {k:.4f}")

        k1, k2 = k_to_k1_k2(k)

        # 1. Estado estacionario previo con beta0_pre
        y_pre = rk4_beta_fija(
            beta_fija=beta0_pre,
            k1=k1,
            k2=k2,
            tmax=tmax_pre,
            dt=dt
        )

        # 2. Barrido de beta1 para esa estrategia k
        for j, beta1 in enumerate(beta1_vals):

            t, R, x, N, beta, mu = rk4_upshift_desde_estado(
                y_pre=y_pre,
                beta1=beta1,
                k1=k1,
                k2=k2,
                tmax=tmax_post,
                dt=dt
            )

            # Máximo growth rate tras el upshift
            mu_max[i, j] = np.max(mu)

            # Variabilidad de R durante el reajuste
            R_variabilidad[i, j] = np.max(R) - np.min(R)

            # Alternativa si prefieres variación respecto al valor inicial:
            # R_variabilidad[i, j] = np.max(np.abs(R - R[0]))

except KeyboardInterrupt:
    print("\nSimulación interrumpida.")
    print("Se graficarán los datos calculados hasta ahora.")


# ============================================================
# 6. MALLA PARA LOS MAPAS DE COLOR
# ============================================================

BETA_GRID, K_GRID = np.meshgrid(beta1_vals, k_vals)

# Enmascarar posibles NaN si se interrumpe la simulación
mu_max_masked = np.ma.masked_invalid(mu_max)
R_var_masked = np.ma.masked_invalid(R_variabilidad)


# ============================================================
# 7. GRÁFICA 1:
#    MÁXIMO GROWTH RATE EN FUNCIÓN DE k Y beta1
# ============================================================

fig, ax = plt.subplots(figsize=(10, 7))

pcm = ax.pcolormesh(
    BETA_GRID,
    K_GRID,
    mu_max_masked,
    shading="auto",
    cmap="viridis"
)

ax.set_yscale("log")

cbar = plt.colorbar(pcm, ax=ax)
cbar.set_label(r"Máximo growth rate, $\max_t \widetilde{\mu}(t)$", fontsize=FS_CBAR)
cbar.ax.tick_params(labelsize=FS_TICKS)

ax.set_xlabel(r"Upshift nutricional, $\widetilde{\beta}_1$", fontsize=FS_LABEL)
ax.set_ylabel(r"Estrategia metabólica, $k$", fontsize=FS_LABEL)

ax.set_title(
    rf"Máximo growth rate tras upshift desde $\widetilde{{\beta}}_0={beta0_pre}$",
    fontsize=FS_TITLE
)

ax.tick_params(axis="both", labelsize=FS_TICKS)
ax.grid(alpha=0.25, which="both")

plt.tight_layout()
plt.show()


# ============================================================
# 8. GRÁFICA 2:
#    VARIABILIDAD DE R EN FUNCIÓN DE k Y beta1
# ============================================================

fig, ax = plt.subplots(figsize=(10, 7))

pcm = ax.pcolormesh(
    BETA_GRID,
    K_GRID,
    R_var_masked,
    shading="auto",
    cmap="plasma"
)

ax.set_yscale("log")

cbar = plt.colorbar(pcm, ax=ax)
cbar.set_label(r"Variabilidad de $R$, $\Delta R$", fontsize=FS_CBAR)
cbar.ax.tick_params(labelsize=FS_TICKS)

ax.set_xlabel(r"Upshift nutricional, $\widetilde{\beta}_1$", fontsize=FS_LABEL)
ax.set_ylabel(r"Estrategia metabólica, $k$", fontsize=FS_LABEL)

ax.set_title(
    rf"Variabilidad de $R$ tras upshift desde $\widetilde{{\beta}}_0={beta0_pre}$",
    fontsize=FS_TITLE
)

ax.tick_params(axis="both", labelsize=FS_TICKS)
ax.grid(alpha=0.25, which="both")

plt.tight_layout()
plt.show()


# ============================================================
# 9. GUARDAR LAS FIGURAS, OPCIONAL
# ============================================================
# Si quieres guardar las figuras, cambia plt.show() por plt.savefig(...)
# o añade savefig antes de cada show.
#
# Ejemplo:
# plt.savefig("mapa_mu_max.png", dpi=300, bbox_inches="tight")
# plt.savefig("mapa_variabilidad_R.png", dpi=300, bbox_inches="tight")


# ============================================================
# 10. RESULTADOS DE CONTROL
# ============================================================

print("\nBarrido terminado.")
print("Dimensiones de los resultados:")
print("mu_max.shape =", mu_max.shape)
print("R_variabilidad.shape =", R_variabilidad.shape)

print("\nEjemplo de valores:")
print("k, beta1, mu_max, R_variabilidad")

indices_k = [0, len(k_vals)//2, -1]
indices_beta = [0, len(beta1_vals)//2, -1]

for i in indices_k:
    for j in indices_beta:
        print(
            f"{k_vals[i]:.5f}, "
            f"{beta1_vals[j]:.5f}, "
            f"{mu_max[i, j]:.6f}, "
            f"{R_variabilidad[i, j]:.6f}"
        )