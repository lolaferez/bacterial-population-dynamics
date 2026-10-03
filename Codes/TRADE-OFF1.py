import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# MODELO RAM SIN DINÁMICA DE N Y SIN DINÁMICA DE beta
# ============================================================
# Variables dinámicas:
#   R(t) : fracción ribosomal
#   x(t) : sustrato interno
#
# Parámetro externo:
#   beta : calidad nutricional del medio, constante en cada simulación
#
# Sistema:
#   dR/dt = R g(x) [f(x) - R]
#   dx/dt = beta (1 - R) h(x) - R g(x)
#   mu    = R g(x)
#
# Protocolo:
#   1) Para cada k, relajar con beta0 = 0.5 hasta estado estacionario.
#   2) Aplicar un upshift brusco beta0 -> beta1.
#   3) Medir max_t mu(t) y Delta R tras el upshift.
#   4) Representar mapas de calor en el plano (beta1, k).
# ============================================================


# ============================================================
# 1. PARÁMETROS GENERALES
# ============================================================

# Condición basal pre-upshift
beta0 = 0.5

# Barrido de beta1 tras el upshift
beta1_vals = np.linspace(0.55, 30, 100)

# Barrido de estrategias k
k_vals = np.logspace(np.log10(0.0001), np.log10(100), 85)

# Condiciones iniciales comunes
R0 = 0.2
x0 = 0.1

# Tiempos de simulación
tmax_relax = 100.0      # tiempo para alcanzar estado estacionario con beta0
tmax_upshift = 100.0     # tiempo tras el upshift

# Paso temporal
dt = 0.05

# Tamaños de letra grandes
FS_TITLE = 24
FS_LABEL = 22
FS_TICKS = 17
FS_CBAR = 19


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
        k = 0.001  -> k1 = 1000, k2 = 1
        k = 1      -> k1 = 1,    k2 = 1
        k = 1000   -> k1 = 1,    k2 = 1000
    """
    if k < 1.0:
        k1 = 1.0 / k
        k2 = 1.0
    else:
        k1 = 1.0
        k2 = k

    return k1, k2


def h_of_x(x, k1):
    """
    Función de importación/inhibición catabólica.
    """
    x = positive(x)
    return k1 / (k1 + x)


def g_of_x(x, k2):
    """
    Actividad ribosomal.
    """
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
    Función reguladora óptima:

        f(x) = 1 / [1 - (g'/g)(h/h')]

    Se recorta entre 0 y 1 para mantener interpretación biológica.
    """
    x = positive(x)

    h = h_of_x(x, k1)
    g = g_of_x(x, k2)
    hp = h_prim_of_x(x, k1)
    gp = g_prim_of_x(x, k2)

    f = 1.0 / (1.0 - (gp / g) * (h / hp))

    return np.clip(f, 0.0, 1.0)


# ============================================================
# 3. DINÁMICA CON beta CONSTANTE
# ============================================================

def rhs(y, beta, k1, k2):
    """
    Sistema dinámico para una sola población.

    y = [R, x]

    beta NO es variable dinámica.
    beta es un parámetro constante durante cada simulación.
    """

    R = clamp_R(y[0])
    x = positive(y[1])
    beta = max(beta, 0.0)

    h = h_of_x(x, k1)
    g = g_of_x(x, k2)
    f = f_of_x(x, k1, k2)

    dR = R * g * (f - R)
    dx = beta * (1.0 - R) * h - R * g

    return np.array([dR, dx], dtype=float)


def rk4(y0, beta, k1, k2, tmax, dt):
    """
    Integra el sistema con Runge-Kutta de orden 4.
    """

    n_steps = int(tmax / dt) + 1
    t = np.linspace(0.0, tmax, n_steps)

    y = np.zeros((n_steps, 2), dtype=float)
    y[0] = y0

    for n in range(n_steps - 1):
        yn = y[n]

        q1 = rhs(yn, beta, k1, k2)
        q2 = rhs(yn + 0.5 * dt * q1, beta, k1, k2)
        q3 = rhs(yn + 0.5 * dt * q2, beta, k1, k2)
        q4 = rhs(yn + dt * q3, beta, k1, k2)

        y_next = yn + (dt / 6.0) * (q1 + 2.0*q2 + 2.0*q3 + q4)

        # Restricciones físicas
        y_next[0] = clamp_R(y_next[0])
        y_next[1] = positive(y_next[1])

        y[n + 1] = y_next

    R = y[:, 0]
    x = y[:, 1]
    mu = R * g_of_x(x, k2)

    return t, R, x, mu


# ============================================================
# 4. BARRIDO DOBLE EN k Y beta1
# ============================================================

mu_max = np.full((len(k_vals), len(beta1_vals)), np.nan)
delta_R = np.full((len(k_vals), len(beta1_vals)), np.nan)

R_ss_vals = np.full(len(k_vals), np.nan)
x_ss_vals = np.full(len(k_vals), np.nan)
mu_ss_vals = np.full(len(k_vals), np.nan)

try:
    for i, k in enumerate(k_vals):

        print(f"Procesando estrategia {i+1}/{len(k_vals)} | k = {k:.5f}")

        k1, k2 = k_to_k1_k2(k)

        # ----------------------------------------------------
        # 1. Estado estacionario con beta0 = 0.5
        # ----------------------------------------------------

        y0_relax = np.array([R0, x0], dtype=float)

        t_relax, R_relax, x_relax, mu_relax = rk4(
            y0=y0_relax,
            beta=beta0,
            k1=k1,
            k2=k2,
            tmax=tmax_relax,
            dt=dt
        )

        R_ss = R_relax[-1]
        x_ss = x_relax[-1]
        mu_ss = mu_relax[-1]

        R_ss_vals[i] = R_ss
        x_ss_vals[i] = x_ss
        mu_ss_vals[i] = mu_ss

        y_ss = np.array([R_ss, x_ss], dtype=float)

        # ----------------------------------------------------
        # 2. Upshift brusco beta0 -> beta1
        # ----------------------------------------------------

        for j, beta1 in enumerate(beta1_vals):

            t_up, R_up, x_up, mu_up = rk4(
                y0=y_ss,
                beta=beta1,
                k1=k1,
                k2=k2,
                tmax=tmax_upshift,
                dt=dt
            )

            # Pico máximo de crecimiento tras el upshift
            mu_max[i, j] = np.max(mu_up)

            # Variación absoluta de R durante el reajuste
            delta_R[i, j] = (np.max(R_up) - np.min(R_up))/np.max(R_up)

            # Alternativa si prefieres respecto al estado inicial:
            # delta_R[i, j] = np.max(np.abs(R_up - R_up[0]))

except KeyboardInterrupt:
    print("\nSimulación interrumpida por el usuario.")
    print("Se graficarán los datos calculados hasta ahora.")


# ============================================================
# 5. MALLA PARA LOS MAPAS DE COLOR
# ============================================================

BETA_GRID, K_GRID = np.meshgrid(beta1_vals, k_vals)

mu_max_masked = np.ma.masked_invalid(mu_max)
delta_R_masked = np.ma.masked_invalid(delta_R)


# ============================================================
# 6. MAPA DE COLOR 1:
#    PICO MÁXIMO DE mu TRAS EL UPSHIFT
# ============================================================

fig, ax = plt.subplots(figsize=(11, 7.5))

pcm = ax.pcolormesh(
    BETA_GRID,
    K_GRID,
    mu_max_masked,
    shading="auto",
    cmap="viridis"
)

ax.set_yscale("log")

cbar = plt.colorbar(pcm, ax=ax)
cbar.set_label(
    r"Pico máximo de crecimiento, $\max_t \widetilde{\mu}(t)$",
    fontsize=FS_CBAR
)
cbar.ax.tick_params(labelsize=FS_TICKS)

ax.set_xlabel(
    r"Upshift nutricional, $\widetilde{\beta}_1$",
    fontsize=FS_LABEL
)
ax.set_ylabel(
    r"Estrategia metabólica, $k$",
    fontsize=FS_LABEL
)

ax.set_title(
    rf"Máximo growth rate tras upshift desde $\widetilde{{\beta}}_0={beta0}$",
    fontsize=FS_TITLE
)

ax.tick_params(axis="both", labelsize=FS_TICKS)
ax.grid(alpha=0.25, which="both")

plt.tight_layout()

# Guardar opcionalmente
plt.savefig("mapa_mu_max_sin_N_sin_beta_dinamica.png", dpi=300, bbox_inches="tight")

plt.show()


# ============================================================
# 7. MAPA DE COLOR 2:
#    VARIACIÓN ABSOLUTA DE R TRAS EL UPSHIFT
# ============================================================

fig, ax = plt.subplots(figsize=(11, 7.5))

pcm = ax.pcolormesh(
    BETA_GRID,
    K_GRID,
    delta_R_masked,
    shading="auto",
    cmap="plasma"
)

ax.set_yscale("log")

cbar = plt.colorbar(pcm, ax=ax)
cbar.set_label(
    r"Variación ribosomal relativa, $\Delta R = (R_{\max}-R_{\min})/R_{\max}$",
    fontsize=FS_CBAR
)
cbar.ax.tick_params(labelsize=FS_TICKS)

ax.set_xlabel(
    r"Upshift nutricional, $\widetilde{\beta}_1$",
    fontsize=FS_LABEL
)
ax.set_ylabel(
    r"Estrategia metabólica, $k$",
    fontsize=FS_LABEL
)

ax.set_title(
    rf"Variabilidad de $R$ tras upshift desde $\widetilde{{\beta}}_0={beta0}$",
    fontsize=FS_TITLE
)

ax.tick_params(axis="both", labelsize=FS_TICKS)
ax.grid(alpha=0.25, which="both")

plt.tight_layout()

# Guardar opcionalmente
plt.savefig("mapa_delta_R_sin_N_sin_beta_dinamica.png", dpi=300, bbox_inches="tight")

plt.show()


# ============================================================
# 8. RESULTADOS DE CONTROL
# ============================================================

print("\nBarrido terminado.")
print("Modelo de una sola población, sin N y sin beta dinámica.")
print("Cada punto del mapa representa una combinación (k, beta1).")
print("\nDimensiones:")
print("mu_max.shape =", mu_max.shape)
print("delta_R.shape =", delta_R.shape)

print("\nEjemplos:")
print("k, beta1, mu_max, delta_R")

indices_k = [0, len(k_vals)//2, -1]
indices_beta = [0, len(beta1_vals)//2, -1]

for i in indices_k:
    for j in indices_beta:
        print(
            f"{k_vals[i]:.6f}, "
            f"{beta1_vals[j]:.6f}, "
            f"{mu_max[i, j]:.6f}, "
            f"{delta_R[i, j]:.6f}"
        )