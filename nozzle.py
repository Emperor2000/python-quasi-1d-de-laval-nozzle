"""
Simulating a Rocket Nozzle in ~150 Lines of Python
---------------------------------------------------
We SOLVE the steady quasi-1D isentropic flow field u(x) in a de Laval nozzle,
then SIMULATE tracer particles advecting through it. The particles accelerate
through the throat and go supersonic in the *widening* section because they
integrate a physically correct velocity field -- not because we keyframed them.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# ----------------------------------------------------------------------
# 1. Gas + chamber. Combustion enters ONLY as stagnation conditions.
# ----------------------------------------------------------------------
GAMMA = 1.20        # ratio of specific heats, hot combustion products
R_S   = 320.0       # specific gas constant, J/(kg K)  (~26 g/mol)
T0    = 3000.0      # chamber stagnation temperature, K  <- combustion knob
P0    = 7.0e6       # chamber stagnation pressure, Pa

# ----------------------------------------------------------------------
# 2. Nozzle geometry: area ratio A(x)/A* along the axis, min (=1) at throat.
# ----------------------------------------------------------------------
AR_CHAMBER, AR_EXIT, X_THROAT = 4.0, 4.0, 0.40

def area_ratio(x):
    x = np.asarray(x, float)
    conv = AR_CHAMBER - (AR_CHAMBER - 1.0) * (x / X_THROAT)
    div  = 1.0 + (AR_EXIT - 1.0) * ((x - X_THROAT) / (1 - X_THROAT))**2
    return np.where(x < X_THROAT, conv, div)

# ----------------------------------------------------------------------
# 3. Area-Mach relation and its inversion by table lookup (two branches).
# ----------------------------------------------------------------------
def area_mach(M):
    g = GAMMA
    return (1/M) * ((2/(g+1)) * (1 + 0.5*(g-1)*M**2))**((g+1)/(2*(g-1)))

_M_sub = np.linspace(1e-4, 1.0, 20000)
_M_sup = np.linspace(1.0, 6.0, 20000)
_AR_sub, _AR_sup = area_mach(_M_sub), area_mach(_M_sup)

def mach_from_area(ar, supersonic):
    if supersonic:
        return np.interp(ar, _AR_sup, _M_sup)
    return np.interp(ar, _AR_sub[::-1], _M_sub[::-1])

# ----------------------------------------------------------------------
# 4. Solve the field: M(x) -> T, a, u.  Radius for drawing ~ sqrt(area).
# ----------------------------------------------------------------------
def solve_field(T0):
    x  = np.linspace(0, 1, 400)
    ar = area_ratio(x)
    M  = np.where(x < X_THROAT,
                  mach_from_area(ar, supersonic=False),
                  mach_from_area(ar, supersonic=True))
    T  = T0 / (1 + 0.5*(GAMMA-1)*M**2)
    a  = np.sqrt(GAMMA * R_S * T)
    u  = M * a
    r  = np.sqrt(ar)                      # radius proportional to sqrt(area)
    return x, ar, M, T, u, r

x, ar, M, T, u, r = solve_field(T0)
R = r / r.max() * 0.9                     # normalized half-height for display

# ----------------------------------------------------------------------
# 5. Output
# ----------------------------------------------------------------------
P    = P0 * (T/T0)**(GAMMA/(GAMMA-1))
rho  = P / (R_S * T)
flux = rho * u * ar                       # rho*u*A/A* must be constant
M_e  = M[-1]
cp   = GAMMA * R_S / (GAMMA - 1)
Pe_P0 = (1 + 0.5*(GAMMA-1)*M_e**2)**(-GAMMA/(GAMMA-1))
v_e  = np.sqrt(2*cp*T0*(1 - Pe_P0**((GAMMA-1)/GAMMA)))
print(f"throat Mach                 = {M[np.argmin(ar)]:.4f}   (target 1.0)")
print(f"exit Mach                   = {M_e:.3f}")
print(f"solver exit u  vs  v_e      = {u[-1]:.1f}  vs  {v_e:.1f} m/s")
print(f"mass-flux spread over field = {(flux.max()-flux.min())/flux.mean()*100:.2e} %")

# ----------------------------------------------------------------------
# 6. Particles: each has an axial position px and a fixed radial fraction f.
#    Advect by the real velocity field; recycle at the exit.
# ----------------------------------------------------------------------
N = 1400
rng = np.random.default_rng(0)
px = rng.uniform(0, 1, N)
pf = rng.uniform(-1, 1, N)                 # radial fraction in [-1, 1]
u_max = u.max()
STEP = 0.010                               # visual pacing

def advect():
    global px, pf
    px += np.interp(px, x, u) / u_max * STEP
    out = px > 1.0
    px[out] = 0.0                          # recycle to chamber
    pf[out] = rng.uniform(-1, 1, out.sum())

def particle_xy_c():
    rp = np.interp(px, x, R)
    return px, pf * rp, np.interp(px, x, u) / u_max   # x, y, color value

# ----------------------------------------------------------------------
# 7. Figure: nozzle walls + velocity-colored particles.
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 3.2))
fig.patch.set_facecolor("#0a0a0f"); ax.set_facecolor("#0a0a0f")
ax.fill_between(x,  R,  1.05, color="#15151d")
ax.fill_between(x, -R, -1.05, color="#15151d")
ax.plot(x,  R, color="#3a3a4a", lw=1.5)
ax.plot(x, -R, color="#3a3a4a", lw=1.5)
ax.axvline(X_THROAT, color="#333", lw=0.8, ls="--")
ax.text(X_THROAT, 1.0, " throat  M=1", color="#888", fontsize=8, va="top")
scat = ax.scatter([], [], s=6, c=[], cmap="inferno", vmin=0, vmax=1)
ax.set_xlim(0, 1); ax.set_ylim(-1.1, 1.1); ax.axis("off")

def frame(_):
    advect()
    xx, yy, cc = particle_xy_c()
    scat.set_offsets(np.column_stack([xx, yy]))
    scat.set_array(cc)
    return scat,

# ----------------------------------------------------------------------
# 8. Render to GIF.
# ----------------------------------------------------------------------
anim = FuncAnimation(fig, frame, frames=200, interval=40, blit=True)
anim.save("nozzle.gif", writer=PillowWriter(fps=25), dpi=90)
print("saved nozzle.gif")
