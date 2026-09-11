"""
  fig_hotter.png  : hotter chamber -> faster exhaust  (velocity field at 3 temps)
  fig_scaling.png : exit velocity scales as sqrt(T0)   (closed-form, with reference)
"""
import numpy as np
import matplotlib.pyplot as plt

GAMMA, R_S = 1.20, 320.0          # gas: ratio of specific heats, specific gas constant
AR_CHAMBER, AR_EXIT, X_THROAT = 4.0, 4.0, 0.40

def area_ratio(x):
    x = np.asarray(x, float)
    conv = AR_CHAMBER - (AR_CHAMBER - 1) * (x / X_THROAT)
    div  = 1 + (AR_EXIT - 1) * ((x - X_THROAT) / (1 - X_THROAT))**2
    return np.where(x < X_THROAT, conv, div)

def area_mach(M):
    g = GAMMA
    return (1/M) * ((2/(g+1)) * (1 + 0.5*(g-1)*M**2))**((g+1)/(2*(g-1)))

_Msub = np.linspace(1e-4, 1, 20000); _Msup = np.linspace(1, 6, 20000)
_ARsub, _ARsup = area_mach(_Msub), area_mach(_Msup)

def solve_u(T0):
    x  = np.linspace(0, 1, 400); ar = area_ratio(x)
    M  = np.where(x < X_THROAT,
                  np.interp(ar, _ARsub[::-1], _Msub[::-1]),
                  np.interp(ar, _ARsup, _Msup))
    T  = T0 / (1 + 0.5*(GAMMA-1)*M**2)
    return x, M * np.sqrt(GAMMA * R_S * T), M[-1]

# ==============================================================================
# Diagram 1: hotter chamber -> faster exhaust
# ==============================================================================
fig, ax = plt.subplots(figsize=(6.5, 4))
for T0, c in zip([2000, 3000, 4000], ["#f59f00", "#e8590c", "#c92a2a"]):
    x, u, _ = solve_u(T0)
    ax.plot(x, u, color=c, lw=2.5, label=f"$T_0$ = {T0} K")
ax.axvline(X_THROAT, color="#bbb", ls="--", lw=1)
ax.text(X_THROAT, ax.get_ylim()[1]*0.05, " throat", color="#888", fontsize=9)
ax.set_xlabel("position along nozzle  $x$")
ax.set_ylabel("velocity  $u(x)$  [m/s]")
ax.set_title("Hotter chamber → faster exhaust")
ax.legend()
plt.tight_layout(); plt.savefig("fig_hotter.png", dpi=120); plt.close()

# ==============================================================================
# Diagram 2: exit velocity scales as sqrt(T0).
#   Each u_e is taken from the FULL field solver (M_e * a_e), independently per T0.
#   Overlay a sqrt(T0) reference anchored at ONE point.
# ==============================================================================
T0s = np.linspace(1500, 4500, 25)
u_e = np.array([solve_u(t)[1][-1] for t in T0s])          # field-solver exit velocity
ref = u_e[0] * np.sqrt(T0s / T0s[0])                       # sqrt-law anchored at lowest T0

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(T0s, u_e, "o", color="#e8590c", ms=6, label="field solver  $u_e$")
ax.plot(T0s, ref, "k:", lw=1.4, label=r"$\sqrt{T_0}$ law (anchored at 1500 K)")
for T0, c in zip([2000, 3000, 4000], ["#f59f00", "#e8590c", "#c92a2a"]):
    ax.scatter([T0], [solve_u(T0)[1][-1]], color=c, s=70, zorder=5, edgecolor="k", lw=0.5)
ax.set_xlabel("chamber temperature  $T_0$  [K]   (the combustion knob)")
ax.set_ylabel("exit velocity  $u_e$  [m/s]")
ax.set_title(r"Exit velocity scales as $\sqrt{T_0}$")
ax.legend()
plt.tight_layout(); plt.savefig("fig_scaling.png", dpi=120); plt.close()

dev = np.max(np.abs(u_e - ref) / ref) * 100
print(f"exit Mach (fixed by geometry) = {solve_u(3000.0)[2]:.3f}")
print(f"field-solver u_e at 3000 K    = {solve_u(3000.0)[1][-1]:.1f} m/s")
print(f"max deviation from sqrt(T0)    = {dev:.2e} %   (why: M(x) is set by geometry, so u ∝ sqrt(T0))")