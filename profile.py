"""
Generate profile.png: Mach number and velocity along the nozzle.
Standalone (numpy + matplotlib only). Same physics as the main sim.
"""
import numpy as np
import matplotlib.pyplot as plt

GAMMA, R_S = 1.20, 320.0                       # gas properties
AR_CHAMBER, AR_EXIT, X_THROAT = 4.0, 4.0, 0.40  # nozzle geometry

def area_ratio(x):
    x = np.asarray(x, float)
    conv = AR_CHAMBER - (AR_CHAMBER - 1) * (x / X_THROAT)
    div  = 1 + (AR_EXIT - 1) * ((x - X_THROAT) / (1 - X_THROAT))**2
    return np.where(x < X_THROAT, conv, div)

def area_mach(M):
    g = GAMMA
    return (1/M) * ((2/(g+1)) * (1 + 0.5*(g-1)*M**2))**((g+1)/(2*(g-1)))

_Msub, _Msup = np.linspace(1e-4, 1, 20000), np.linspace(1, 6, 20000)
_ARsub, _ARsup = area_mach(_Msub), area_mach(_Msup)

# Solve the field at T0 = 3000 K
x  = np.linspace(0, 1, 400)
ar = area_ratio(x)
M  = np.where(x < X_THROAT,
              np.interp(ar, _ARsub[::-1], _Msub[::-1]),
              np.interp(ar, _ARsup, _Msup))
T  = 3000.0 / (1 + 0.5*(GAMMA-1)*M**2)
u  = M * np.sqrt(GAMMA * R_S * T)

# Plot: Mach (left axis) and velocity (right axis), sharing x
fig, ax1 = plt.subplots(figsize=(8, 3.6))
fig.patch.set_facecolor("white")
ax1.fill_between(x, 0, np.sqrt(ar)/np.sqrt(ar).max()*3, color="#eee", zorder=0)  # nozzle silhouette
ax1.axvline(X_THROAT, color="#bbb", ls="--", lw=1)

c1, c2 = "#d9480f", "#1971c2"
l1 = ax1.plot(x, M, color=c1, lw=2.5, label="Mach number M(x)")[0]
ax1.axhline(1, color=c1, ls=":", lw=1)
ax1.set_ylabel("Mach number", color=c1); ax1.tick_params(axis="y", labelcolor=c1)
ax1.set_xlabel("position along nozzle  x")

ax2 = ax1.twinx()
l2 = ax2.plot(x, u, color=c2, lw=2.5, label="velocity u(x) [m/s]")[0]
ax2.set_ylabel("velocity  u  [m/s]", color=c2); ax2.tick_params(axis="y", labelcolor=c2)

ax1.text(X_THROAT, 0.1, " throat", color="#888", fontsize=9)
ax1.text(0.05, 2.3, "converging\n(subsonic)", fontsize=8, color="#666")
ax1.text(0.72, 2.3, "diverging\n(supersonic)", fontsize=8, color="#666")
ax1.set_title("Mach number and velocity by position along nozzle", fontsize=11)
ax1.legend(handles=[l1, l2], loc="upper left", fontsize=9, framealpha=0.9)

plt.tight_layout()
plt.savefig("profile.png", dpi=110, facecolor="white")
print("saved profile.png")