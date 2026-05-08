import numpy as np
import matplotlib.pyplot as plt

G = 6.674e-11

# ============================================
# MASSEN
# ============================================

x1 = float(input("Anzahl Erdmassen die der erste Planet wiegt [Standard 1]: ") or 1)
M1 = x1 * 5.972e24

x2 = float(input("Anzahl Erdmassen die der zweite Planet wiegt [Standard 1]: ") or 1)
M2 = x2 * 5.972e24

x3 = float(input("Anzahl Mondmassen die der Mond wiegt [Standard 1]: ") or 1)
M3 = x3 * 7.35e22

# ============================================
# ANFANGSPOSITIONEN
# ============================================

x4 = float(input("Abstand in Anzahl Erdradien [Standard 4]: ") or 4)
d = x4 * 6.371e6

x5 = float(input("Abstand Mond zu Systemmittelpunkt in E-M-Abständen [Standard 1]: ") or 1)
dmond = x5 * 2.0e7

r1 = np.array([-d/2, 0.0])
r2 = np.array([ d/2, 0.0])
r3 = np.array([0.0, dmond])

# ============================================
# ANFANGSGESCHWINDIGKEITEN
# ============================================

v = np.sqrt(G * (M1 + M2) / d) / 2
v_korr = (M1*v + M2*v)/(M1+M2)

v1 = np.array([0.0, -v + 0.5*v_korr])
v2 = np.array([0.0,  v - 0.5*v_korr])

v_mond = np.sqrt(G * (M1 + M2) / dmond)
v3 = np.array([v_mond, 0.0])

# ============================================
# ZEIT
# ============================================

dt = float(input("Zeitintervall [Standard 1]: ") or 1)
t_max = float(input("Zeitdauer [Standard 50000]: ") or 50000)
n_steps = int(t_max / dt)

# ============================================
# HILFSFUNKTIONEN
# ============================================

def accel(r_i, r_j, M_j):
    r_vec = r_j - r_i
    dist = np.linalg.norm(r_vec)
    return G * M_j * r_vec / dist**3

def f(y):
    r1, r2, r3 = y[:2], y[2:4], y[4:6]
    v1, v2, v3 = y[6:8], y[8:10], y[10:12]

    a1 = accel(r1, r2, M2) + accel(r1, r3, M3)
    a2 = accel(r2, r1, M1) + accel(r2, r3, M3)
    a3 = accel(r3, r1, M1) + accel(r3, r2, M2)

    return np.concatenate([v1, v2, v3, a1, a2, a3])

# ============================================
# ENERGIE
# ============================================

def total_energy(y):

    r1, r2, r3 = y[:2], y[2:4], y[4:6]
    v1, v2, v3 = y[6:8], y[8:10], y[10:12]

    T1 = 0.5 * M1 * np.dot(v1, v1)
    T2 = 0.5 * M2 * np.dot(v2, v2)
    T3 = 0.5 * M3 * np.dot(v3, v3)

    T = T1 + T2 + T3

    r12 = np.linalg.norm(r2 - r1)
    r13 = np.linalg.norm(r3 - r1)
    r23 = np.linalg.norm(r3 - r2)

    U = (-G*M1*M2/r12) + (-G*M1*M3/r13) + (-G*M2*M3/r23)

    return T + U

# ============================================
# INITIALISIERUNG
# ============================================

y0 = np.concatenate([r1, r2, r3, v1, v2, v3])

y_heun = np.zeros((n_steps, 12))
y_euler = np.zeros((n_steps, 12))

y_heun[0] = y0
y_euler[0] = y0

# Energie + Stabilität
energy_heun = np.zeros(n_steps)
energy_euler = np.zeros(n_steps)

rel_heun = np.zeros(n_steps)
rel_euler = np.zeros(n_steps)

energy_heun[0] = total_energy(y_heun[0])
energy_euler[0] = total_energy(y_euler[0])

E0_heun = energy_heun[0]
E0_euler = energy_euler[0]

rel_heun[0] = 0
rel_euler[0] = 0

# ============================================
# ZEITSCHLEIFE
# ============================================

for i in range(n_steps - 1):

    # ---------- HEUN ----------
    k1 = f(y_heun[i])
    y_pred = y_heun[i] + dt * k1
    k2 = f(y_pred)

    y_heun[i+1] = y_heun[i] + dt * 0.5 * (k1 + k2)

    energy_heun[i+1] = total_energy(y_heun[i+1])
    rel_heun[i+1] = (energy_heun[i+1] - E0_heun) / abs(E0_heun)

    # ---------- EULER ----------
    y_euler[i+1] = y_euler[i] + dt * f(y_euler[i])

    energy_euler[i+1] = total_energy(y_euler[i+1])
    rel_euler[i+1] = (energy_euler[i+1] - E0_euler) / abs(E0_euler)

print("Simulation fertig!")

# ============================================
# BAHNEN
# ============================================

plt.figure(figsize=(8,8))

plt.plot(y_heun[:,0], y_heun[:,1], label="Planet 1 (Heun)")
plt.plot(y_heun[:,2], y_heun[:,3], label="Planet 2 (Heun)")
plt.plot(y_heun[:,4], y_heun[:,5], label="Mond (Heun)")

plt.gca().set_aspect("equal")
plt.title("Heun")
plt.legend()
plt.grid()
plt.savefig("cx_out/heun.png", dpi=150)

plt.figure(figsize=(8,8))

plt.plot(y_euler[:,0], y_euler[:,1], label="Planet 1 (Euler)")
plt.plot(y_euler[:,2], y_euler[:,3], label="Planet 2 (Euler)")
plt.plot(y_euler[:,4], y_euler[:,5], label="Mond (Euler)")

plt.gca().set_aspect("equal")
plt.title("Euler")
plt.legend()
plt.grid()
plt.savefig("cx_out/euler.png", dpi=150)

# ============================================
# STABILITÄT (ENERGIE-DRIFT)
# ============================================

plt.figure(figsize=(10,5))

plt.plot(rel_heun, label="Heun")
plt.plot(rel_euler, label="Euler")

plt.xlabel("Zeitschritt")
plt.ylabel("relative Energieänderung")

plt.title("Stabilitätsanalyse")

plt.legend()
plt.grid()
plt.savefig("cx_out/stabilitaet.png", dpi=150)
