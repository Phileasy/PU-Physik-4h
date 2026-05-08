import numpy as np
import matplotlib.pyplot as plt

# ============================================
# PHYSIKALISCHE KONSTANTEN (REAL!)
# ============================================

G = 6.674e-11              # Gravitationskonstante
M = 5.972e24              # Masse Erde (kg)

# ============================================
# ANFANGSBEDINGUNGEN (z.B. Satellit)
# ============================================
x1 = float(input("Abstand des Mondes zum Erdzentrum in m [Standard 7.0e6]" )or 7.0e6)
r0 = np.array([x1, 0.0])      # 7000 km vom Erdzentrum
x2 = float(input("Orbitgeschwindigkeit in m/s [Standard 7500]" )or 7500)
v0 = np.array([0.0, x2])     # ca. Orbitgeschwindigkeit (m/s)

# ============================================
# SIMULATIONSPARAMETER
# ============================================

dt = float(input("Zeitintervall [Standard 0.1]") or 0.1)         # !!! sehr klein !!!
t_max = float(input("Zeitdauer [Standard 6000]") or 6000)    # ~1.5 Stunden

n_steps = int(t_max / dt)

# ============================================
# BESCHLEUNIGUNG
# ============================================

def acceleration(r):
    dist = np.linalg.norm(r)
    return -G * M * r / dist**3

# Zustand: [x, y, vx, vy]
def f(y):
    r = y[:2]
    v = y[2:]
    a = acceleration(r)
    return np.array([v[0], v[1], a[0], a[1]])

# ============================================
# INITIALISIERUNG
# ============================================

y_heun = np.zeros((n_steps, 4))
y_rk4 = np.zeros((n_steps, 4))

y0_vec = np.array([r0[0], r0[1], v0[0], v0[1]])

y_heun[0] = y0_vec
y_rk4[0] = y0_vec

# ============================================
# ZEITSCHLEIFE
# ============================================

for i in range(n_steps - 1):
    
    # ---------- HEUN ----------
    k1 = f(y_heun[i])
    y_pred = y_heun[i] + dt * k1
    k2 = f(y_pred)
    
    y_heun[i+1] = y_heun[i] + dt * 0.5 * (k1 + k2)
    
    # ---------- RK4 ----------
    k1 = f(y_rk4[i])
    k2 = f(y_rk4[i] + dt/2 * k1)
    k3 = f(y_rk4[i] + dt/2 * k2)
    k4 = f(y_rk4[i] + dt * k3)
    
    y_rk4[i+1] = y_rk4[i] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

print("Simulation fertig!")

# ============================================
# PLOT
# ============================================

plt.figure(figsize=(4,4))

plt.plot(y_heun[:,0], y_heun[:,1], label="Mond")
#plt.plot(y_rk4[:,0], y_rk4[:,1], label="RK4")

# Erde
plt.scatter(0, 0, label="Erde")

plt.gca().set_aspect("equal")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Mondbahn ")
plt.legend(loc="upper left")
plt.grid()

plt.savefig('cx_out/dsfgdf.png', dpi=150)
