import numpy as np
import matplotlib.pyplot as plt

# ============================================
# KONSTANTEN (REAL!)
# ============================================

G = 6.674e-11
x1 = float(input("Anzahl Erdmassen die der erste Planet wiegt [Standard 1]") or 1)
M1 = x1 * 5.972e24   # Masse je Planet (z.B. Erde)
x2 = float(input("Anzahl Erdmassen die der zweite Planet wiegt [Standard 1]") or 1)
M2 = x2 * 5.972e24
# Abstand der Planeten (z.B. 2 * Erdradius)
x3 = float(input("Abstand der beiden Planeten [Standard 4 Erdradien]") or 4) 
d = x3 * 6.371e6
#6.371e6  

r1 = np.array([-d/2, 0.0])
r2 = np.array([ d/2, 0.0])

# ============================================
# ANFANGSBEDINGUNGEN MOND
# ============================================

r0 = np.array([0.0, 2.0e7])   # Start oberhalb
v0 = np.array([3000.0, 0.0])  # vorsichtig wählen!

# ============================================
# SIMULATION
# ============================================

dt = float(input("Zeitintervall [Standard 0.5]") or 0.5) 
t_max = float(input("Zeitdauer [Standard 20000]") or 20000)

n_steps = int(t_max / dt)

# ============================================
# BESCHLEUNIGUNG
# ============================================

def acceleration(r):
    # zu Planet 1
    r_vec1 = r1 - r
    dist1 = np.linalg.norm(r_vec1)
    
    # zu Planet 2
    r_vec2 = r2 - r
    dist2 = np.linalg.norm(r_vec2)
    
    a1 = G * M1 * r_vec1 / dist1**3
    a2 = G * M2 * r_vec2 / dist2**3
    return a1 + a2

def f(y):
    r = y[:2]
    v = y[2:]
    a = acceleration(r)
    return np.array([v[0], v[1], a[0], a[1]])

# ============================================
# ARRAYS
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

plt.figure(figsize=(5,5))

# Bahnen
plt.plot(y_heun[:,0], y_heun[:,1], label="Mond")
#plt.plot(y_rk4[:,0], y_rk4[:,1], label="RK4")

# Planeten
plt.scatter(r1[0], r1[1], label="Planet 1")
plt.scatter(r2[0], r2[1], label="Planet 2")

plt.gca().set_aspect("equal")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Mondbahn im Doppelplanet-System")
plt.legend(loc="upper left")
plt.grid()

plt.savefig('cx_out/x.png', dpi=150)
