import numpy as np
import matplotlib.pyplot as plt

G = 6.674e-11

# ============================================
# MASSEN
# ============================================
x1 = float(input("Anzahl Erdmassen die der erste Planet wiegt [Standard 1]") or 1)
M1 = x1 * 5.972e24   # Masse je Planet (z.B. Erde)
x2 = float(input("Anzahl Erdmassen die der zweite Planet wiegt [Standard 1]") or 1)
M2 = x2 * 5.972e24
x3 = float(input("Anzahl Mondmassen die der Mond wiegt [Standard 1]") or 1)
M3 = x3 * 7.35e22   # kleiner Körper (Mond)

# ============================================
# ANFANGSPOSITIONEN
# ============================================
x4 = float(input("Abstand in Anzahl Erdradien [Standard 4]") or 4)
d = x4 * 6.371e6
x5 = float(input("Abstand Mond zu systemmittelpunkt in E-M-Abständen [Standard 1]") or 1)
dmond = x5 * 2.0e7
r1 = np.array([-d/2, 0.0])
r2 = np.array([ d/2, 0.0])
r3 = np.array([0.0, dmond])

# ============================================
# ANFANGSGESCHWINDIGKEITEN
# ============================================

# Kreisbewegung der beiden grossen Planeten um Schwerpunkt
v = np.sqrt(G * (M1 + M2) / d) / 2
v_korr = (M1*v + M2* v)/(M1+M2)
v1 = np.array([0.0, -v+0.5*v_korr])
v2 = np.array([0.0,  v-0.5*v_korr])

# Mond
v_mond = np.sqrt(G * (M1+M2)/ dmond)
v3 = np.array([v_mond, 0.0])

# ============================================
# ZEIT
# ============================================

dt = float(input("Zeitintervall [Standard 1]") or 1)
t_max = float(input("Zeitdauer [Standard 50000]") or 50000)
n_steps = int(t_max / dt)

# ============================================
# HILFSFUNKTION
# ============================================

def accel(r_i, r_j, M_j):
    r_vec = r_j - r_i
    dist = np.linalg.norm(r_vec)
    return G * M_j * r_vec / dist**3

def f(y):
    r1, r2, r3 = y[:2], y[2:4], y[4:6]
    v1, v2, v3 = y[6:8], y[8:10], y[10:12]
    
    # Beschleunigungen
    a1 = accel(r1, r2, M2) + accel(r1, r3, M3)
    a2 = accel(r2, r1, M1) + accel(r2, r3, M3)
    a3 = accel(r3, r1, M1) + accel(r3, r2, M2)
    
    return np.concatenate([v1, v2, v3, a1, a2, a3])
# ============================================
# INITIALISIERUNG
# ============================================
y0 = np.concatenate([r1, r2, r3, v1, v2, v3])
y_heun = np.zeros((n_steps, 12))
y_rk4  = np.zeros((n_steps, 12))
y_heun[0] = y0
y_rk4[0]  = y0
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
plt.figure(figsize=(8,8))
# --- RK4 ---
#plt.plot(y_rk4[:,0], y_rk4[:,1], label="Planet 1 (RK4)")
#plt.plot(y_rk4[:,2], y_rk4[:,3], label="Planet 2 (RK4)")
#plt.plot(y_rk4[:,4], y_rk4[:,5], label="Mond (RK4)")
# --- HEUN (gestrichelt) ---
plt.plot(y_heun[:,0], y_heun[:,1], label="Planet 1 ")
plt.plot(y_heun[:,2], y_heun[:,3], label="Planet 2 ")
plt.plot(y_heun[:,4], y_heun[:,5], label="Mond ")
plt.gca().set_aspect("equal")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("2 bewegte Planeten und Mond")
plt.legend(loc="upper left")
plt.grid()

plt.savefig('cx_out/x.png', dpi=150)
