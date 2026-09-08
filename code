#An2
#Projet Analyse Numérique 2

import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.integrate import solve_ivp
import time


# Paramètres du modèle


params = {
    # Végétation Eq. (F1)
    'rV': 1.20, 'KV': 500,
    'alphaN': 15.0, 'nuN': 200,
    'alphaD': 8.0,  'nuD': 100,
    'alphaB': 3.0,  'nuB': 120,

    # Ongulé principal (wapiti/orignal) Eq. (F2)
    'rN': 0.30, 'KN': 7.0, 'V*': 70,
    'tetaN': 4.0, 'cWN': 7.5, 'beta': 0.5,
    'cBN': 0.5,  'hBN': 3.0, 'deltaN': 0.05,

    # Ongulé secondaire (cerf) Eq. (F3)
    'rD': 0.50, 'KD': 4.0, 'V**': 50,
    'tetaD': 2.0, 'cWD': 2.0, 'hWD': 2.0,
    'cBD': 0.3,  'hBD': 1.5, 'deltaD': 0.08,

    # Loups Eq. (F4)
    'eps1': 0.25, 'eps2': 0.5, 'eta': 0.01,
    'mW': 0.15, 'muW': 0.10,

    # Ours Eq. (F5)
    'eB': 0.12, 'eV': 0.02, 'mB': 0.08
}

# Système d'EDO

def F(t, u, params):
    V, N, D, W, B = u
    KN_V = params['KN'] * (V / (params['V*'] + V))
    KD_V = params['KD'] * (V / (params['V**'] + V))
    fWN  = params['cWN'] * N / (W + params['beta'] * N + 1e-10)
    fWD  = params['cWD'] * D / (params['hWD'] + D)
    phiW = fWN + fWD

    dV = (params['rV'] * V * (1 - V / params['KV'])
          - params['alphaN'] * V * N / (params['nuN'] + V)
          - params['alphaD'] * V * D / (params['nuD'] + V)
          - params['alphaB'] * V * B / (params['nuB'] + V))

    dN = (params['rN'] * N * (1 - (N / KN_V) ** params['tetaN'])
          - params['cWN'] * N * W / (W + params['beta'] * N + 1e-10)
          - params['cBN'] * N * B / (params['hBN'] + N)
          - params['deltaN'] * N)

    dD = (params['rD'] * D * (1 - (D / KD_V) ** params['tetaD'])
          - params['cWD'] * D * W / (params['hWD'] + D)
          - params['cBD'] * D * B / (params['hBD'] + D)
          - params['deltaD'] * D)

    dW = ((params['eps1'] * np.log(phiW + params['eta']) - params['eps2']) * W
          - params['mW'] * W
          - params['muW'] * (np.sin(np.pi * t) ** 2) * W)

    dB = ((params['eB'] * (params['cBN'] * N / (params['hBN'] + N)
                           + params['cBD'] * D / (params['hBD'] + D))
           + params['eV'] * params['alphaB'] * V / (params['nuB'] + V)
           - params['mB']) * B)

    return np.array([dV, dN, dD, dW, dB])



# Jacobienne par différences finies centrées

def Jacobienne_num(F_func, t, u, params, h_diff=1e-6):
    n = len(u)
    J = np.zeros((n, n))
    for i in range(n):
        u_plus  = np.copy(u); u_plus[i]  += h_diff
        u_minus = np.copy(u); u_minus[i] -= h_diff
        J[:, i] = (F_func(t, u_plus, params) - F_func(t, u_minus, params)) / (2 * h_diff)
    return J


# Euler explicite

def Euler_explicite(T, h, U0, F, params):
    n = int(T / h)
    U = np.zeros((n, 5)); U[0] = U0
    vectT = np.zeros(n)
    for i in range(1, n):
        U[i] = U[i-1] + h * F(vectT[i-1], U[i-1], params)
        vectT[i] = vectT[i-1] + h
    return vectT, U

# Euler implicite + Newton

def Euler_implicite_Newton(T, h, U0, F, params, eps=1e-8, Nmax=50):
    n_steps = int(T / h)
    U = np.zeros((n_steps, 5)); U[0] = U0
    vectT = np.linspace(0, T, n_steps)
    I = np.eye(5)
    total_iter = 0 

    for i in range(1, n_steps):
        t_next = vectT[i]
        u_prev = U[i-1]
        u_k    = np.maximum(u_prev + h * F(vectT[i-1], u_prev, params), 1e-10)
        for _ in range(Nmax):
            total_iter +=1
            F_eval = F(t_next, u_k, params)
            G      = u_k - u_prev - h * F_eval
            if np.linalg.norm(G) < eps:
                break
            J_G   = I - h * Jacobienne_num(F, t_next, u_k, params)
            try:
                delta = np.linalg.solve(J_G, G)
            except np.linalg.LinAlgError:
                delta = G * 0.1
            u_k = np.maximum(u_k - delta, 1e-10)
        U[i] = u_k
    return vectT, U, total_iter


# RK4

def RK4(T, h, U0, F, params):
    n = int(T / h)
    U = np.zeros((n, 5)); U[0] = U0
    vectT = np.zeros(n)
    for i in range(1, n):
        t  = vectT[i-1]
        ui = U[i-1]
        k1 = F(t,       ui,              params)
        k2 = F(t + h/2, ui + h*k1/2,    params)
        k3 = F(t + h/2, ui + h*k2/2,    params)
        k4 = F(t + h,   ui + h*k3,      params)
        U[i]     = ui + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
        vectT[i] = t + h
    return vectT, U


# Adams-Bashforth 4 (AB4)

def Adams_Bashforth4(T, h, U0, F, params):
    n = int(T / h)
    U = np.zeros((n, 5)); U[0] = U0
    vectT = np.linspace(0, T, n)

    # Les 3 premiers pas de temps doivent être fait par RK4
    for i in range(1, min(4, n)):
        t  = vectT[i-1]; ui = U[i-1]
        k1 = F(t,       ui,           params)
        k2 = F(t + h/2, ui + h*k1/2, params)
        k3 = F(t + h/2, ui + h*k2/2, params)
        k4 = F(t + h,   ui + h*k3,   params)
        U[i] = ui + (h/6) * (k1 + 2*k2 + 2*k3 + k4)

    # Stockage des F aux 4 derniers pas
    F_vals = [F(vectT[i], U[i], params) for i in range(4)]

    for i in range(4, n):
        
        U[i] = U[i-1] + (h / 24) * (
              55 * F_vals[-1]
            - 59 * F_vals[-2]
            + 37 * F_vals[-3]
            -  9 * F_vals[-4]
        )
        U[i] = np.maximum(U[i], 1e-10)
        F_vals.append(F(vectT[i], U[i], params))
        F_vals.pop(0)

    return vectT, U


#  CONDITIONS INITIALES ET SIMULATIONS

U0 = np.array([440, 4.5, 2.5, 0.04, 0.25])
T, h = 50, 0.01
n_pas = int(T/h)

# Analyse de la raideur : valeurs propres de la Jacobienne en t=0

J0 = Jacobienne_num(F, 0, U0, params)
valp = np.linalg.eigvals(J0)

print("=" * 55)
print("Valeurs propres de la Jacobienne en t=0, u=U0 :")
for i, v in enumerate(valp):
    print(f"  λ{i+1} = {v.real:.6f}  (partie imag. = {v.imag:.2e})")

lambda_max = np.max(np.abs(valp))
lambda_min = np.min(np.abs(valp[np.abs(valp) > 1e-10]))
rapport    = lambda_max / lambda_min
h_crit     = 2.0 / lambda_max

print(f"\n  |λ_max| = {lambda_max:.4f}")
print(f"  |λ_min| = {lambda_min:.4f}  (hors zéro)")
print(f"  Rapport spectral = {rapport:.1f}")
print(f"  Pas critique Euler explicite h_crit = 2/|λ_max| = {h_crit:.4f}")
print("=" * 55)

# Simulation principale RK4 avec tous les prédateurs
start = time.time()
vectT4, U4 = RK4(T, h, U0, F, params)
t_rk4 = time.time() - start
iter_rk4 = n_pas*4  

# Calcul Euler Implicite avec Newton

start = time.time()
vectT3, U3, iter_euler = Euler_implicite_Newton(T, h, U0, F, params)
t_euler = time.time() - start

# Calcul AB4
start = time.time()
vectTAB, UAB = Adams_Bashforth4(T, h, U0, F, params)
t_ab4 = time.time() - start
iter_ab4 = n_pas +12  

# Référence scipy BDF
sol_ref = solve_ivp(lambda t, u: F(t, u, params),
                    [0, T], U0, method='BDF', max_step=h)


# Figure 1 : RK4 – 3 subplots (Végétation / Proies / Prédateurs)

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
fig.suptitle('Cascade Trophique – RK4 (h = 0.01 an)', fontsize=14, fontweight='bold')

# Subplot 1 : Végétation
ax = axes[0]
ax.plot(vectT4, U4[:, 0], color='#2ecc71', lw=2, label='Végétation V (kg·km⁻²)')
ax.set_ylabel('Végétation\n(kg·km⁻²)')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_title('Niveau 1 – Végétation', fontsize=11)

# Subplot 2 : Ongulés
ax = axes[1]
ax.plot(vectT4, U4[:, 1], color='#3498db', lw=2, label='Ongulés principaux N (wapiti/orignal)')
ax.plot(vectT4, U4[:, 2], color='#9b59b6', lw=2, label='Ongulés secondaires D (cerfs)')
ax.set_ylabel('Densité\n(animaux·km⁻²)')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_title('Niveau 2 – Ongulés', fontsize=11)

# Subplot 3 : Prédateurs
ax = axes[2]
ax.plot(vectT4, U4[:, 3], color='#e74c3c', lw=2, label='Loups W')
ax.plot(vectT4, U4[:, 4], color='#e67e22', lw=2, label='Ours B')
ax.set_ylabel('Densité\n(animaux·km⁻²)')
ax.set_xlabel('Temps (années)')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_title('Niveau 3 – Prédateurs', fontsize=11)

plt.tight_layout(pad=3.0)
plt.subplots_adjust(top=0.9, bottom=0.15, hspace=0.4)
plt.savefig('rk4_3subplots.png', dpi=150, bbox_inches='tight')
plt.show()


# Figure 2: Euler Implicite (Newton) – 3 subplots

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
fig.suptitle('Cascade Trophique – Euler Implicite avec Newton (h = 0.01 an)', fontsize=14, fontweight='bold')

# Subplot 1 : Végétation
ax = axes[0]
ax.plot(vectT3, U3[:, 0], color='#2ecc71', lw=2, label='Végétation V (kg·km⁻²)')
ax.set_ylabel('Végétation\n(kg·km⁻²)')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_title('Niveau 1 – Végétation', fontsize=11)

# Subplot 2 : Ongulés
ax = axes[1]
ax.plot(vectT3, U3[:, 1], color='#3498db', lw=2, label='Ongulés principaux N (wapiti/orignal)')
ax.plot(vectT3, U3[:, 2], color='#9b59b6', lw=2, label='Ongulés secondaires D (cerfs)')
ax.set_ylabel('Densité\n(animaux·km⁻²)')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_title('Niveau 2 – Ongulés', fontsize=11)

# Subplot 3 : Prédateurs
ax = axes[2]
ax.plot(vectT3, U3[:, 3], color='#e74c3c', lw=2, label='Loups W')
ax.plot(vectT3, U3[:, 4], color='#e67e22', lw=2, label='Ours B')
ax.set_ylabel('Densité\n(animaux·km⁻²)')
ax.set_xlabel('Temps (années)')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_title('Niveau 3 – Prédateurs', fontsize=11)

plt.subplots_adjust(top=0.9, bottom=0.15, hspace=0.4)
plt.tight_layout(pad=3.0)
plt.savefig('euler_implicite_3subplots.png', dpi=150, bbox_inches='tight')
plt.show()


# Figure 3 : Étude de la raideur et stabilité d'Euler explicite

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Raideur du système – Euler explicite selon h', fontsize=13, fontweight='bold')

pas_list = [0.01, 0.70, 1.00]
titres_pas = ['h = 0.01 (Stable)', 'h = 0.70 (Limite)', 'h = 1.00 (Divergence)']

for ax, h_test, titre in zip(axes, pas_list, titres_pas):
    
    t_e, U_e = Euler_explicite(T, h_test, U0, F, params)

    ax.plot(t_e, U_e[:, 0], label='V', color='#2ecc71')
    ax.plot(t_e, U_e[:, 1], label='N', color='#3498db')
    ax.plot(t_e, U_e[:, 2], label='D', color='#9b59b6')
    ax.plot(t_e, U_e[:, 3], label='W', color='#e74c3c')
    ax.plot(t_e, U_e[:, 4], label='B', color='#e67e22')

    ax.set_title(titre, fontsize=10)
    ax.set_xlabel('Temps (années)')
    ax.set_yscale('log')

    ax.set_ylim(1e-3, 1e4)

    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

axes[0].set_ylabel('Densité (log)')
plt.tight_layout()
plt.savefig('raideur_euler.png', dpi=150, bbox_inches='tight')
plt.show()


# ─────────────────────────────────────────────
# Figure 4 : Scénario Yellowstone – réintroduction des loups
# Phase 1 (t = 0..20) : loups quasi-absents (W0 = 0.001)
# Phase 2 (t = 20..50) : réintroduction W = 0.04
# On raccorde les deux simulations

T1, T2 = 20, 30
h_ys   = 0.01

# État initial sans loups
U0_ys_phase1 = np.array([440, 4.5, 2.5, 0.001, 0.25])

# Phase 1 : sans loups (prédation nulle)
params_no_wolf        = params.copy()
params_no_wolf['cWN'] = 0.0
params_no_wolf['cWD'] = 0.0
params_no_wolf['mW']  = 1.0  

t1, U_ys1 = RK4(T1, h_ys, U0_ys_phase1, F, params_no_wolf)

# État en fin de phase 1, on réintroduit les loups
U_end_phase1    = U_ys1[-1].copy()
U_start_phase2  = U_end_phase1.copy()
U_start_phase2[3] = 0.04   

t2, U_ys2 = RK4(T2, h_ys, U_start_phase2, F, params)
t2_shifted = t2 + T1 

# Concaténation
t_ys = np.concatenate([t1, t2_shifted[1:]])
U_ys = np.concatenate([U_ys1, U_ys2[1:]], axis=0)

fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
fig.suptitle('Scénario Yellowstone – Réintroduction des loups à t = 20 ans', fontsize=13, fontweight='bold')

labels_col = [
    ('Végétation V', '#2ecc71', 0),
]
ylabels = ['Végétation\n(kg·km⁻²)', 'Ongulés\n(animaux·km⁻²)', 'Prédateurs\n(animaux·km⁻²)']
groups = [
    [(0, 'Végétation V', '#2ecc71')],
    [(1, 'Ongulés principaux N', '#3498db'), (2, 'Ongulés secondaires D', '#9b59b6')],
    [(3, 'Loups W', '#e74c3c'), (4, 'Ours B', '#e67e22')],
]
titles_ys = ['Niveau 1 – Végétation', 'Niveau 2 – Ongulés', 'Niveau 3 – Prédateurs']

for ax, group, ylabel, title in zip(axes, groups, ylabels, titles_ys):
    for idx, label, color in group:
        ax.plot(t_ys, U_ys[:, idx], color=color, lw=2, label=label)
    
    ax.axvline(x=20, color='black', linestyle='--', lw=1.5, alpha=0.7)
    ax.text(20.3, ax.get_ylim()[1] * 0.85 if ax.get_ylim()[1] > 0 else 1,
            'Réintro.\nloups', fontsize=8, color='black', alpha=0.8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

axes[2].set_xlabel('Temps (années)')

for ax in axes:
    ymin, ymax = ax.get_ylim()
    ax.axvline(x=20, color='black', linestyle='--', lw=1.5, alpha=0.7)
    ax.text(20.4, ymin + (ymax - ymin) * 0.7, 'Réintro.\nloups', fontsize=8, color='black', alpha=0.8)

plt.tight_layout()
plt.savefig('yellowstone.png', dpi=150, bbox_inches='tight')
plt.show()

# Figure 5 : Scénarios climatiques / chasse


scenarios = {
    'Référence'         : params.copy(),
    'Sécheresse (rV×0.5)': {**params, 'rV': params['rV'] * 0.5},
    'Chasse intense (µW×3)': {**params, 'muW': params['muW'] * 3},
    'Extinction ours'   : {**params, 'mB': 1.0},   # ours disparaissent rapidement
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
fig.suptitle('Sensibilité aux paramètres – RK4 (h = 0.01)', fontsize=13, fontweight='bold')

for ax, (titre, p) in zip(axes, scenarios.items()):
    t_s, U_s = RK4(T, h, U0, F, p)
    ax.plot(t_s, U_s[:, 0], color='#2ecc71', lw=1.8, label='V')
    ax.plot(t_s, U_s[:, 1], color='#3498db', lw=1.8, label='N')
    ax.plot(t_s, U_s[:, 2], color='#9b59b6', lw=1.8, label='D')
    ax.plot(t_s, U_s[:, 3], color='#e74c3c', lw=1.8, label='W')
    ax.plot(t_s, U_s[:, 4], color='#e67e22', lw=1.8, label='B')
    ax.set_title(titre, fontsize=11)
    ax.set_xlabel('Temps (années)')
    ax.set_ylabel('Densité (log)')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.subplots_adjust(top=0.9, bottom=0.15, hspace=0.4)
plt.savefig('scenarios.png', dpi=150, bbox_inches='tight')
plt.show()


# Figure 6 : Comparaison RK4 vs scipy BDF (validation)

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
fig.suptitle('Validation RK4 vs référence scipy BDF', fontsize=13, fontweight='bold')

groups_val = [
    [(0, 'V RK4', '#2ecc71', '-', None, None), (0, 'V BDF', "#1b663a", '--', '*', 0.2)],
    [(1, 'N RK4', '#3498db', '-', None, None), (1, 'N BDF', "#275472",'--','*',0.2),
     (2, 'D RK4', '#9b59b6', '-', None, None), (2, 'D BDF', "#410856", '--','*',0.2)],
    [(3, 'W RK4', '#e74c3c', '-', None, None), (3, 'W BDF', "#601414", '--', '*',0.2),
     (4, 'B RK4', '#e67e22', '-', None, None), (4, 'B BDF', "#7A4406",'--', '*',0.2)],
]
ylabels_val = ['Végétation', 'Ongulés', 'Prédateurs']

for ax, group, ylabel in zip(axes, groups_val, ylabels_val):
    for idx, label, color, ls, mark, sz in group:
        if 'RK4' in label:
            ax.plot(vectT4, U4[:, idx], color=color, lw=2, ls=ls, label=label, marker = mark, markevery = sz)
        else:
            ax.plot(sol_ref.t, sol_ref.y[idx], color=color, lw=1.5, ls=ls, alpha=0.7, label=label, marker=mark, markevery=sz)
    ax.set_ylabel(ylabel)
    ax.set_yscale('log')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

axes[2].set_xlabel('Temps (années)')
plt.subplots_adjust(top=0.9, bottom=0.15, hspace=0.4)
plt.tight_layout(pad=3.0)
plt.savefig('validation_rk4_bdf.png', dpi=150, bbox_inches='tight')
plt.show()


# Figure 7 : Lotka-Volterra classique vs modèle complet (N et W uniquement)

def F_LV(t, u, r=0.30, a=7.5, b=0.5, m=0.30):
    """Lotka-Volterra canonique à 2 espèces."""
    N, W = u
    dN =  r * N - a * N * W
    dW =  b * N * W - m * W
    return np.array([dN, dW])

t_lv = np.linspace(0, T, int(T/h))
U_lv = np.zeros((len(t_lv), 2))
U_lv[0] = [4.5, 0.04]
for i in range(1, len(t_lv)):
    k1 = F_LV(t_lv[i-1], U_lv[i-1])
    k2 = F_LV(t_lv[i-1] + h/2, U_lv[i-1] + h*k1/2)
    k3 = F_LV(t_lv[i-1] + h/2, U_lv[i-1] + h*k2/2)
    k4 = F_LV(t_lv[i-1] + h, U_lv[i-1] + h*k3)
    U_lv[i] = U_lv[i-1] + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Lotka-Volterra canonique vs Modèle complet (N et W)', fontsize=13, fontweight='bold')

# LV
ax = axes[0]
ax.plot(t_lv, U_lv[:, 0], color='#3498db', lw=2, label='Ongulés N (LV)')
ax.plot(t_lv, U_lv[:, 1] * 50, color='#e74c3c', lw=2, label='Loups W × 50 (LV)')  # ×50 pour visibilité
ax.set_title('Lotka-Volterra classique\n(oscillations non amorties, écologiquement irréaliste)', fontsize=10)
ax.set_xlabel('Temps (années)'); ax.set_ylabel('Densité')
ax.legend(); ax.grid(True, alpha=0.3)


# Modèle complet
ax = axes[1]
ax.plot(vectT4, U4[:, 1], color='#3498db', lw=2, label='Ongulés N')
ax.plot(vectT4, U4[:, 3] * 50, color='#e74c3c', lw=2, label='Loups W × 50')
ax.set_title('Modèle complet à 5 espèces\n(équilibre stable biologiquement réaliste)', fontsize=10)
ax.set_xlabel('Temps (années)'); ax.set_ylabel('Densité')
ax.legend(); ax.grid(True, alpha=0.3)


plt.tight_layout()
plt.savefig('lv_vs_complet.png', dpi=150, bbox_inches='tight')
plt.show()


# Figure 8 erreur de RK4 et Euler-Newton vs référence BDF

from scipy.interpolate import interp1d

labels_err = ['V', 'N', 'D', 'W', 'B']
colors_err = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#e67e22']

# Interpolation de BDF sur la grille commune de RK4
interp_bdf = [interp1d(sol_ref.t, sol_ref.y[i], kind='linear') for i in range(5)]
BDF_sur_RK4 = np.array([interp_bdf[i](vectT4) for i in range(5)]).T  # shape (n, 5)

# Interpolation d'Euler Newton sur la grille de RK4
interp_EN = [interp1d(vectT3, U3[:, i], kind='linear') for i in range(5)]
EN_sur_RK4 = np.array([interp_EN[i](vectT4) for i in range(5)]).T

fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)
fig.suptitle('Erreur de chaque méthode par rapport à la référence BDF scipy\n'
             'RK4 (ordre 4) vs Euler Implicite Newton (ordre 1) — h = 0.01 an',
             fontsize=12, fontweight='bold')

for idx, (label, color) in enumerate(zip(labels_err, colors_err)):
    err_rk4 = np.abs(U4[:, idx]       - BDF_sur_RK4[:, idx])
    err_en  = np.abs(EN_sur_RK4[:, idx] - BDF_sur_RK4[:, idx])

    axes[idx].semilogy(vectT4, err_en,  color='black', lw=1.5, ls='--',
                       label=f'Euler Newton  (moy={np.mean(err_en):.1e})')
    axes[idx].semilogy(vectT4, err_rk4, color=color,   lw=1.5, ls='-',
                       label=f'RK4           (moy={np.mean(err_rk4):.1e})')

    axes[idx].set_ylabel(f'|err| {label}')
    axes[idx].legend(fontsize=8, loc='upper left')
    axes[idx].grid(True, alpha=0.3)

axes[-1].set_xlabel('Temps (années)')
plt.subplots_adjust(top=0.9, bottom=0.15, hspace=0.4)
plt.tight_layout(pad=3.0)
plt.savefig('erreur_vs_BDF.png', dpi=150, bbox_inches='tight')
plt.show()


# Figure 9 : Tableau des performances


# Préparation des données pour le tableau
donnees_tableau = [
    ["Euler Implicite (N)", f"{t_euler:.5f}", f"{iter_euler}", "O(k*N)"],
    ["RK4", f"{t_rk4:.5f}", f"{iter_rk4}", "O(4N)"],
    ["Adams-Bashforth 4", f"{t_ab4:.5f}", f"{iter_ab4}", "O(N)"]
]
colonnes = ["Méthode", "Temps (sec)", "Itérations", "Complexité"]

# Création de la figure
fig_tab, ax_tab = plt.subplots(figsize=(10, 4))
ax_tab.axis('off')  # Masque X et Y

# Création du tableau 
le_tableau = ax_tab.table(cellText=donnees_tableau, 
                          colLabels=colonnes, 
                          loc='center', 
                          cellLoc='center')


le_tableau.auto_set_font_size(False)
le_tableau.set_fontsize(11)
le_tableau.scale(1.2, 2.5) 


for (row, col), cell in le_tableau.get_celld().items():
    if row == 0:
        cell.set_text_props(weight='bold')

plt.title("Comparaison des performances des méthodes numériques", fontweight='bold', pad=20)

plt.savefig('tableau_performances.png', dpi=200, bbox_inches='tight')
plt.show()


print("\n=== Toutes les figures sauvegardées ===")
