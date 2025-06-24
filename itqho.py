import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from scipy.special import hermite, factorial

#Parameters
L = 10
N = 4096
x = np.linspace(-L, L, N)
dx = x[1] - x[0]

dt = 1e-5
N_t = 25000      #Number of time steps taken
record = 10

k = 2 * np.pi * fftfreq(N, d=dx)

T = 0.5 * k**2
V = 0.5 * x **2

#---Computation---
psi_0i = np.exp(-x**2 / 2).astype(complex)

psi_1i = x * np.exp(-x**2 / 2).astype(complex)

psi_2i = (x**2 - 1) * np.exp(-x**2 / 2).astype(complex)

psi_3i = (2 * x**3 - x) * np.exp(-x**2 / 2).astype(complex)

    #Orthoganlization 
def project_out(psi, *states):
    for phi in states:
        overlap = np.trapz(np.conj(phi) * psi, x)
        psi -= overlap * phi
    return psi

    #Split-Step Fourier
def split_step(psi, T, V, dt, ortho=[]):
    expT = np.exp(-T * dt)
    expV = np.exp(-V * dt/2)

    evol = []

    for t in range(N_t):
        psi *= expV
        psi_k = fft(psi)
        psi_k *= expT
        psi = ifft(psi_k) 
        psi *= expV
        
        if ortho:
            psi = project_out(psi, *ortho)

        psi /= np.sqrt(np.trapz(np.abs(psi)**2, x))
        
        if t % record == 0:
            evol.append(psi.copy())

    return psi, evol

    #Calculation of energy using E = <T> /||ψ(k, t)^2|| + <V> 
def energy(psi):
    psi_k = fft(psi)
    normk = np.trapz(np.abs(psi_k)**2, k)
    E_T = np.trapz(np.abs(psi_k)**2 * T, k) / normk
    E_V = np.trapz(np.abs(psi)**2 * V, x)
    
    return E_T + E_V

    #Analytical
def psi_analytic(x, n):
    Hn = hermite(n)
    norm = np.pi**(-0.25)/ np.sqrt(2**n * factorial (n))
    return norm * Hn(x) * np.exp(-x**2 / 2)

psi_0f, evo0 = split_step(psi_0i, T, V, dt)
psi_1f, evo1 = split_step(psi_1i, T, V, dt, ortho=[psi_0f])
psi_2f, evo2 = split_step(psi_2i, T, V, dt, ortho=[psi_0f, psi_1f])
psi_3f, evo3 = split_step(psi_3i, T, V, dt, ortho=[psi_0f, psi_1f, psi_2f])

psi_0 = psi_analytic(x, 0).astype(complex)
psi_1 = psi_analytic(x, 1).astype(complex)
psi_2 = psi_analytic(x, 2).astype(complex)
psi_3 = psi_analytic(x, 3).astype(complex)

E0 = energy(psi_0f)
E1 = energy(psi_1f)
E2 = energy(psi_2f)
E3 = energy(psi_3f)

print(f'The energy for n = 0: {E0:.5f}')
print(f'The energy for n = 1: {E1:.5f}')
print(f'The energy for n = 2: {E2: .5f}')
print(f'The energy for n = 3: {E3: .5f}')

numerical = [psi_0f, psi_1f, psi_2f, psi_3f]
analytic = [psi_0, psi_1, psi_2, psi_3]
E = [E0, E1, E2, E3]
times = np.arange(0, N_t, record)

s = 0
for t in range(len(times)):
    for j in range(len(E)):
        s += np.exp(-E[j] * t) * numerical[j]
    
    s /= np.sqrt(np.trapz(np.abs(s)**2, x))

#Plot of Wavefunction densities
plt.plot(x, s)
plt.grid(True)
plt.show()

fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

colors = ['blue', 'green', 'red', 'orange']
labels = ['Ground State', '1st Excited', '2nd Excited', '3rd Excited']

for i, ax in enumerate(axs):
    # Plot wavefunctions
    ax.plot(x, np.abs(numerical[i])**2, color=colors[i], label=f"{labels[i]} (Numerical)")
    ax.plot(x, np.abs(analytic[i])**2, color=colors[i], linestyle='--', label=f"{labels[i]} (Analytic)")

    # Plot potential
    V_scaled = V / V.max() * np.max(np.abs(numerical[i])**2)
    ax.plot(x, V_scaled, color='magenta', label='Potential $V(x)$')

    ax.set_ylabel(r"$|\psi(x)|^2$")
    ax.legend(loc='upper right')
    ax.grid(True)

axs[-1].set_xlabel("x")
fig.suptitle("Numerical vs Analytical Harmonic Oscillator Eigenstates", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

#Time Evolution of Wavefunctions(?)
evolution_stack = [evo0, evo1, evo2, evo3]
titles = [
    r"Ground State $\Re(\psi_0(x,t))$",
    r"1st Excited $\Re(\psi_1(x,t))$", 
    r"2nd Excited $\Re(\psi(x, t))$",
    r"3rd Excited $\Re(\psi(x, t))$"]

fig, axs = plt.subplots(4, 1, figsize=(12, 8), sharex=True)

for i, ax in enumerate(axs):
    im = ax.imshow(
        np.real(evolution_stack[i]),
        extent=[x[0], x[-1], times[-1], times[0]],
        aspect='auto', cmap='plasma'
    )
    ax.set_ylabel("Time")
    ax.set_title(titles[i])
    fig.colorbar(im, ax=ax, label=r"$\Re(\psi(x,t))$")

axs[-1].set_xlabel("x")
plt.suptitle("Time Evolution of Harmonic Oscillator States", fontsize=14)
plt.tight_layout()
plt.show()

#Time Evolution of Wavefunctions(?)
times = np.arange(0, N_t, record) * dt
evolution_stack = [evo0, evo1, evo2, evo3]
titles = [
    r"Ground State $\Im(\psi_0(x,t))$",
    r"1st Excited $\Im(\psi_1(x,t))$", 
    r"2nd Excited $\Im(\psi(x, t))$",
    r"3rd Excited $\Im(\psi(x, t))$"]

fig, axs = plt.subplots(4, 1, figsize=(12, 8), sharex=True)

for i, ax in enumerate(axs):
    im = ax.imshow(
        np.imag(evolution_stack[i]),
        extent=[x[0], x[-1], times[-1], times[0]],
        aspect='auto', cmap='plasma'
    )
    ax.set_ylabel("Time")
    ax.set_title(titles[i])
    fig.colorbar(im, ax=ax, label=r"$\Im(\psi(x,t))$")

axs[-1].set_xlabel("x")
plt.suptitle("Time Evolution of Harmonic Oscillator States", fontsize=14)
plt.tight_layout()
plt.show()
import matplotlib.animation as animation

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(10, 6))
lines = []
colors = ['blue', 'green', 'red', 'orange']
labels = ['Ground State', '1st Excited', '2nd Excited', '3rd Excited']

for i in range(4):
    (line,) = ax.plot([], [], label=labels[i], color=colors[i])
    lines.append(line)

ax.set_xlim(x[0], x[-1])
ax.set_ylim(-1, 1)
ax.set_xlabel("x")
ax.set_ylabel(r"$\Re(\psi(x,t))$")
ax.set_title("Time Evolution of Real Part of Harmonic Oscillator States")
ax.legend(loc='upper right')
ax.grid(True)

# Initialization function
def init():
    for line in lines:
        line.set_data([], [])
    return lines

# Animation function
def animate(i):
    for j, line in enumerate(lines):
        line.set_data(x, np.real(evolution_stack[j][i]))
    return lines

ani = animation.FuncAnimation(
    fig,
    animate,
    frames=len(evolution_stack[0]),
    init_func=init,
    blit=True,
    interval=30  # Adjust speed here (ms between frames)
)

plt.tight_layout()
plt.show()

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(10, 6))
lines = []
colors = ['blue', 'green', 'red', 'orange']
labels = ['Ground State', '1st Excited', '2nd Excited', '3rd Excited']

for i in range(4):
    (line,) = ax.plot([], [], label=labels[i], color=colors[i])
    lines.append(line)

ax.set_xlim(x[0], x[-1])
ax.set_ylim(-1, 1)
ax.set_xlabel("x")
ax.set_ylabel(r"$\Re(\psi(x,t))$")
ax.set_title("Time Evolution of Imaginary Part of Harmonic Oscillator States")
ax.legend(loc='upper right')
ax.grid(True)

# Initialization function
def init():
    for line in lines:
        line.set_data([], [])
    return lines

# Animation function
def animate(i):
    for j, line in enumerate(lines):
        line.set_data(x, np.imag(evolution_stack[j][i]))
    return lines

ani = animation.FuncAnimation(
    fig,
    animate,
    frames=len(evolution_stack[0]),
    init_func=init,
    blit=True,
    interval=10  # Adjust speed here (ms between frames)
)

plt.tight_layout()
plt.show()

