import numpy as np
from scipy.fft import fft, ifft, fftfreq
import matplotlib.pyplot as plt

# Dimensions
L = 20 
N = 4096
x = np.linspace(-L, L, N)
dx = x[1] - x[0]

dt = 4e-3
steps = 200000
record = 100

k = 2 * np.pi * fftfreq(N, d=dx)

# Constants
U = 3.0
Omega = 0.4
g11 = 1.0
g22 = g11
g12 = 0.88
g = g12 / np.sqrt(g11 * g22)

# Initial Wavefunctions
psi0 = np.zeros((2, N), dtype=complex)
psi0[0] = np.exp(-(0.5 * (x - L / 4)**2) + 1j) * np.tanh(x / np.sqrt(2))
psi0[1] = np.exp(-(0.5 * (x + L / 4)**2) - 1j) * np.tanh(x / np.sqrt(2))

for i in range(len(psi0)):
    psi0[i] /= np.trapz(np.abs(psi0[i])**2, x)

# Split-Step and the Operators
def split_step(psi, U, Omega, g, dt, steps, x):
    evo  = []

    for step in range(steps):
        n1 = np.abs(psi[0])**2
        n2 = np.abs(psi[1])**2

        V1 = (n1 + g * n2)
        V2 = (g * n1 + n2)

        expV1 = np.exp(-V1 * dt / 2)
        expV2 = np.exp(-V2 * dt / 2)

        t = step * dt
        t_mid = t + 0.5 * dt

        u = U * np.cos(Omega * t_mid)
        T1 = 0.5 * k**2 - u * k
        T2 = 0.5 * k**2 + u * k

        expT1 = np.exp(-T1 * dt)
        expT2 = np.exp(-T2 * dt)

        psi[0] *= expV1
        psi[1] *= expV2

        psi[0] = ifft(fft(psi[0]) * expT1)
        psi[1] = ifft(fft(psi[1]) * expT2)

        n1 = np.abs(psi[0])**2
        n2 = np.abs(psi[1])**2

        V1 = (n1 + g * n2)
        V2 = (g * n1 + n2)

        expV1 = np.exp(-V1 * dt / 2)
        expV2 = np.exp(-V2 * dt / 2)

        psi[0] *= expV1
        psi[1] *= expV2
       
        if step % 100 == 0:
            psi[0] /= np.sqrt(np.trapz(np.abs(psi[0])**2, x))
            psi[1] /= np.sqrt(np.trapz(np.abs(psi[1])**2, x))

        if step % record == 0:
            evo.append(psi.copy())
                
    return psi[0], psi[1], [e[0] for e in evo], [e[1] for e in evo]
# Run simulation
psi1, psi2, evo1, ev02 = split_step(psi0, U, Omega, g, dt, steps, x)

# Final state plot
plt.plot(x, np.real(np.conj(psi1) * psi1), 'g', label=r"$|\psi_1(x, t)|^2 / n_0$")
plt.plot(x, np.real(np.conj(psi2) * psi2), 'r', label=r"$|\psi_2(x, t)|^2 / n_0$")
plt.xlabel(r"$x / \xi_j$")
plt.ylabel(r'$|\psi(x)|^2 / n_0$')
plt.grid(True)
plt.legend()
plt.title("Final State Densities (Imaginary Time Evolution)")
plt.show()


times = np.arange(0, steps, record) * dt
evo1_array = np.array(evo1)
evo2_array = np.array(ev02)

X, T = np.meshgrid(x, times)

fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# First component
pcm0 = axs[0].pcolormesh(
    T, X,
        np.real(np.conj(evo1_array) * evo1_array),
    shading='auto',
    cmap='gnuplot'
)
axs[0].invert_yaxis()
axs[0].set_ylabel(r"$x / \xi_j$")
axs[0].set_title(r"$|\psi_1(x,t)|^2 / n_0$")
fig.colorbar(pcm0, ax=axs[0], label=r"$|\psi_1|^2 / n_0$")

# Second component
pcm1 = axs[1].pcolormesh(
    T, X,
    np.real(np.conj(evo2_array) * evo2_array),
    shading='auto',
    cmap='gnuplot'
)
axs[1].invert_yaxis()
axs[1].set_xlabel(r"$t / \tau$")
axs[1].set_title(r"$\psi_2(x,t)|^2 / n_0$")
axs[1].set_ylabel(r"$x / \xi_j$")
fig.colorbar(pcm1, ax=axs[1], label=r"$|\psi_2|^2 / n_0$")

plt.suptitle("Probability Density of the Wavefunctions Over Time (Imaginary Time Evolution)")
plt.tight_layout()
plt.show()
