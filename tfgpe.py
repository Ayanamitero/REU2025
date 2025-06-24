import numpy as np
from scipy.fft import fft, ifft, fftfreq
import matplotlib.pyplot as plt

#Parameters
g = 500
m = 1
omega = 1
R = ((9 * g**2) / (4 * m**2 * omega**4))**0.20
L = 1.2 * R
N = 4096
x = np.linspace(-L, L, N)
dx = x[1] - x[0]

dt = 1e-5
N_t = 20000
record = 10

k = 2 * np.pi * fftfreq(N, d=dx)

T = 0.5 * k**2
V = 0.5 * m * omega**2 * x**2

def psi_i(x, omega, g, m, V):
    mu = ((9 * g**2 * m * omega**2) / 32)**(1/3)
    modulus = np.maximum(0, (mu - V) / g)
    return np.sqrt(modulus)

def split_step(psi, T, V, dt):
    expT = np.exp(-T * dt)
    expV = np.exp(-(V + g * np.abs(psi)**2) * dt/2)
    
    evol = []

    for t in range(N_t):
        psi *= expV
        psi_k = fft(psi)
        psi_k *= expT
        psi = ifft(psi_k) 
        psi *= expV
        
        psi /= np.sqrt(np.trapz(np.abs(psi)**2, x))
        
        if t % record == 0:
            evol.append(psi.copy())
    
    return psi, evol

psi0 = psi_i(x, omega, g, m, V).astype(complex)

psif, evo = split_step(psi0, T, V, dt)

mu = ((9 * g**2 * m * omega**2) / 32)**(1/3)
psi_tf_sq = np.maximum(0, (mu - V) / g)

def energy(psi):
    psi_k = fft(psi)
    normk = np.trapz(np.abs(psi_k)**2, k)
    E_T = np.trapz(np.abs(psi_k)**2 * T, k) / normk
    E_V = np.trapz(np.abs(psi)**2 * (V + g * np.abs(psi)**2), x)

    return E_T + E_V

E = energy(psif) 
print(f'The total energy is {E:.2f}.')

plt.plot(x, np.abs(psif)**2, 'm', label='Numerical $|\psi(x)|^2$')
plt.plot(x, psi_tf_sq, 'k--', label='TF Profile $|\psi_{TF}(x)|^2$')
plt.xlabel('x')
plt.ylabel(r'$|\psi(x)|^2$')
plt.legend()
plt.grid(True)
plt.title('TF Initial Guess vs. Final Numerical Ground State')
plt.show()

times = np.arange(0, N_t, record) * dt
density_evo = np.abs(np.array(evo))**2

plt.figure(figsize=(8, 4))
plt.imshow(density_evo, extent=[x[0], x[-1], times[-1], times[0]], 
           aspect='auto', cmap='viridis', origin='upper')
plt.colorbar(label=r'$|\psi(x,t)|^2$')
plt.xlabel('x')
plt.ylabel('Time')
plt.title('Density Evolution $|\psi(x,t)|^2$ during Imaginary Time Evolution')
plt.tight_layout()
plt.show()

