"""
Module for the Morse potential.
"""
# -----------------------------------------------------------
# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, root
from scipy.integrate import odeint
from scipy import integrate
import scipy.special
from scipy.integrate import quad, romberg
from tqdm import tqdm
from constants import hbar, pi, epsilon_0, E, mu, rmax
from jax import grad, jit, vmap
import jax.numpy as jnp
from utils import patching
import sys
import os

# -----------------------------------------------------------
# Add the directory containing 'constants' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# -----------------------------------------------------------
# Parameters for numerical solution
PT = 10000    # Number of grid points
x_morse = np.linspace(0, 3, PT)  # Spatial grid
epsilon_morse = 10e-3 # Bound for the size of the patching region

# -----------------------------------------------------------
# Parameters for the Morse potential
D_e = 5.211   # Dissociation energy in eV
a = 2.78   # Width parameter in A
x_e = 1.207  # Equilibrium position in A^-1
morse_m = mu # Reduced mass of the system

# -----------------------------------------------------------
# Numerical parameters for each potential's wavefunction
# First classical region
morse_ct = 1 # Constant used in the morse wavefunction in the classical region
# First quantum region
morse_ct_qt_1 = 1 # Constant used in the morse wavefunction in the quantum region
# Second quantum region
morse_ct_qt_2 = 1 # Constant used in the morse wavefunction in the quantum region
# First patching region
A_morse_1 = 1 # Constant used in the Morse wavefunction in the patching region for the A Airy function
B_morse_1 = 1 # Constant used in the Morse wavefunction in the patching region for the B Airy function
# Second patching region
A_morse_2 = 1 # Constant used in the Morse wavefunction in the patching region for the A Airy function
B_morse_2 = 1 # Constant used in the Morse wavefunction in the patching region for the B Airy function

# -----------------------------------------------------------
# Begin Code
# -----------------------------------------------------------
# Morse Potential Function
def morse(x):
    return D_e * (1 - jnp.exp(-a * (x - x_e)))**2

# -----------------------------------------------------------
# Derivative of the Morse potential
def derivative(x: float) -> float:  # sourcery skip: remove-unnecessary-cast
    return grad(morse)(float(x))
    # return -2*D_e*a* (1 - np.exp(-a * (x - x_e))) * np.exp(-a * (x - x_e))

# -----------------------------------------------------------
# Define morse classical momentum
def classical_momentum(x):
    return np.sqrt(2*morse_m*(E-morse(x)))

# -----------------------------------------------------------
# Morse Energy Levels
def energy_levels(n):
    initial_guesses_energy_morse = [1, 1, 2] # Initial guesses for the energy value
    def energy_optimiser_morse(energy, tp1, tp2):
        eq1 = (np.integrate(tp1, tp2 , classical_momentum(x_morse,energy),x_morse) - (n+1/2)*pi*hbar)
        eq2 = morse(tp1) - energy
        eq3 = morse(tp2) - energy
        return [eq1, eq2, eq3]
    return fsolve(energy_optimiser_morse, initial_guesses_energy_morse)

# -----------------------------------------------------------
# Speed of oscillation of the wave function
def speed_of_oscillation(x: float, energy: float) -> float:
    return morse_m * hbar * abs(derivative(x)) /\
          (2*morse_m*((abs(energy-morse(x)))))**(3/2)

# -----------------------------------------------------------
# Classical turning points
initial_guesses = [1e-10, 1e-9, 2e-9, 3e-9] # Initial guesses for the turning points' x values
def turning_points(x):
    return morse(x)-E

# # Lists containing the turning points' x values
# morse_tp = fsolve(turning_points, initial_guesses)

# -----------------------------------------------------------
# Wavefunction in the Classical Region
def wavefunction_classical_region(x,n):
    def morse_classical_1(x,n):
       return 2 * (-1)**n * morse_ct * (np.sin(np.integrate(morse_tp[0],x,classical_momentum(x)) / hbar + (pi/4))) / (np.sqrt((classical_momentum(x))))
    def morse_classical_2(x):
       return 2 *  morse_ct * (np.sin(np.integrate(x, morse_tp[1], classical_momentum(x)) / hbar + (pi/4))) / (np.sqrt((classical_momentum(x))))
    
    return morse_classical_1(x,n) + morse_classical_2(x)

# -----------------------------------------------------------
# Wavefunction in the first Quantum Region
def wavefunction_quantum_region_1(x,n):
    return (-1)**n * morse_ct_qt_1 * (np.exp(-np.integrate(x,morse_tp[0],abs(classical_momentum(x))) / hbar)) / (np.sqrt(abs(classical_momentum(x))))

# -----------------------------------------------------------
# Wavefunction in the second Quantum Region
def wavefunction_quantum_region_2(x):
    return morse_ct_qt_2 * (np.exp(-np.integrate(morse_tp[1],x ,abs(classical_momentum(x))) / hbar)) / (np.sqrt(abs(classical_momentum(x))))

# -----------------------------------------------------------
# Patching region wavefunction and match constants
def wavefunction_patching_region_1(x):
    F0 = derivative(morse_tp[0])
    alpha = np.cbrt(2*mu*F0/(hbar**2))
    z = (x - morse_tp[0]) * alpha
    ai, aip, bi, bip = scipy.special.airy(z)
    globals()["A_morse_1"] = (np.sqrt(pi) / np.power(2 * mu * hbar * F0, 1/6)) * morse_ct
    globals()["morse_ct_qt_1"] = A_morse_1 * ( np.power(2 * mu * hbar * F0 , 1/6) / (2 * np.sqrt(pi)) )
    return A_morse_1 * ai 


def wavefunction_patching_region_2(x):
    F0 = derivative(morse_tp[1])
    alpha = np.cbrt(2*mu*F0/(hbar**2))
    z = (x - morse_tp[1]) * alpha
    ai, aip, bi, bip = scipy.special.airy(z)
    globals()["A_morse_2"] = (np.sqrt(pi) / np.power(2 * mu * hbar * F0, 1/6)) * morse_ct
    globals()["morse_ct_qt_2"] = A_morse_2 * ( np.power(2 * mu * hbar * F0 , 1/6) / np.sqrt(pi) )
    return A_morse_2 * ai 

# -----------------------------------------------------------
# Define patching region limits 
# x_1_left_morse = patching(morse, derivative, morse_tp, speed_of_oscillation, E, epsilon_morse)[0][0]
# x_2_left_morse = patching(morse, derivative, morse_tp, speed_of_oscillation, E, epsilon_morse)[0][1]
# x_1_right_morse = patching(morse, derivative, morse_tp, speed_of_oscillation, E, epsilon_morse)[1][0]
# x_2_right_morse = patching(morse, derivative, morse_tp, speed_of_oscillation, E, epsilon_morse)[1][1]

# -----------------------------------------------------------
# Glue the wavefunctions for each potential
def wavefunction(x,n):
    if 0<x <= x_1_left_morse:
        return wavefunction_quantum_region_1(x,n)
    if x_1_left_morse < x <= x_2_left_morse:
        return wavefunction_patching_region_1(x,n)
    if x_2_left_morse < x <= x_1_right_morse:
        return wavefunction_classical_region(x,n)
    if x_1_right_morse < x <= x_2_right_morse:
        return wavefunction_patching_region_2(x,n)
    if x_2_right_morse < x:
        return wavefunction_quantum_region_2(x)
    return 0

# -----------------------------------------------------------
# Fix the constants for the wavefunction through the normalisation condition applied one the 0
def normalisation(n):
    initial_guesses = [0, 1, 2, 3] # Initial guesses for the energy value
    def normaliser(ct):
        globals()["morse_ct"] = ct
        return np.sqrt(np.integrate(0, np.inf, np.abs(wavefunction(x_morse,n))**2, x_morse)) - 1
    globals()["morse_ct"] = fsolve(normaliser, initial_guesses) 
    return None

# -----------------------------------------------------------
# Define the final wavefunction
def final_wavefunction(x,n):
    normalisation(n)
    return wavefunction(x,n)

# -----------------------------------------------------------
# Compute the exact solution for the wavefunction and energies
def wavefunction_exact(x,n):
    K = np.sqrt(2*morse_m*D_e)/(hbar*a)

    def whittaker_M(n,k,x):
        return np.exp(-x/2) * x**(k+1/2) * scipy.special.hyp1f1(k-n+0.5,1+2*k,x)
    
    def N(n):
        return (scipy.special.gamma(n+1) * scipy.special.gamma(2*K-2*n-1)) / (a * scipy.special.gamma(2*K-n))
    
    return whittaker_M(K,K-n-0.5,2*K*np.exp(-a*(x-x_e))) / np.sqrt(2*K*N(n)*np.exp(-a*(x-x_e)))


def energy_exact(n):
    a_0 = a*hbar/np.sqrt(mu)
    return D_e + 0.5*(np.sqrt(2*D_e)-(n+0.5)*a_0)**2

# -----------------------------------------------------------
# Error in the energies and wavefunction

def wavefunction_error(x,n):
    return abs(final_wavefunction(x,n) - wavefunction_exact(x,n))


def energy_error(n):
    return abs(energy_levels(n) - energy_exact(n))

# -----------------------------------------------------------
# Plot Functions 

def plot_potencial(x: range = x_morse, log: bool = False, save_as: str = None):
    plt.figure(figsize=(10,6))
    plt.plot(x, morse(x), label="Morse Potential")
    plt.xlabel('Position - r')
    plt.ylabel('Potential - V(r)')
    plt.title('Morse Potential')
    plt.legend()
    plt.ylim(-1, 20)
    if log:
        plt.yscale('log')
    if save_as:
        plt.savefig(save_as)
    plt.show()

def plot_speed_of_osciallation(x: range = x_morse, Energy: float = E, save_as: str = None):
    plt.figure(figsize=(10,6))
    plt.plot(x, speed_of_oscillation(x, Energy), label="Speed of Oscillation")
    plt.xlabel('Position - r')
    plt.ylabel('Speed of Oscillation - V(r)')
    plt.title('Speed of oscillation of the Morse Potential')
    plt.legend()
    if save_as:
        plt.savefig(save_as)
    plt.show()

def plot_wavefunction(x: range = x_morse, limit: int = 0, save_as: str = None):
    plt.figure(figsize=(10,6))
    for n in range(limit):
        plt.plot(x, final_wavefunction(x,n), label="Numerical Solution")
        plt.plot(x, wavefunction_exact(x,n), label="Exact Solution")
    plt.xlabel('Position - r')
    plt.ylabel(r'Wavefunction - $\psi$(r)')
    plt.title('Wavefunction for the Morse Potential')
    plt.legend()
    if save_as:
        plt.savefig(save_as)
    plt.show()


def plot_wavefunction_error(x: range = x_morse, limit: int = 0, save_as: str = None):
    plt.figure(figsize=(10,6))
    for n in range(limit):
        plt.plot(x, wavefunction_error(x,n), label="Error")
    plt.xlabel('Position - r')
    plt.ylabel(r'Wavefunction Error - $\psi$(r)')
    plt.title('Wavefunction Error for the Morse Potential')
    plt.legend()
    if save_as:
        plt.savefig(save_as)
    plt.show()

def plot_energy(limit: int = 0, save_as: str = None):
    plt.figure(figsize=(10,6))
    for n in range(limit):
        plt.plot(n, energy_levels(n), label=f"Energy {n}")
    plt.xlabel('Energy Level - n')
    plt.ylabel('Energy Value [J]')
    plt.title('Energy Levels for the Morse Potential')
    plt.legend()
    if save_as:
        plt.savefig(save_as)
    plt.show()


def plot_energy_error(limit: int = 0, save_as: str = None):
    plt.figure(figsize=(10,6))
    for n in range(limit):
        plt.plot(n, energy_levels(n), label=f"Energy {n}")
    plt.xlabel('Energy Level - n')
    plt.ylabel('Energy Error Value [J]')
    plt.title('Energy Levels Error for the Morse Potential')
    plt.legend()
    if save_as:
        plt.savefig(save_as)
    plt.show()

