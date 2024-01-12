"""
Module for the Buckingham Coulomb potential.
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

# -----------------------------------------------------------
# Parameters for numerical solution
PT = 10000    # Number of grid points
x_buckingham_coulomb = np.linspace(0.05, 5, PT)  # Spatial grid
epsilon_buckingham_coulomb = 10e-3 # Bound for the size of the patching region

# -----------------------------------------------------------
# Parameters for the Buckingham Coulomb Potential
buckcoul_exp_amp = 1388.7730 # Amplitude of the exponential term
buckcoul_exp_decay = 0.1 # Decay constant of the exponential term
buckcoul_sixth_amp = 30 # Amplitude of the inverse sextic power term
charge_bc_1 = 1.0 # Charge of the first particle
charge_bc_2 = 1.0 # Charge of the second particle

# -----------------------------------------------------------
# Numerical parameters for each potential's wavefunction
# First classical region
buckingham_coulomb_ct = 1 # Constant used in the Buckingham-Coulomb wavefunction in the classical region
# First quantum region
buckingham_coulomb_ct_qt_1 = 1 # Constant used in the Buckingham-Coulomb wavefunction in the quantum region
# Second quantum region
buckingham_coulomb_ct_qt_2 = 1 # Constant used in the Buckingham-Coulomb wavefunction in the quantum region
# First patching region
A_buckingham_coulomb_1 = 1 # Constant used in the Buckingham-Coulomb wavefunction in the patching region for the A Airy function
B_buckingham_coulomb_1 = 1 # Constant used in the Buckingham-Coulomb wavefunction in the patching region for the B Airy function
# Second patching region
A_buckingham_coulomb_2 = 1 # Constant used in the Buckingham-Coulomb wavefunction in the patching region for the A Airy function
B_buckingham_coulomb_2 = 1 # Constant used in the Buckingham-Coulomb wavefunction in the patching region for the B Airy function

# -----------------------------------------------------------
# Begin Code
# -----------------------------------------------------------
# Buckingham-Coulomb Potential Function
def buckingham_coulomb(x):
    return buckcoul_exp_amp * np.exp(-buckcoul_exp_decay * x) - (buckcoul_sixth_amp)/(x**6) + (charge_bc_1)*(charge_bc_2)/(4*pi*epsilon_0*x)

# -----------------------------------------------------------
# Derivative of the Buckingham-Coulomb potential 
def derivative(x):
    return grad(buckingham_coulomb)(float(x))
    # return -1*buckcoul_exp_amp * buckcoul_exp_decay * np.exp(-buckcoul_exp_decay * x) + (6 * buckcoul_sixth_amp)/ x**7 - (charge_bc_1 * charge_bc_2) / (4 * pi * epsilon_0 * x**2)

# -----------------------------------------------------------
# Define buckingham_coulomb classical momentum
def buckingham_coulomb_classical_momentum(x):
       return np.sqrt(2*mu*(E-buckingham_coulomb(x)))

# -----------------------------------------------------------
# Buckingham-Coulomb Energy Levels
def energy_levels(n):
    pass

# -----------------------------------------------------------
# Speed of oscillation of the wave function
def speed_of_oscillation(x, energy):
    return mu * hbar * abs(derivative(x)) /\
          (2*mu*((abs(energy - buckingham_coulomb(x)))))**(3/2)

# -----------------------------------------------------------
# Classical turning points
initial_guesses = [1e-10, 1e-9, 2e-9, 3e-9] # Initial guesses for the turning points' x values

def turning_points(x):
    return buckingham_coulomb(x)-E

# Lists containing the turning points' x values
# buckingham_coulomb_tp = fsolve(turning_points_buckingham_coulomb, initial_guesses)

# -----------------------------------------------------------
# Wavefunction in the Classical Region
def wavefunction_classical_region(x,n):
    def buckingham_coulomb_classical_1(x,n):
       return 2 * (-1)**n * buckingham_coulomb_ct * (np.sin(np.integrate(buckingham_coulomb_tp[0],x,buckingham_coulomb_classical_momentum(x)) / hbar + (pi/4))) / (np.sqrt((buckingham_coulomb_classical_momentum(x))))
    def buckingham_coulomb_classical_2(x):
       return 2 *  buckingham_coulomb_ct * (np.sin(np.integrate(x, buckingham_coulomb_tp[1], buckingham_coulomb_classical_momentum(x)) / hbar + (pi/4))) / (np.sqrt((buckingham_coulomb_classical_momentum(x))))
    
    return buckingham_coulomb_classical_1(x,n) + buckingham_coulomb_classical_2(x)

# -----------------------------------------------------------
# Wavefunction in the first Quantum Region
def wavefunction_quantum_region_1(x,n):
    def buckingham_coulomb_classical_momentum(x):
        return np.sqrt(2*lj_m*(E-buckingham_coulomb(x)))
    
    return (-1)**n * buckingham_coulomb_ct_qt_1 * (np.exp(-np.integrate(x,buckingham_coulomb_tp[0],abs(buckingham_coulomb_classical_momentum(x))) / hbar)) / (np.sqrt(abs(buckingham_coulomb_classical_momentum(x))))

# -----------------------------------------------------------
# Wavefunction in the second Quantum Region
def wavefunction_quantum_region_2(x):
    def buckingham_coulomb_classical_momentum(x):
        return np.sqrt(2*mu*(E-buckingham_coulomb(x)))
    
    return buckingham_coulomb_ct_qt_2 * (np.exp(-np.integrate(buckingham_coulomb_tp[1],x ,abs(buckingham_coulomb_classical_momentum(x))) / hbar)) / (np.sqrt(abs(buckingham_coulomb_classical_momentum(x))))

# -----------------------------------------------------------
# Patching region wavefunction and match constants
def wavefunction_patching_region_1(x):
    F0 = derivative(buckingham_coulomb_tp[0])
    alpha = np.cbrt(2*mu*F0/(hbar**2))
    z = (x - buckingham_coulomb_tp[0]) * alpha
    ai, aip, bi, bip = scipy.special.airy(z)
    globals()["A_buckingham_coulomb_1"] = (np.sqrt(pi) / np.power(2 * mu * hbar * F0, 1/6)) * buckingham_coulomb_ct
    globals()["buckingham_coulomb_ct_qt_1"] = A_buckingham_coulomb_1 * ( np.power(2 * mu * hbar * F0 , 1/6) / (2 * np.sqrt(pi)) )
    return A_buckingham_coulomb_1 * ai 

def wavefunction_patching_region_2_buckingham_coulomb(x):
    F0 = derivative(buckingham_coulomb_tp[1])
    alpha = np.cbrt(2*mu*F0/(hbar**2))
    z = (x - buckingham_coulomb_tp[1]) * alpha
    ai, aip, bi, bip = scipy.special.airy(z)
    globals()["A_buckingham_coulomb_2"] = (np.sqrt(pi) / np.power(2 * mu * hbar * F0, 1/6)) * buckingham_coulomb_ct
    globals()["buckingham_coulomb_ct_qt_2"] = A_buckingham_coulomb_2 * ( np.power(2 * mu * hbar * F0 , 1/6) / np.sqrt(pi) )
    return A_buckingham_coulomb_2 * ai 

# -----------------------------------------------------------
# Define patching region limits 

#x_1_left_buckingham_coulomb = patching(buckingham_coulomb, buckingham_coulomb_derivative, buckingham_coulomb_tp, speed_of_oscillation_buckingham_coulomb, E, epsilon_buckingham_coulomb)[0][0]
#x_2_left_buckingham_coulomb = patching(buckingham_coulomb, buckingham_coulomb_derivative, buckingham_coulomb_tp, speed_of_oscillation_buckingham_coulomb, E, epsilon_buckingham_coulomb)[0][1]
#x_1_right_buckingham_coulomb = patching(buckingham_coulomb, buckingham_coulomb_derivative, buckingham_coulomb_tp, speed_of_oscillation_buckingham_coulomb, E, epsilon_buckingham_coulomb)[1][0]
#x_2_right_buckingham_coulomb = patching(buckingham_coulomb, buckingham_coulomb_derivative, buckingham_coulomb_tp, speed_of_oscillation_buckingham_coulomb, E, epsilon_buckingham_coulomb)[1][1]

# -----------------------------------------------------------
# Glue the wavefunctions for each potential
def wavefunction(x,n):
    if 0<x <= x_1_left_buckingham_coulomb:
        return wavefunction_quantum_region_1(x,n)
    if x_1_left_buckingham_coulomb < x <= x_2_left_buckingham_coulomb:
        return wavefunction_patching_region_1(x,n)
    if x_2_left_buckingham_coulomb < x <= x_1_right_buckingham_coulomb:
        return wavefunction_classical_region(x,n)
    if x_1_right_buckingham_coulomb < x <= x_2_right_buckingham_coulomb:
        return wavefunction_patching_region_2_buckingham_coulomb(x,n)
    if x_2_right_buckingham_coulomb < x:
        return wavefunction_quantum_region_2(x)
    return 0

# -----------------------------------------------------------
# Fix the constants for the wavefunction through the normalisation condition applied one the 0
def normalisation(n):
    initial_guesses = [0, 1, 2, 3] # Initial guesses for the energy value
    def normaliser(ct):
        globals()["buckingham_coulomb_ct"] = ct
        return np.sqrt(np.integrate(0, np.inf, np.abs(wavefunction(x_buckingham_coulomb,n))**2, x_buckingham_coulomb)) - 1
    globals()["buckingham_coulomb_ct"] = fsolve(normaliser, initial_guesses) 
    return None
    
# -----------------------------------------------------------
# Define the final wavefunction
def final_wavefunction(x,n):
    normalisation(n)
    return wavefunction(x,n)

# -----------------------------------------------------------
# Compute the exact solution for the wavefunction and energies
def wavefunction_exact(x,n):
    return None

def energy_exact(n):
    return None

# -----------------------------------------------------------
# Error in the energies and wavefunction
def wavefunction_error(x,n):
    return abs(final_wavefunction(x,n) - wavefunction_exact(x,n))


def energy_error(n):
    return abs(energy_levels_buckingham_coulomb(n) - energy_exact(n))
