"""
Module for the Buckingham potential.
"""
import numpy as np
from scipy.optimize import fsolve
import jax.numpy as jnp
from jax import grad, jit, vmap
from timeit import default_timer as timer
import scipy 
from utils import patching
from constants import hbar, pi, epsilon_0, E, mu, rmax


# Parameters for numerical solution
L = 10.0   # Spatial domain length
PT = 10000    # Number of grid points
x_morse = np.linspace(0, 3, PT)  # Spatial grid
x_lj = np.linspace(1e-2, 10, PT)  # Spatial grid
x_buckingham = np.linspace(0.05, 5, PT)  # Spatial grid
x_buckingham_coulomb = np.linspace(0.05, 5, PT)  # Spatial grid
epsilon_morse = 10e-3 # Bound for the size of the patching region
epsilon_lenard_jones = 1e-10 # Bound for the size of the patching region
epsilon_buckingham = 10e-3 # Bound for the size of the patching region
epsilon_buckingham_coulomb = 10e-3 # Bound for the size of the patching region
N = 10 # Number of levels to be computed

# Parameters for the Buckingham Potential
buck_exp_amp = 1388.7730 # Amplitude of the exponential term
buck_exp_decay = 10 # Decay constant of the exponential term
buck_sixth_amp = 0.0001 # Amplitude of the inverse sextic power term

# Numerical parameters for each potential's wavefunction
# First classical region
buckingham_ct = 1 # Constant used in the Buckingham wavefunction in the classical region

# First quantum region
buckingham_ct_qt_1 = 1 # Constant used in the Buckingham wavefunction in the quantum region

# Second quantum region
buckingham_ct_qt_2 = 1 # Constant used in the Buckingham wavefunction in the quantum region

# First patching region
A_buckingham_1 = 1 # Constant used in the Buckingham wavefunction in the patching region for the A Airy function
B_buckingham_1 = 1 # Constant used in the Buckingham wavefunction in the patching region for the B Airy function

# Second patching region
A_buckingham_2 = 1 # Constant used in the Buckingham wavefunction in the patching region for the A Airy function
B_buckingham_2 = 1 # Constant used in the Buckingham wavefunction in the patching region for the B Airy function



# Buckingham Potential Function
def buckingham(x):
    return buck_exp_amp * np.exp(-buck_exp_decay * x) - (buck_sixth_amp)/(x**6)

# Derivative of the Buckingham potential 
def buckingham_derivative(x):
    return -buck_exp_amp * buck_exp_decay * np.exp(-buck_exp_decay * x) + \
        (6 * buck_sixth_amp) / x**7

##############################################################################

# Computation of the speed of oscillation of the wave function

def speed_of_oscillation_buckingham(x, energy):
    return mu * hbar * abs(buckingham_derivative(x)) /\
          (2*mu*((abs(energy-buckingham(x)))))**(3/2)


##############################################################################


# Compute the classical turning points
# All potentials studied have 2 strictly positive turning points

initial_guesses = [1e-10, 1e-9, 2e-9, 3e-9] # Initial guesses for the turning points' x values

def turning_points_buckingham(x):
    return buckingham(x)-E

# Lists containing the turning points' x values
buckingham_tp = fsolve(turning_points_buckingham, initial_guesses)

##############################################################################

# Define patching region limits 

x_1_left_buckingham = patching(buckingham, buckingham_derivative, buckingham_tp, speed_of_oscillation_buckingham, E, epsilon_buckingham)[0][0]
x_2_left_buckingham =  patching(buckingham, buckingham_derivative, buckingham_tp, speed_of_oscillation_buckingham, E, epsilon_buckingham)[0][1]
x_1_right_buckingham = patching(buckingham, buckingham_derivative, buckingham_tp, speed_of_oscillation_buckingham, E, epsilon_buckingham)[1][0]
x_2_right_buckingham = patching(buckingham, buckingham_derivative, buckingham_tp, speed_of_oscillation_buckingham, E, epsilon_buckingham)[1][1]


def wavefunction_classical_region_buckingham(x,n):
    def buckingham_classical_momentum(x):
       return np.sqrt(2*mu*(E-buckingham(x)))
    def buckingham_classical_1(x,n):
       return 2 * (-1)**n * buckingham_ct * (np.sin(np.integrate(buckingham_tp[0],x,buckingham_classical_momentum(x)) / hbar + (pi/4))) / (np.sqrt((buckingham_classical_momentum(x))))
    def buckingham_classical_2(x):
       return 2 *  buckingham_ct * (np.sin(np.integrate(x, buckingham_tp[1], buckingham_classical_momentum(x)) / hbar + (pi/4))) / (np.sqrt((buckingham_classical_momentum(x))))
    
    return buckingham_classical_1(x,n) + buckingham_classical_2(x)


def wavefunction_quantum_region_1_buckingham(x,n):
    def buckingham_classical_momentum(x):
        return np.sqrt(2*lj_m*(E-buckingham(x)))
    
    return (-1)**n * buckingham_ct_qt_1 * (np.exp(-np.integrate(x,buckingham_tp[0],abs(buckingham_classical_momentum(x))) / hbar)) / (np.sqrt(abs(buckingham_classical_momentum(x))))


def wavefunction_quantum_region_2_buckingham(x):
    def buckingham_classical_momentum(x):
        return np.sqrt(2*mu*(E-buckingham(x)))
    
    return buckingham_ct_qt_2 * (np.exp(-np.integrate(buckingham_tp[1],x ,abs(buckingham_classical_momentum(x))) / hbar)) / (np.sqrt(abs(buckingham_classical_momentum(x))))


def wavefunction_patching_region_1_buckingham(x):
    F0 = buckingham_derivative(buckingham_tp[0])
    alpha = np.cbrt(2*mu*F0/(hbar**2))
    z = (x - buckingham_tp[0]) * alpha
    ai, aip, bi, bip = scipy.special.airy(z)
    globals()["A_buckingham_1"] = (np.sqrt(pi) / np.power(2 * mu * hbar * F0, 1/6)) * buckingham_ct
    globals()["buckingham_ct_qt_1"] = A_buckingham_1 * ( np.power(2 * mu * hbar * F0 , 1/6) / (2 * np.sqrt(pi)) )
    return A_buckingham_1 * ai 


def wavefunction_patching_region_2_buckingham(x):
    F0 = buckingham_derivative(buckingham_tp[1])
    alpha = np.cbrt(2*mu*F0/(hbar**2))
    z = (x - buckingham_tp[1]) * alpha
    ai, aip, bi, bip = scipy.special.airy(z)
    globals()["A_buckingham_2"] = (np.sqrt(pi) / np.power(2 * mu * hbar * F0, 1/6)) * buckingham_ct
    globals()["buckingham_ct_qt_2"] = A_buckingham_2 * ( np.power(2 * mu * hbar * F0 , 1/6) / np.sqrt(pi) )
    return A_buckingham_2 * ai 


def buckingham_wavefunction(x,n):
    if 0<x <= x_1_left_buckingham:
        return wavefunction_quantum_region_1_buckingham(x,n)
    if x_1_left_buckingham < x <= x_2_left_buckingham:
        return wavefunction_patching_region_1_buckingham(x,n)
    if x_2_left_buckingham < x <= x_1_right_buckingham:
        return wavefunction_classical_region_buckingham(x,n)
    if x_1_right_buckingham < x <= x_2_right_buckingham:
        return wavefunction_patching_region_2_buckingham(x,n)
    if x_2_right_buckingham < x:
        return wavefunction_quantum_region_2_buckingham(x)
    return 0


def buckingham_normalisation(n):
    initial_guesses = [0, 1, 2, 3] # Initial guesses for the energy value
    def normaliser(ct):
        globals()["buckingham_ct"] = ct
        return np.sqrt(np.integrate(0, np.inf, np.abs(buckingham_wavefunction(x_buckingham,n))**2, x_buckingham)) - 1
    globals()["buckingham_ct"] = fsolve(normaliser, initial_guesses) 
    return None
    

def final_wavefunction_buckingham(x,n):
    buckingham_normalisation(n)
    return buckingham_wavefunction(x,n)


def wavefunction_exact_buckingham(x,n):
    return None


def energy_exact_buckingham(n):
    return None


def buckingham_wavefunction_error(x,n):
    return abs(final_wavefunction_buckingham(x,n) - wavefunction_exact_buckingham(x,n))


def energy_error_buckingham(n):
    return abs(energy_levels_buckingham(n) - energy_exact_buckingham(n))


