"""
Module for the Lenard-Jones potential.
"""
import numpy as np
from scipy.optimize import fsolve, root, minimize
from scipy.integrate import quad, romberg
import jax.numpy as jnp
from jax import grad, jit, vmap
from timeit import default_timer as timer
from tqdm import tqdm
import scipy 
from utils import patching
from constants import hbar, pi, epsilon_0, E, mu, rmax
import matplotlib.pyplot as plt
import warnings


# Parameters for the Lennard-Jones potential
r_e = 2.556e-10  # Equilibrium distance
epslj =  10.22*1.3805e-23    # Potential intensity parameter
lj_m = 8.006204e-3/(6.02214076e23*2) # Reduced mass of the system

# Numerical parameters for each potential's wavefunction
# First classical region
lenard_jones_ct = 1 # Constant used in the Lenard-Jones wavefunction in the classical region

# First quantum region
lenard_jones_ct_qt_1 = 1 # Constant used in the Lenard-Jones wavefunction in the quantum region

# Second quantum region
lenard_jones_ct_qt_2 = 1 # Constant used in the Lenard-Jones wavefunction in the quantum region

# First patching region
A_lenard_jones_1 = 1 # Constant used in the Lenard-Jones wavefunction in the patching region for the A Airy function
B_lenard_jones_1 = 1 # Constant used in the Lenard-Jones wavefunction in the patching region for the B Airy function

# Second patching region
A_lenard_jones_2 = 1 # Constant used in the Lenard-Jones wavefunction in the patching region for the A Airy function
B_lenard_jones_2 = 1 # Constant used in the Lenard-Jones wavefunction in the patching region for the B Airy function


# Lenard-Jones Potential Function
def lenard_jones(x):
    return 4 * ((1/x)**12-(1/x)**6)

# Derivative of the Lenard-Jones potential
def lenard_jones_derivative(x):
    return grad(lenard_jones)(float(x))
    return 4 * (-12*(r_e**12)*(1/x)**13+6*(r_e**6)*(1/x)**7)


# Compute the energy levels
def energy_levels(n: int) -> float:
    E = np.random.uniform(-1, 0)
    # Deactivate Warning
    warnings.filterwarnings('ignore')
    def function_to_minimize(energy):
        beta = lj_m * r_e**2 * epslj / (hbar **2)
        x_space = np.linspace(-rmax, rmax, 100000)
        Vn = min(lenard_jones(x_space))
        tp1 = np.power(2*((1-np.sqrt(1+energy)) / -energy), 1/6)
        tp2 = np.power(2*((1+np.sqrt(1+energy))) / -energy, 1/6)
        integral, error = quad(lambda energy: np.sqrt(2 * abs(energy - Vn)), tp1, tp2)
        return abs(integral - (n+1/2)*pi*hbar/(r_e*np.sqrt(lj_m*epslj)))

    solution = minimize(function_to_minimize, E, method='Nelder-Mead')
    return solution.x



    


# Computation of the speed of oscillation of the wave function

def speed_of_oscillation_lenard_jones(x, energy):
    return lj_m * hbar * abs(lenard_jones_derivative(x)) /\
          (2*lj_m*((abs(energy-lenard_jones(x)))))**(3/2)

# Compute the classical turning points
# All potentials studied have 2 strictly positive turning points

initial_guesses = [1e-10, 1e-9, 2e-9, 3e-9] # Initial guesses for the turning points' x values


def turning_points_lenard(x):
    return lenard_jones(x)-E


# lenard_jones_tp = fsolve(turning_points_lenard, initial_guesses)

# x_1_left_lenard_jones = patching(lenard_jones, lenard_jones_derivative, lenard_jones_tp, speed_of_oscillation_lenard_jones, Elj, epsilon_lenard_jones)[0][0]

# x_2_left_lenard_jones = patching(lenard_jones, lenard_jones_derivative, lenard_jones_tp, speed_of_oscillation_lenard_jones, Elj, epsilon_lenard_jones)[0][1]
# x_1_right_lenard_jones = patching(lenard_jones, lenard_jones_derivative, lenard_jones_tp, speed_of_oscillation_lenard_jones, Elj, epsilon_lenard_jones)[1][0]
# x_2_right_lenard_jones = patching(lenard_jones, lenard_jones_derivative, lenard_jones_tp, speed_of_oscillation_lenard_jones, Elj, epsilon_lenard_jones)[1][1]


def wavefunction_classical_region_lenard_jones(x,n):
    def lenard_jones_classical_momentum(x):
        return np.sqrt(2*lj_m*(Elj-lenard_jones(x)))
    def lenard_jones_classical_1(x,n):
        integral, error = quad(lambda y: lenard_jones_classical_momentum(y), lenard_jones_tp[0] , x, epsabs=1e-3, epsrel=1e-3)
        return 2 * (-1)**n * lenard_jones_ct * (np.sin(integral/ hbar + (pi/4))) / (np.sqrt((lenard_jones_classical_momentum(x))))
    def lenard_jones_classical_2(x):
        integral, error = quad(lambda y: lenard_jones_classical_momentum(y), x, lenard_jones_tp[1] , epsabs=1e-3, epsrel=1e-3)
        return 2 *  lenard_jones_ct * (np.sin(integral / hbar + (pi/4))) / (np.sqrt((lenard_jones_classical_momentum(x))))
    
    return lenard_jones_classical_1(x,n) + lenard_jones_classical_2(x)


def wavefunction_quantum_region_1_lenard_jones(x,n):
    def lenard_jones_classical_momentum(x):
        return np.sqrt(2*lj_m*(Elj-lenard_jones(x)))
    integral, error = quad(lambda y: abs(lenard_jones_classical_momentum(y)), x, lenard_jones_tp[0], epsabs=1e-3, epsrel=1e-3)
    
    return (-1)**n * lenard_jones_ct_qt_1 * (np.exp(- integral/ hbar)) / (np.sqrt(abs(lenard_jones_classical_momentum(x))))

def wavefunction_quantum_region_2_lenard_jones(x):
    def lenard_jones_classical_momentum(x):
        return np.sqrt(2*lj_m*(Elj-lenard_jones(x)))
    integral, error = quad(lambda y: lenard_jones_classical_momentum(y), lenard_jones_tp[1] , x, epsabs=1e-3, epsrel=1e-3)

    return lenard_jones_ct_qt_2 * (np.exp(-1*integral / hbar)) / (np.sqrt(abs(lenard_jones_classical_momentum(x))))

def wavefunction_patching_region_1_lenard_jones(x):
    F0 = lenard_jones_derivative(lenard_jones_tp[0])
    alpha = np.cbrt(2*mu*F0/(hbar**2))
    z = (x - lenard_jones_tp[0]) * alpha
    ai, aip, bi, bip = scipy.special.airy(z)
    globals()["A_lenard_jones_1"] = (np.sqrt(pi) / np.power(2 * mu * hbar * F0, 1/6)) * lenard_jones_ct
    globals()["lenard_jones_ct_qt_1"] = A_lenard_jones_1 * ( np.power(2 * mu * hbar * F0 , 1/6) / (2 * np.sqrt(pi)) )
    return A_lenard_jones_1 * ai 

def wavefunction_patching_region_2_lenard_jones(x):
    F0 = lenard_jones_derivative(lenard_jones_tp[1])
    alpha = np.cbrt(2*mu*F0/(hbar**2))
    z = (x - lenard_jones_tp[1]) * alpha
    ai, aip, bi, bip = scipy.special.airy(z)
    globals()["A_lenard_jones_2"] = (np.sqrt(pi) / np.power(2 * mu * hbar * F0, 1/6)) * lenard_jones_ct
    globals()["lenard_jones_ct_qt_2"] = A_lenard_jones_2 * ( np.power(2 * mu * hbar * F0 , 1/6) / np.sqrt(pi) )
    return A_lenard_jones_2 * ai 


def lenard_jones_wavefunction(x,n):
    if 0<x and x <= x_1_left_lenard_jones:
        return wavefunction_quantum_region_1_lenard_jones(x,n)
    if x_1_left_lenard_jones < x and x <= x_2_left_lenard_jones:
        return wavefunction_patching_region_1_lenard_jones(x,n)
    if x_2_left_lenard_jones < x <= x_1_right_lenard_jones:
        return wavefunction_classical_region_lenard_jones(x,n)
    if x_1_right_lenard_jones < x <= x_2_right_lenard_jones:
        return wavefunction_patching_region_2_lenard_jones(x,n)
    if x_2_right_lenard_jones < x:
        return wavefunction_quantum_region_2_lenard_jones(x)
    return 0


def lenard_jones_normalisation(n):
    initial_guesses = [0, 1, 2, 3] # Initial guesses for the energy value
    def normaliser(ct):
        globals()["lenard_jones_ct"] = ct
        return quad(lambda x: np.abs(lenard_jones_wavefunction(x,n))**2, 0, np.inf, epsabs=1e-12, epsrel=1e-12) - 1
    globals()["lenard_jones_ct"] = fsolve(normaliser, initial_guesses) 
    return None

def final_wavefunction_lenard_jones(x,n):
    lenard_jones_normalisation(n)
    return lenard_jones_wavefunction(x,n)

def wavefunction_exact_lenard_jones(x,n):
    return None

def energy_exact_lenard_jones(n):
    return None

def lenard_jones_wavefunction_error(x,n):
    return abs(final_wavefunction_lenard_jones(x,n) - wavefunction_exact_lenard_jones(x,n))

def energy_error_lenard_jones(n):
    return abs(energy_levels_lenard_jones(n).x[0] - energy_exact_lenard_jones(n))