# https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html

#VER OS INITIAL GUESSES TODOS
#VER A INTEGRACAO
#VER INFINITAS CONSTANTES TODAS
#Já so falta as coisas exatas

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, root
from scipy.integrate import odeint
from scipy import integrate
import scipy.special
from scipy.integrate import quad, romberg
from tqdm import tqdm
##############################################################################

print('Potenciais disponíveis (código): \n Lennard-Jones (lj) \n Morse (m) \n Buckingham (b) \n Buckingham-Coulomb (bc)')
potencial = input('Potencial pretendido (inserir apenas código): ')
En = int(input('Nível de energia pretendido: '))


# Global Constants
hbar = 1.055e-34  # Reduced Planck's constant in J*s
pi = np.pi  # Pi
epsilon_0 = 8.8541878128e-12  # Vacuum eletric permittivity in C^2/(N*m^2)
E = -498000/6.02214076e23 # Energy of the diatomic oxygen molecule in Joule
mu = 5.314e-26 # Reduced mass of the diatomic oxygen molecule in kg
rmax = 10.0 # Maximum value of the spatial domain considered in meter
#En = 0 #Energy level

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

# Parameters for the Morse potential
D_e = 5.211   # Dissociation energy in eV
a = 2.78   # Width parameter in A
x_e = 1.207  # Equilibrium position in A^-1
morse_m = mu # Reduced mass of the system

# Parameters for the Lennard-Jones potential
r_e = 2.556e-10  # Equilibrium distance
epslj =  10.22*1.3805e-23    # Potential intensity parameter
lj_m = 8.006204e-3/(6.02214076e23*2) # Reduced mass of the system

# Parameters for the Buckingham Potential
buck_exp_amp = 1388.7730 # Amplitude of the exponential term
buck_exp_decay = 10 # Decay constant of the exponential term
buck_sixth_amp = 0.0001 # Amplitude of the inverse sextic power term

# Parameters for the Buckingham Coulomb Potential
buckcoul_exp_amp = 1388.7730 # Amplitude of the exponential term
buckcoul_exp_decay = 0.1 # Decay constant of the exponential term
buckcoul_sixth_amp = 30 # Amplitude of the inverse sextic power term
charge_bc_1 = 1.0 # Charge of the first particle
charge_bc_2 = 1.0 # Charge of the second particle

# Numerical parameters for each potential's wavefunction
# First classical region
morse_ct = 1 # Constant used in the morse wavefunction in the classical region
lenard_jones_ct = 1 # Constant used in the Lenard-Jones wavefunction in the classical region
buckingham_ct = 1 # Constant used in the Buckingham wavefunction in the classical region
buckingham_coulomb_ct = 1 # Constant used in the Buckingham-Coulomb wavefunction in the classical region

# First quantum region
morse_ct_qt_1 = 1 # Constant used in the morse wavefunction in the quantum region
lenard_jones_ct_qt_1 = 1 # Constant used in the Lenard-Jones wavefunction in the quantum region
buckingham_ct_qt_1 = 1 # Constant used in the Buckingham wavefunction in the quantum region
buckingham_coulomb_ct_qt_1 = 1 # Constant used in the Buckingham-Coulomb wavefunction in the quantum region

# Second quantum region
morse_ct_qt_2 = 1 # Constant used in the morse wavefunction in the quantum region
lenard_jones_ct_qt_2 = 1 # Constant used in the Lenard-Jones wavefunction in the quantum region
buckingham_ct_qt_2 = 1 # Constant used in the Buckingham wavefunction in the quantum region
buckingham_coulomb_ct_qt_2 = 1 # Constant used in the Buckingham-Coulomb wavefunction in the quantum region

# First patching region
A_morse_1 = 1 # Constant used in the Morse wavefunction in the patching region for the A Airy function
B_morse_1 = 1 # Constant used in the Morse wavefunction in the patching region for the B Airy function
A_lenard_jones_1 = 1 # Constant used in the Lenard-Jones wavefunction in the patching region for the A Airy function
B_lenard_jones_1 = 1 # Constant used in the Lenard-Jones wavefunction in the patching region for the B Airy function
A_buckingham_1 = 1 # Constant used in the Buckingham wavefunction in the patching region for the A Airy function
B_buckingham_1 = 1 # Constant used in the Buckingham wavefunction in the patching region for the B Airy function
A_buckingham_coulomb_1 = 1 # Constant used in the Buckingham-Coulomb wavefunction in the patching region for the A Airy function
B_buckingham_coulomb_1 = 1 # Constant used in the Buckingham-Coulomb wavefunction in the patching region for the B Airy function
# Second patching region
A_morse_2 = 1 # Constant used in the Morse wavefunction in the patching region for the A Airy function
B_morse_2 = 1 # Constant used in the Morse wavefunction in the patching region for the B Airy function
A_lenard_jones_2 = 1 # Constant used in the Lenard-Jones wavefunction in the patching region for the A Airy function
B_lenard_jones_2 = 1 # Constant used in the Lenard-Jones wavefunction in the patching region for the B Airy function
A_buckingham_2 = 1 # Constant used in the Buckingham wavefunction in the patching region for the A Airy function
B_buckingham_2 = 1 # Constant used in the Buckingham wavefunction in the patching region for the B Airy function
A_buckingham_coulomb_2 = 1 # Constant used in the Buckingham-Coulomb wavefunction in the patching region for the A Airy function
B_buckingham_coulomb_2 = 1 # Constant used in the Buckingham-Coulomb wavefunction in the patching region for the B Airy function

##############################################################################

# Define the potentials used in the project

# Morse Potential Function
def morse(x):
    return D_e * (1 - np.exp(-a * (x - x_e)))**2

# Lenard-Jones Potential Function
def lenard_jones(x):
    return 4 * ((1/x)**12-(1/x)**6)

# Buckingham Potential Function
def buckingham(x):
    return buck_exp_amp * np.exp(-buck_exp_decay * x) - (buck_sixth_amp)/(x**6)

# Buckingham-Coulomb Potential Function
def buckingham_coulomb(x):
    return buckcoul_exp_amp * np.exp(-buckcoul_exp_decay * x) - (buckcoul_sixth_amp)/(x**6) + (charge_bc_1)*(charge_bc_2)/(4*pi*epsilon_0*x)

##############################################################################

# Compute the energy levels

def energy_levels_morse(n):
    initial_guesses_energy_morse = [1, 1, 2] # Initial guesses for the energy value
    def morse_classical_momentum(x,energy):
        return np.sqrt(2*morse_m*(energy-morse(x)))
    def energy_optimiser_morse(energy, tp1, tp2):
        eq1 = (np.integrate(tp1, tp2 , morse_classical_momentum(x_morse,energy),x_morse) - (n+1/2)*pi*hbar)
        eq2 = morse(tp1) - energy
        eq3 = morse(tp2) - energy
        return [eq1, eq2, eq3]
    return fsolve(energy_optimiser_morse, initial_guesses_energy_morse)
    
def energy_levels_lenard_jones(n):
    initial_guesses_energy_lenard_jones = [-0.1] # Initial guesses for the energy value

    def lenard_jones_classical_momentum(x,energy):
        return np.sqrt(2*abs(energy-lenard_jones(x)))
    def energy_optimiser_lenard_jones(energy):
        tp1 = np.power(2*((1-np.sqrt(1+energy)) / -energy), 1/6)
        tp2 = np.power(2*((1+np.sqrt(1+energy))) / -energy, 1/6)
        print(tp1, "|", tp2, "|", energy)
        integral = romberg(lambda x: lenard_jones_classical_momentum(x, energy), tp1, tp2, tol=1e-10, rtol=1e-10, divmax=20)
        print("integral: ",integral)
        eq1 = integral - (n+1/2)*pi*hbar/(r_e*np.sqrt(lj_m*epslj))
        return eq1
    
    solution = root(energy_optimiser_lenard_jones, initial_guesses_energy_lenard_jones, method='lm')
    tp1 = np.power(abs(2*((1-np.sqrt(1+solution.x[0]))) / solution.x[0]), 1/6)
    tp2 = np.power(abs(2*((1+np.sqrt(1+solution.x[0]))) / solution.x[0]), 1/6)
    final = [solution.x[0],tp1 ,tp2]
    return final

def energy_test(n: int) -> float:
    eps = 10e-2
    value = np.inf
    # while abs(value) > abs(eps):
    for i in tqdm(range(10000)):
        try:
            E = np.random.uniform(-1, 0)
            beta = lj_m * r_e**2 * epslj / (hbar **2)
            E_0 = 2 * E / epslj
            E_0 = np.random.uniform(-1, 0)
            x = (1/2) ** 2 / beta
            def lenard_jones_classical_momentum(z):
                beta = lj_m * r_e**2 * epslj / (hbar **2)
                E_0 = 2 * E / epslj
                x = (1/2) ** 2 / beta
                v_grande = lenard_jones(z)
                Vn = 2 / epslj * v_grande + x/(z**2)
                return np.sqrt(abs(E_0 - Vn))
                
            tp1 = np.power(2*((1-np.sqrt(1+E_0)) / -E_0), 1/6)
            tp2 = np.power(2*((1+np.sqrt(1+E_0))) / -E_0, 1/6)
            integral, error = integrate.quad(lenard_jones_classical_momentum, tp1, tp2)
            # integral = romberg(lambda x: lenard_jones_classical_momentum(x, E), tp1, tp2, tol=1e-10, rtol=1e-10, divmax=1000)
            valor = (np.sqrt(beta) / np.pi) * integral - (n+1/2)
            if valor < value:
                value = valor
            # x = np.random.uniform(-1, 0)
            # def func(E):
            #     return integral - (n+1/2)*pi*hbar/(r_e*np.sqrt(lj_m*epslj))
            # sol = scipy.optimize.root_scalar(func, bracket=[-10, 10])
            # print(sol)
        except KeyboardInterrupt:
            print(value)
            break
        except:
            pass
    return E_0

x = []
y = []

for i in range(10):
    valor = energy_test(10)
    print(f'N: {i} | Valor: {valor}\n')
    x.append(i)
    y.append(valor)

plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.show()
print(x, y)
Elj, tp1, tp2 = energy_levels_lenard_jones(En)
lenard_jones_tp = (tp1, tp2)

#print("energy: ", Elj)
#print(lenard_jones_tp)

##############################################################################
# Plot Potential Functions

if potencial == 'm':
    plt.figure(figsize=(10, 6))
    plt.plot(x_morse, morse(x_morse), label='Morse Potential')
    plt.xlabel('Position - r')
    plt.ylabel('Potential - V(r)')
    plt.title('Morse Potential')
    plt.legend()
    plt.ylim(-1, 20)
    plt.show()

if potencial == 'lj':
    plt.plot(x_lj, lenard_jones(x_lj), label='Lenard-Jones Potential')
    plt.plot(x_lj, np.linspace(Elj, Elj, PT), label='Lenard-Jones Potential')
    plt.xlabel('Position - r')
    plt.ylabel('Potential - V(r)')
    plt.title('Lenard-Jones Potential')
    plt.legend()
    plt.ylim(-2.5, 10)
    #plt.yscale('log')
    plt.show()
'''
if potencial == 'b':
    plt.plot(x_buckingham, buckingham(x_buckingham), label='Buckingham Potential')
    plt.xlabel('Position - r')
    plt.ylabel('Potential - V(r)')
    plt.title('Buckingham Potential')
    plt.legend()
    #plt.yscale('log')
    plt.show()

if potencial == 'bc':
    plt.plot(x_buckingham_coulomb, buckingham_coulomb(x_buckingham_coulomb), label='Buckingham-Coulomb Potential')
    plt.xlabel('Position - r')
    plt.ylabel('Potential - V(r)')
    plt.title('Buckingham-Coulomb Potential')
    plt.legend()
    #plt.yscale('log')
    plt.show()

##############################################################################

# Compute the derivatives of the potential functions

# Derivative of the Morse potential
def morse_derivative(x):
    return -2*D_e*a* (1 - np.exp(-a * (x - x_e))) * np.exp(-a * (x - x_e))

# Derivative of the Lenard-Jones potential
def lenard_jones_derivative(x):
    return 4 * (-12*(r_e**12)*(1/x)**13+6*(r_e**6)*(1/x)**7)

# Derivative of the Buckingham potential 
def buckingham_derivative(x):
    return -buck_exp_amp * buck_exp_decay * np.exp(-buck_exp_decay * x) + \
        (6 * buck_sixth_amp) / x**7

# Derivative of the Buckingham-Coulomb potential 
def buckingham_coulomb_derivative(x):
    return -1*buckcoul_exp_amp * buckcoul_exp_decay * \
        np.exp(-buckcoul_exp_decay * x) + (6 * buckcoul_sixth_amp)/ x**7 \
            - (charge_bc_1 * charge_bc_2) / (4 * pi * epsilon_0 * x**2)

##############################################################################

# Computation of the speed of oscillation of the wave function

def speed_of_oscillation_morse(x, energy):
    return morse_m * hbar * abs(morse_derivative(x)) /\
          (2*morse_m*((abs(energy-morse(x)))))**(3/2)

def speed_of_oscillation_lenard_jones(x, energy):
    return lj_m * hbar * abs(lenard_jones_derivative(x)) /\
          (2*lj_m*((abs(energy-lenard_jones(x)))))**(3/2)

def speed_of_oscillation_buckingham(x, energy):
    return mu * hbar * abs(buckingham_derivative(x)) /\
          (2*mu*((abs(energy-buckingham(x)))))**(3/2)

def speed_of_oscillation_buckingham_coulomb(x, energy):
    return mu * hbar * abs(buckingham_coulomb_derivative(x)) /\
          (2*mu*((abs(energy - buckingham_coulomb(x)))))**(3/2)

##############################################################################

# Plot Speed of Oscilation 

if potencial == 'm':
    plt.plot(x_morse, speed_of_oscillation_morse(x_morse,E), label='Morse Potential')
    plt.xlabel('Position - r')
    plt.ylabel('Speed of Oscillation - V(r)')
    plt.title('Speed of oscillation of the Morse Potential')
    plt.legend()
    plt.show()

if potencial == 'lj':
    plt.plot(x_lj, speed_of_oscillation_lenard_jones(x_lj, Elj), label='Lenard-Jones Potential')
    plt.xlabel('Position - r')
    plt.ylabel('Speed of Oscillation - V(r')
    plt.title('Speed of oscillation of the Lennard-jones Potential')
    plt.legend()
    plt.show()

if potencial == 'b':
    plt.plot(x_buckingham, speed_of_oscillation_buckingham(x_buckingham,E), label='Buckingham Potential')
    plt.xlabel('Position - r')
    plt.ylabel('Speed of Oscillation - V(r')
    plt.title('Speed of oscillation of the Buckingham Potential')
    plt.legend()
    plt.show()

if potencial == 'bc':
    plt.plot(x_buckingham_coulomb, speed_of_oscillation_buckingham_coulomb(x_buckingham_coulomb,E), label='Buckingham-Coulomb Potential')
    plt.xlabel('Position - r')
    plt.ylabel('Speed of Oscillation - V(r')
    plt.title('Speed of oscillation of the Buckingham-Coulomb Potential')
    plt.legend()
    plt.show()


##############################################################################


# Compute the classical turning points
# All potentials studied have 2 strictly positive turning points
"""
initial_guesses = [1e-10, 1e-9, 2e-9, 3e-9] # Initial guesses for the turning points' x values

def turning_points_morse(x):
    return morse(x)-E


def turning_points_buckingham(x):
    return buckingham(x)-E

def turning_points_buckingham_coulomb(x):
    return buckingham_coulomb(x)-E

# Lists containing the turning points' x values
morse_tp = fsolve(turning_points_morse, initial_guesses)

buckingham_tp = fsolve(turning_points_buckingham, initial_guesses)
buckingham_coulomb_tp = fsolve(turning_points_buckingham_coulomb, initial_guesses)
"""

#print(lenard_jones_tp)

##############################################################################

# Compute the patching region size

def patching(potential, pot_derivative, tp :list, oscillation_speed, energy : float, epsilon : float):
    def lin_approx(x,tp2):
        return pot_derivative(tp2) * (x-tp2) + potential(tp2)
    final_points = list()
    size_right1 = 1e-17
    size_left1 = 1e-17
    epsilon1 = 1e-11
        
    # Determine the size of the region of validity of the linear approximation
    #print(tp[0], ": ", tp[0]+size_right1, ": ", abs(lin_approx(tp[0]+size_right1,tp[0]) - potential(tp[0]+size_right1)))
    #print(tp[0], ": ", tp[0]-size_left1, ": ", abs(lin_approx(tp[0]-size_left1,tp[0]) - potential(tp[0]-size_left1)))

    while(abs(lin_approx(tp[0]+size_right1,tp[0]) - potential(tp[0]+size_right1)) < epsilon1):
        size_right1 += 1e-17
        #print(tp[0], ": ", tp[0]+size_right1, ": ", abs(lin_approx(tp[0]+size_right1,tp[0]) - potential(tp[0]+size_right1)))
        #break

    while(abs(lin_approx(tp[0]-size_left1,tp[0]) - potential(tp[0]-size_left1)) < epsilon1):
        #print(tp[0], ": ", tp[0]-size_left1, ": ", abs(lin_approx(tp[0]-size_left1,tp[0]) - potential(tp[0]-size_left1)))
        size_left1 -= 1e-17
        
    # Determine the size of the region where wkb is not valid
    region_left = list()
    region_right = list()
    for x in np.linspace(tp[0]-size_left1, tp[0], int(abs(size_left1)/1e-17)):
        if oscillation_speed(x, energy) < epsilon1:
            region_left.append(x)        
    for x in np.linspace(tp[0], tp[0] + size_right1, int(abs(size_right1)/1e-17)):
        if oscillation_speed(x, energy) < epsilon1:
            region_right.append(x)
    final_points.append((max(region_left),min(region_right)))
    
    size_right2 = 1e-7
    size_left2 = 1e-7
    epsilon2 = 1e-26
        
    # Determine the size of the region of validity of the linear approximation
    print(tp[1], ": ", tp[1]+size_right2, ": ", abs(lin_approx(tp[1]+size_right2,tp[1]) - potential(tp[1]+size_right2)))

    while(abs(lin_approx(tp[1]+size_right2,tp[1]) - potential(tp[1]+size_right2)) < epsilon2):
        size_right2 += 1e-7
        #print(tp[1], ": ", tp[1]+size_right2, ": ", abs(lin_approx(tp[1]+size_right2,tp[1]) - potential(tp[1]+size_right2)))
        

    while(abs(lin_approx(tp[1]-size_left2,tp[1]) - potential(tp[1]-size_left2)) < epsilon2):
        size_left2 -= 1e-7
        
    # Determine the size of the region where wkb is not valid
    region_left = list()
    region_right = list()
    for x in np.linspace(tp[1]-size_left2, tp[1], int(abs(size_left2)/1e-7)):
        if oscillation_speed(x, energy) < epsilon:
            region_left.append(x)        
    for x in np.linspace(tp[1], tp[1] + size_right2, int(abs(size_right2)/1e-7)):
        if oscillation_speed(x, energy) < epsilon:
            region_right.append(x)
    final_points.append((max(region_left),min(region_right)))

    return final_points

##############################################################################

# Define patching region limits 
#x_1_left_morse = patching(morse, morse_derivative, morse_tp, speed_of_oscillation_morse, E, epsilon_morse)[0][0]
#x_2_left_morse = patching(morse, morse_derivative, morse_tp, speed_of_oscillation_morse, E, epsilon_morse)[0][1]
#x_1_right_morse = patching(morse, morse_derivative, morse_tp, speed_of_oscillation_morse, E, epsilon_morse)[1][0]
#x_2_right_morse = patching(morse, morse_derivative, morse_tp, speed_of_oscillation_morse, E, epsilon_morse)[1][1]


#print(lenard_jones_derivative(lenard_jones_tp[1]))
x_1_left_lenard_jones = patching(lenard_jones, lenard_jones_derivative, lenard_jones_tp, speed_of_oscillation_lenard_jones, Elj, epsilon_lenard_jones)[0][0]

x_2_left_lenard_jones = patching(lenard_jones, lenard_jones_derivative, lenard_jones_tp, speed_of_oscillation_lenard_jones, Elj, epsilon_lenard_jones)[0][1]
x_1_right_lenard_jones = patching(lenard_jones, lenard_jones_derivative, lenard_jones_tp, speed_of_oscillation_lenard_jones, Elj, epsilon_lenard_jones)[1][0]
x_2_right_lenard_jones = patching(lenard_jones, lenard_jones_derivative, lenard_jones_tp, speed_of_oscillation_lenard_jones, Elj, epsilon_lenard_jones)[1][1]
print("ola2")

#x_1_left_buckingham = patching(buckingham, buckingham_derivative, buckingham_tp, speed_of_oscillation_buckingham, E, epsilon_buckingham)[0][0]
#x_2_left_buckingham =  patching(buckingham, buckingham_derivative, buckingham_tp, speed_of_oscillation_buckingham, E, epsilon_buckingham)[0][1]
#x_1_right_buckingham = patching(buckingham, buckingham_derivative, buckingham_tp, speed_of_oscillation_buckingham, E, epsilon_buckingham)[1][0]
#x_2_right_buckingham = patching(buckingham, buckingham_derivative, buckingham_tp, speed_of_oscillation_buckingham, E, epsilon_buckingham)[1][1]

#_1_left_buckingham_coulomb = patching(buckingham_coulomb, buckingham_coulomb_derivative, buckingham_coulomb_tp, speed_of_oscillation_buckingham_coulomb, E, epsilon_buckingham_coulomb)[0][0]
#x_2_left_buckingham_coulomb = patching(buckingham_coulomb, buckingham_coulomb_derivative, buckingham_coulomb_tp, speed_of_oscillation_buckingham_coulomb, E, epsilon_buckingham_coulomb)[0][1]
#x_1_right_buckingham_coulomb = patching(buckingham_coulomb, buckingham_coulomb_derivative, buckingham_coulomb_tp, speed_of_oscillation_buckingham_coulomb, E, epsilon_buckingham_coulomb)[1][0]
#x_2_right_buckingham_coulomb = patching(buckingham_coulomb, buckingham_coulomb_derivative, buckingham_coulomb_tp, speed_of_oscillation_buckingham_coulomb, E, epsilon_buckingham_coulomb)[1][1]

##############################################################################


# Compute wavefunction in the Classical Region

#def wavefunction_classical_region_morse(x,n):
 #   def morse_classical_momentum(x):
  #      return np.sqrt(2*morse_m*(E-morse(x)))
   # def morse_classical_1(x,n):
    #    return 2 * (-1)**n * morse_ct * (np.sin(np.integrate(morse_tp[0],x,morse_classical_momentum(x)) / hbar + (pi/4))) / (np.sqrt((morse_classical_momentum(x))))
    #def morse_classical_2(x):
     #   return 2 *  morse_ct * (np.sin(np.integrate(x, morse_tp[1], morse_classical_momentum(x)) / hbar + (pi/4))) / (np.sqrt((morse_classical_momentum(x))))
    
    #return morse_classical_1(x,n) + morse_classical_2(x)

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

#def wavefunction_classical_region_buckingham(x,n):
 #   def buckingham_classical_momentum(x):
  #      return np.sqrt(2*mu*(E-buckingham(x)))
   # def buckingham_classical_1(x,n):
    #    return 2 * (-1)**n * buckingham_ct * (np.sin(np.integrate(buckingham_tp[0],x,buckingham_classical_momentum(x)) / hbar + (pi/4))) / (np.sqrt((buckingham_classical_momentum(x))))
    #def buckingham_classical_2(x):
     #   return 2 *  buckingham_ct * (np.sin(np.integrate(x, buckingham_tp[1], buckingham_classical_momentum(x)) / hbar + (pi/4))) / (np.sqrt((buckingham_classical_momentum(x))))
    
    #return buckingham_classical_1(x,n) + buckingham_classical_2(x)

#def wavefunction_classical_region_buckingham_coulomb(x,n):
 #   def buckingham_coulomb_classical_momentum(x):
  #      return np.sqrt(2*mu*(E-buckingham_coulomb(x)))
   # def buckingham_coulomb_classical_1(x,n):
    #    return 2 * (-1)**n * buckingham_coulomb_ct * (np.sin(np.integrate(buckingham_coulomb_tp[0],x,buckingham_coulomb_classical_momentum(x)) / hbar + (pi/4))) / (np.sqrt((buckingham_coulomb_classical_momentum(x))))
    #def buckingham_coulomb_classical_2(x):
     #   return 2 *  buckingham_coulomb_ct * (np.sin(np.integrate(x, buckingham_coulomb_tp[1], buckingham_coulomb_classical_momentum(x)) / hbar + (pi/4))) / (np.sqrt((buckingham_coulomb_classical_momentum(x))))
    
 #   return buckingham_coulomb_classical_1(x,n) + buckingham_coulomb_classical_2(x)

##############################################################################

# Compute wavefunction in the first Quantum Region
"""
def wavefunction_quantum_region_1_morse(x,n):
    def morse_classical_momentum(x):
        return np.sqrt(2*lj_m*(E-morse(x)))
    
    return (-1)**n * morse_ct_qt_1 * (np.exp(-np.integrate(x,morse_tp[0],abs(morse_classical_momentum(x))) / hbar)) / (np.sqrt(abs(morse_classical_momentum(x))))
"""
def wavefunction_quantum_region_1_lenard_jones(x,n):
    def lenard_jones_classical_momentum(x):
        return np.sqrt(2*lj_m*(Elj-lenard_jones(x)))
    integral, error = quad(lambda y: abs(lenard_jones_classical_momentum(y)), x, lenard_jones_tp[0], epsabs=1e-3, epsrel=1e-3)
    
    return (-1)**n * lenard_jones_ct_qt_1 * (np.exp(- integral/ hbar)) / (np.sqrt(abs(lenard_jones_classical_momentum(x))))
"""
def wavefunction_quantum_region_1_buckingham(x,n):
    def buckingham_classical_momentum(x):
        return np.sqrt(2*lj_m*(E-buckingham(x)))
    
    return (-1)**n * buckingham_ct_qt_1 * (np.exp(-np.integrate(x,buckingham_tp[0],abs(buckingham_classical_momentum(x))) / hbar)) / (np.sqrt(abs(buckingham_classical_momentum(x))))

def wavefunction_quantum_region_1_buckingham_coulomb(x,n):
    def buckingham_coulomb_classical_momentum(x):
        return np.sqrt(2*lj_m*(E-buckingham_coulomb(x)))
    
    return (-1)**n * buckingham_coulomb_ct_qt_1 * (np.exp(-np.integrate(x,buckingham_coulomb_tp[0],abs(buckingham_coulomb_classical_momentum(x))) / hbar)) / (np.sqrt(abs(buckingham_coulomb_classical_momentum(x))))

##############################################################################

# Compute wavefunction in the second Quantum Region

def wavefunction_quantum_region_2_morse(x):
    def morse_classical_momentum(x):
        return np.sqrt(2*morse_m*(E-morse(x)))
    
    return morse_ct_qt_2 * (np.exp(-np.integrate(morse_tp[1],x ,abs(morse_classical_momentum(x))) / hbar)) / (np.sqrt(abs(morse_classical_momentum(x))))
"""
def wavefunction_quantum_region_2_lenard_jones(x):
    def lenard_jones_classical_momentum(x):
        return np.sqrt(2*lj_m*(Elj-lenard_jones(x)))
    integral, error = quad(lambda y: lenard_jones_classical_momentum(y), lenard_jones_tp[1] , x, epsabs=1e-3, epsrel=1e-3)

    return lenard_jones_ct_qt_2 * (np.exp(-1*integral / hbar)) / (np.sqrt(abs(lenard_jones_classical_momentum(x))))
"""
def wavefunction_quantum_region_2_buckingham(x):
    def buckingham_classical_momentum(x):
        return np.sqrt(2*mu*(E-buckingham(x)))
    
    return buckingham_ct_qt_2 * (np.exp(-np.integrate(buckingham_tp[1],x ,abs(buckingham_classical_momentum(x))) / hbar)) / (np.sqrt(abs(buckingham_classical_momentum(x))))

def wavefunction_quantum_region_2_buckingham_coulomb(x):
    def buckingham_coulomb_classical_momentum(x):
        return np.sqrt(2*mu*(E-buckingham_coulomb(x)))
    
    return buckingham_coulomb_ct_qt_2 * (np.exp(-np.integrate(buckingham_coulomb_tp[1],x ,abs(buckingham_coulomb_classical_momentum(x))) / hbar)) / (np.sqrt(abs(buckingham_coulomb_classical_momentum(x))))

##############################################################################

# Compute the patching region wavefunction and match constants

def wavefunction_patching_region_1_morse(x):
    F0 = morse_derivative(morse_tp[0])
    alpha = np.cbrt(2*mu*F0/(hbar**2))
    z = (x - morse_tp[0]) * alpha
    ai, aip, bi, bip = scipy.special.airy(z)
    globals()["A_morse_1"] = (np.sqrt(pi) / np.power(2 * mu * hbar * F0, 1/6)) * morse_ct
    globals()["morse_ct_qt_1"] = A_morse_1 * ( np.power(2 * mu * hbar * F0 , 1/6) / (2 * np.sqrt(pi)) )
    return A_morse_1 * ai 
"""
def wavefunction_patching_region_1_lenard_jones(x):
    F0 = lenard_jones_derivative(lenard_jones_tp[0])
    alpha = np.cbrt(2*mu*F0/(hbar**2))
    z = (x - lenard_jones_tp[0]) * alpha
    ai, aip, bi, bip = scipy.special.airy(z)
    globals()["A_lenard_jones_1"] = (np.sqrt(pi) / np.power(2 * mu * hbar * F0, 1/6)) * lenard_jones_ct
    globals()["lenard_jones_ct_qt_1"] = A_lenard_jones_1 * ( np.power(2 * mu * hbar * F0 , 1/6) / (2 * np.sqrt(pi)) )
    return A_lenard_jones_1 * ai 
"""
def wavefunction_patching_region_1_buckingham(x):
    F0 = buckingham_derivative(buckingham_tp[0])
    alpha = np.cbrt(2*mu*F0/(hbar**2))
    z = (x - buckingham_tp[0]) * alpha
    ai, aip, bi, bip = scipy.special.airy(z)
    globals()["A_buckingham_1"] = (np.sqrt(pi) / np.power(2 * mu * hbar * F0, 1/6)) * buckingham_ct
    globals()["buckingham_ct_qt_1"] = A_buckingham_1 * ( np.power(2 * mu * hbar * F0 , 1/6) / (2 * np.sqrt(pi)) )
    return A_buckingham_1 * ai 

def wavefunction_patching_region_1_buckingham_coulomb(x):
    F0 = buckingham_coulomb_derivative(buckingham_coulomb_tp[0])
    alpha = np.cbrt(2*mu*F0/(hbar**2))
    z = (x - buckingham_coulomb_tp[0]) * alpha
    ai, aip, bi, bip = scipy.special.airy(z)
    globals()["A_buckingham_coulomb_1"] = (np.sqrt(pi) / np.power(2 * mu * hbar * F0, 1/6)) * buckingham_coulomb_ct
    globals()["buckingham_coulomb_ct_qt_1"] = A_buckingham_coulomb_1 * ( np.power(2 * mu * hbar * F0 , 1/6) / (2 * np.sqrt(pi)) )
    return A_buckingham_coulomb_1 * ai 

def wavefunction_patching_region_2_morse(x):
    F0 = morse_derivative(morse_tp[1])
    alpha = np.cbrt(2*mu*F0/(hbar**2))
    z = (x - morse_tp[1]) * alpha
    ai, aip, bi, bip = scipy.special.airy(z)
    globals()["A_morse_2"] = (np.sqrt(pi) / np.power(2 * mu * hbar * F0, 1/6)) * morse_ct
    globals()["morse_ct_qt_2"] = A_morse_2 * ( np.power(2 * mu * hbar * F0 , 1/6) / np.sqrt(pi) )
    return A_morse_2 * ai 
"""
def wavefunction_patching_region_2_lenard_jones(x):
    F0 = lenard_jones_derivative(lenard_jones_tp[1])
    alpha = np.cbrt(2*mu*F0/(hbar**2))
    z = (x - lenard_jones_tp[1]) * alpha
    ai, aip, bi, bip = scipy.special.airy(z)
    globals()["A_lenard_jones_2"] = (np.sqrt(pi) / np.power(2 * mu * hbar * F0, 1/6)) * lenard_jones_ct
    globals()["lenard_jones_ct_qt_2"] = A_lenard_jones_2 * ( np.power(2 * mu * hbar * F0 , 1/6) / np.sqrt(pi) )
    return A_lenard_jones_2 * ai 
"""
def wavefunction_patching_region_2_buckingham(x):
    F0 = buckingham_derivative(buckingham_tp[1])
    alpha = np.cbrt(2*mu*F0/(hbar**2))
    z = (x - buckingham_tp[1]) * alpha
    ai, aip, bi, bip = scipy.special.airy(z)
    globals()["A_buckingham_2"] = (np.sqrt(pi) / np.power(2 * mu * hbar * F0, 1/6)) * buckingham_ct
    globals()["buckingham_ct_qt_2"] = A_buckingham_2 * ( np.power(2 * mu * hbar * F0 , 1/6) / np.sqrt(pi) )
    return A_buckingham_2 * ai 

def wavefunction_patching_region_2_buckingham_coulomb(x):
    F0 = buckingham_coulomb_derivative(buckingham_coulomb_tp[1])
    alpha = np.cbrt(2*mu*F0/(hbar**2))
    z = (x - buckingham_coulomb_tp[1]) * alpha
    ai, aip, bi, bip = scipy.special.airy(z)
    globals()["A_buckingham_coulomb_2"] = (np.sqrt(pi) / np.power(2 * mu * hbar * F0, 1/6)) * buckingham_coulomb_ct
    globals()["buckingham_coulomb_ct_qt_2"] = A_buckingham_coulomb_2 * ( np.power(2 * mu * hbar * F0 , 1/6) / np.sqrt(pi) )
    return A_buckingham_coulomb_2 * ai 

##############################################################################

# Glue the wavefunctions for each potential

def morse_wavefunction(x,n):
    if 0<x <= x_1_left_morse:
        return wavefunction_quantum_region_1_morse(x,n)
    if x_1_left_morse < x <= x_2_left_morse:
        return wavefunction_patching_region_1_morse(x,n)
    if x_2_left_morse < x <= x_1_right_morse:
        return wavefunction_classical_region_morse(x,n)
    if x_1_right_morse < x <= x_2_right_morse:
        return wavefunction_patching_region_2_morse(x,n)
    if x_2_right_morse < x:
        return wavefunction_quantum_region_2_morse(x)
    return 0
"""
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
"""
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

def buckingham_coulomb_wavefunction(x,n):
    if 0<x <= x_1_left_buckingham_coulomb:
        return wavefunction_quantum_region_1_buckingham_coulomb(x,n)
    if x_1_left_buckingham_coulomb < x <= x_2_left_buckingham_coulomb:
        return wavefunction_patching_region_1_buckingham_coulomb(x,n)
    if x_2_left_buckingham_coulomb < x <= x_1_right_buckingham_coulomb:
        return wavefunction_classical_region_buckingham_coulomb(x,n)
    if x_1_right_buckingham_coulomb < x <= x_2_right_buckingham_coulomb:
        return wavefunction_patching_region_2_buckingham_coulomb(x,n)
    if x_2_right_buckingham_coulomb < x:
        return wavefunction_quantum_region_2_buckingham_coulomb(x)
    return 0

##############################################################################

# Fix the constants for the wavefunction through the normalisation condition applied one the 0

def morse_normalisation(n):
    initial_guesses = [0, 1, 2, 3] # Initial guesses for the energy value
    def normaliser(ct):
        globals()["morse_ct"] = ct
        return np.sqrt(np.integrate(0, np.inf, np.abs(morse_wavefunction(x_morse,n))**2, x_morse)) - 1
    globals()["morse_ct"] = fsolve(normaliser, initial_guesses) 
    return None
"""    
def lenard_jones_normalisation(n):
    initial_guesses = [0, 1, 2, 3] # Initial guesses for the energy value
    def normaliser(ct):
        globals()["lenard_jones_ct"] = ct
        return quad(lambda x: np.abs(lenard_jones_wavefunction(x,n))**2, 0, np.inf, epsabs=1e-12, epsrel=1e-12) - 1
    globals()["lenard_jones_ct"] = fsolve(normaliser, initial_guesses) 
    return None
"""    
def buckingham_normalisation(n):
    initial_guesses = [0, 1, 2, 3] # Initial guesses for the energy value
    def normaliser(ct):
        globals()["buckingham_ct"] = ct
        return np.sqrt(np.integrate(0, np.inf, np.abs(buckingham_wavefunction(x_buckingham,n))**2, x_buckingham)) - 1
    globals()["buckingham_ct"] = fsolve(normaliser, initial_guesses) 
    return None
    
def buckingham_coulomb_normalisation(n):
    initial_guesses = [0, 1, 2, 3] # Initial guesses for the energy value
    def normaliser(ct):
        globals()["buckingham_coulomb_ct"] = ct
        return np.sqrt(np.integrate(0, np.inf, np.abs(buckingham_coulomb_wavefunction(x_buckingham_coulomb,n))**2, x_buckingham_coulomb)) - 1
    globals()["buckingham_coulomb_ct"] = fsolve(normaliser, initial_guesses) 
    return None
    
##############################################################################

# Define the final wavefunction

def final_wavefunction_morse(x,n):
    morse_normalisation(n)
    return morse_wavefunction(x,n)
"""
def final_wavefunction_lenard_jones(x,n):
    lenard_jones_normalisation(n)
    return lenard_jones_wavefunction(x,n)
"""
def final_wavefunction_buckingham(x,n):
    buckingham_normalisation(n)
    return buckingham_wavefunction(x,n)

def final_wavefunction_buckingham_coulomb(x,n):
    buckingham_coulomb_normalisation(n)
    return buckingham_coulomb_wavefunction(x,n)

##############################################################################

# Compute the exact solution for the wavefunction and energies

def wavefunction_exact_morse(x,n):
    K = np.sqrt(2*morse_m*D_e)/(hbar*a)

    def whittaker_M(n,k,x):
        return np.exp(-x/2) * x**(k+1/2) * scipy.special.hyp1f1(k-n+0.5,1+2*k,x)
    
    def N(n):
        return (scipy.special.gamma(n+1) * scipy.special.gamma(2*K-2*n-1)) / (a * scipy.special.gamma(2*K-n))
    
    return whittaker_M(K,K-n-0.5,2*K*np.exp(-a*(x-x_e))) / np.sqrt(2*K*N(n)*np.exp(-a*(x-x_e)))
"""
def wavefunction_exact_lenard_jones(x,n):
    return None
"""
def wavefunction_exact_buckingham(x,n):
    return None

def wavefunction_exact_buckingham_coulomb(x,n):
    return None

def energy_exact_morse(n):
    a_0 = a*hbar/np.sqrt(mu)
    return D_e + 0.5*(np.sqrt(2*D_e)-(n+0.5)*a_0)**2
"""
def energy_exact_lenard_jones(n):
    return None
"""
def energy_exact_buckingham(n):
    return None

def energy_exact_buckingham_coulomb(n):
    return None
"""
##############################################################################

# Plot the wavefunction and energies
print(final_wavefunction_lenard_jones(1e-10,1))
for n in range(N):
   # if potencial == 'm':
    #    plt.plot(x_morse, final_wavefunction_morse(x_morse,n), label='Morse Potential')
    if potencial == 'lj':
        plt.plot(x_lj, [final_wavefunction_lenard_jones(x,n) for x in x_lj], label='Lenard-Jones Potential')
 #   if potencial == 'b':
  #      plt.plot(x_buckingham, final_wavefunction_buckingham(x_buckingham,n), label='Buckingham Potential')
   # if potencial == 'bc':
    #    plt.plot(x_buckingham_coulomb, final_wavefunction_buckingham_coulomb(x_buckingham_coulomb,n), label='Buckingham-Coulomb Potential')
    plt.xlabel('Position - r')
    plt.ylabel('Wavefunction')
    plt.title('Wavefunction for all Potentials in the n state')
    plt.legend()
    plt.show()


for n in range(N):
#    if potencial == 'm':
 #       plt.pyplot.scatter(n, energy_levels_morse(n), label='Morse Potential',color='blue')
    if potencial == 'lj':
        plt.pyplot.scatter(n, energy_levels_lenard_jones(n).x[0], label='Lenard-Jones Potential',color='red')
  #  if potencial == 'b':
   #     plt.pyplot.scatter(n, energy_levels_buckingham(n), label='Buckingham Potential',color='green')
    #if potencial == 'bc':
     #   plt.pyplot.scatter(n, energy_levels_buckingham_coulomb(n), label='Buckingham-Coulomb Potential',color='black')

plt.xlabel('Energy Level')
plt.ylabel('Energy Value [J]')
plt.title('First 10 Energy Levels for all Potentials')
plt.legend()
plt.show()

##############################################################################

# Compute the error in the energies and wavefunction
"""
def morse_wavefunction_error(x,n):
    return abs(final_wavefunction_morse(x,n) - wavefunction_exact_morse(x,n))
"""
def lenard_jones_wavefunction_error(x,n):
    return abs(final_wavefunction_lenard_jones(x,n) - wavefunction_exact_lenard_jones(x,n))
"""
def buckingham_wavefunction_error(x,n):
    return abs(final_wavefunction_buckingham(x,n) - wavefunction_exact_buckingham(x,n))

def buckingham_coulomb_wavefunction_error(x,n):
    return abs(final_wavefunction_buckingham_coulomb(x,n) - wavefunction_exact_buckingham_coulomb(x,n))

def energy_error_morse(n):
    return abs(energy_levels_morse(n) - energy_exact_morse(n))

def energy_error_lenard_jones(n):
    return abs(energy_levels_lenard_jones(n).x[0] - energy_exact_lenard_jones(n))

def energy_error_buckingham(n):
    return abs(energy_levels_buckingham(n) - energy_exact_buckingham(n))

def energy_error_buckingham_coulomb(n):
    return abs(energy_levels_buckingham_coulomb(n) - energy_exact_buckingham_coulomb(n))

##############################################################################

# Plot the error in the energies and wavefunction

if potencial == 'm':
    plt.plot(x_morse, morse_wavefunction_error(x_morse,n), label='Morse Potential')
if potencial == 'lj':
    plt.plot(x_lj, lenard_jones_wavefunction_error(x_lj,n), label='Lenard-Jones Potential')
if potencial == 'b':
    plt.plot(x_buckingham, buckingham_wavefunction_error(x_buckingham,n), label='Buckingham Potential')
if potencial == 'bc':
    plt.plot(x_buckingham_coulomb, buckingham_coulomb_wavefunction_error(x_buckingham_coulomb,n), label='Buckingham-Coulomb Potential')
plt.xlabel('Position - r')
plt.ylabel('Wavefunction Error')
plt.title('Wavefunction Error for all Potentials')
plt.legend()
plt.show()

for n in range(N):
    if potencial == 'm':
        plt.pyplot.scatter(n, energy_error_morse(n), label='Morse Potential',color='blue')
    if potencial == 'lj':
        plt.pyplot.scatter(n, energy_error_lenard_jones(n), label='Lenard-Jones Potential',color='red')
    if potencial == 'b':
        plt.pyplot.scatter(n, energy_error_buckingham(n), label='Buckingham Potential',color='green')
    if potencial == 'bc':
        plt.pyplot.scatter(n, energy_error_buckingham_coulomb(n), label='Buckingham-Coulomb Potential',color='black')

plt.xlabel('Energy Level')
plt.ylabel('Energy Value [J]')
plt.title('Speed of Oscillation for all Potentials')
plt.legend()
plt.show()

##############################################################################"""

'''