from potencials import lenard_jones, morse
import argparse



# Set up the argument parser
parser = argparse.ArgumentParser(description='Compute the energy levels of a diatomic molecule using the WKB approximation.')
parser.add_argument('potencial', type=str, help='Available potentials: \n Lennard-Jones (lj) \n Morse (m) \n Buckingham (b) \n Buckingham-Coulomb (bc)')
parser.add_argument('nivel', type=int, help='Energy level to be computed (int)')

# Parse the arguments
args = parser.parse_args()

# Assign the arguments to variables
potencial = args.potencial
En = args.nivel

x = [1,2,3,4,5,6,7,8,9,10]
y = []
# Jones 
for i in range(10): 
    print(lenard_jones.energy_levels(i))
    y.append(lenard_jones.energy_levels(i))

import matplotlib.pyplot as plt
plt.plot(x,y)
plt.show()

morse.plot_potencial()



