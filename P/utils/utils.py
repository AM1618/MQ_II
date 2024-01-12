"""
Module for utilities used in the project.
"""
import numpy as np



def patching(potential, pot_derivative, tp :list, oscillation_speed, energy : float, epsilon : float):
    def lin_approx(x,tp2):
        return pot_derivative(tp2) * (x-tp2) + potential(tp2)

    final_points = []
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
    region_left = []
    region_right = []
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
    region_left = []
    region_right = []
    for x in np.linspace(tp[1]-size_left2, tp[1], int(abs(size_left2)/1e-7)):
        if oscillation_speed(x, energy) < epsilon:
            region_left.append(x)
    for x in np.linspace(tp[1], tp[1] + size_right2, int(abs(size_right2)/1e-7)):
        if oscillation_speed(x, energy) < epsilon:
            region_right.append(x)
    final_points.append((max(region_left),min(region_right)))

    return final_points