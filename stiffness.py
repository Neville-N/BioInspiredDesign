# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 14:17:13 2021

@author: Neville
"""

import numpy as np
import matplotlib.pyplot as plt

fig1, ax1 = plt.subplots(num=1, clear=True)
fig2, axs = plt.subplots(3, 2, sharex=True, num=2, clear=True)
fig3, axs2 = plt.subplots(2, 1, sharex=True, num=3, clear=True)


class Beam:
    E = 10.56e9  # youngs modulus [Pa]
    length = 0  # length of the beam [m]
    distance = 0  # horiizontal length of beam at angle [m]
    x0 = 0  # start coordinate [m]
    y0 = 0  # start coordinate [m]
    x1 = 0  # end coordinate [m]
    y1 = 0  # end coordinate [m]
    # Angle of the x beams with horizontal axis when fully extended [rad]
    angle = 0
    z = 0  # When walking over the bridge the left and right [m]
    vertical = 0  # Vertical height of the beam when rotated at angle [m]
    weight = 0  # weight of the beam [kg]
    g = 9.81  # [m/s^2]
    load = 0  # Load that the weight of the bridge produces [N/m]
    color = 'black'

    def __init__(self, start_x, start_y, z, angle_with_horizontal, length,
                 youngsMod=10.56e9,
                 height=0.15,
                 rho=630,
                 color='black'):
        self.x0 = start_x
        self.y0 = start_y
        self.z = z

        self.angle = angle_with_horizontal
        self.length = length

        self.x1 = start_x + self.length * np.cos(self.angle)
        self.y1 = start_y + self.length * np.sin(self.angle)
        self.distance = self.x1 - self.x0

        self.height = height
        self.width = height
        self.vertical = self.height / np.cos(self.angle)
        # print(self.width, self.height, self.vertical)

        self.E = youngsMod

        # self.weight = rho * self.length * self.width * self.height
        self.weight = 19
        self.load = self.weight * self.g / self.distance

        self.color = color

    def draw(self, ax):
        """Plot beam on given pyplot ax."""
        ax.plot([self.x0, self.x1], [self.y0, self.y1], color=self.color)

    def withinRange(self, x):
        """Return if x coordinate is within the length of the beam."""
        return self.x0 <= x and x <= self.x1

    def centerY(self, x):
        """Calculate the y position at x coordinate."""
        frac = (x - self.x0)/(self.x1 - self.x0)
        if abs(frac) > 1:
            print('error')
        return frac * self.y1 + (1 - frac) * self.y0

    def bottomTop(self, x):
        """Return lower and higher edge of the beam at x coordinate."""
        midY = self.centerY(x)
        return (midY - self.vertical/2, midY + self.vertical/2)

    def area(self, x):
        """Return crosssectional area in yz plane, currently const over length."""
        return self.vertical * self.width

    def centroidY(self, x):
        """Weighted center coordinate of beam."""
        return self.centerY(x) * self.area(x)

    def momentArea(self, x):
        """Return moment area, currently constant over length of beam."""
        return 1/12 * self.width * self.vertical**3

    def momentAreaAxis(self, x, y):
        return self.momentArea(x) + self.area(x)*(y - self.centerY(x))**2

    def momentAreaAxisE(self, x, y):
        return self.momentAreaAxis(x, y) * self.E


# Constants
ANGLE = 0.7068418697276234
NUM_CROSSES = int(6)
LENGTH = 3.6
PATHWIDTH = 0.2
FLOORTHICKNESS = 0.01

# Generate bridge elements
startX = 0
startY = LENGTH/2*np.sin(ANGLE)

beams = []

# Number of parralell beams per cross
Ncrosses = [ 2, 1, 1, 1, 1, 1] # r=0.15m, w=19kg

# create X beams
for i in range(NUM_CROSSES):
    #  Four cross beams
    j = 0
    while j < Ncrosses[i]:
        beams.append(Beam(startX, -startY, -PATHWIDTH/2, ANGLE, LENGTH))
        beams.append(Beam(startX, -startY, PATHWIDTH/2, ANGLE, LENGTH))
        beams.append(Beam(startX, startY, -PATHWIDTH/2, -ANGLE, LENGTH))
        beams.append(Beam(startX, startY, PATHWIDTH/2, -ANGLE, LENGTH))
        j += 1

    startX = beams[-1].x1

bridge_length = beams[-1].x1 + 0


# Iterate through length of bridge to calculate multiple values
# x-axis locations at which vals are calculated
Xarr = np.linspace(0, bridge_length, 14766)
dx = Xarr[1]  # stepsize
# y-axis Centroid of the  bridge around which it rotates
CentroidY = np.zeros_like(Xarr)
I = np.zeros_like(Xarr)
EI = np.zeros_like(Xarr)  # Yougs modules multiplied of moment area of bridge
Load = np.zeros_like(Xarr)  # Load on the bridge

for i, x in enumerate(Xarr):
    current_centroidY = 0
    total_area = 0
    for beam in beams:
        if beam.withinRange(x):
            total_area += beam.area(x)
            current_centroidY += beam.centroidY(x)
            Load[i] += beam.load
    CentroidY[i] = current_centroidY / total_area

    for beam in beams:
        if beam.withinRange(x):
            I[i] += beam.momentAreaAxis(x, CentroidY[i])
            EI[i] += beam.momentAreaAxisE(x, CentroidY[i])

#  Place extra static load on bridge
# humanL = 0.2
# humanWeight = 75  # kg
# humanStart = bridge_length - humanL
# humanStartInd = int(humanStart/dx)
# humanEndInd = humanStartInd + int(humanL/dx)
# humanLoad = humanWeight*9.81/humanL
# Load[humanStartInd:humanEndInd+2] += humanLoad


# Calculate shear force graph
ShearForce = np.array([np.sum(Load[i:])*dx for i in range(len(Xarr))])

# Integrate ShearForce to get moment graph
Moment = np.array([np.trapz(ShearForce[:i], x=Xarr[:i])
                  for i in range(len(Xarr))])
Moment -= Moment[-1]

# Calculate displacement
Rotation = np.zeros_like(Xarr)
for i, x in enumerate(Xarr):
    Rotation[i] = np.trapz(Moment[:i]*Xarr[:i]/EI[:i], x=Xarr[:i])
    # Displacement[i] = np.trapz(Moment*Xarr[i]/EI, x=Xarr)

# Calculate displacement
Displacement = np.zeros_like(Xarr)
for i, x in enumerate(Xarr):
    Displacement[i] = np.trapz(Rotation[:i], x=Xarr[:i])


# caclulate tension
bottomY = beams[0].y0
topY = beams[0].y1
Tension = np.zeros_like(Xarr)
for i, x in enumerate(Xarr):
    Tension[i] = -Moment[i]*(topY - CentroidY[i])/I[i]
print('Max Tensile stress:    {:.4g} MPa'.format(max(Tension)/1e6))

# calculate compression
Compression = np.zeros_like(Xarr)
for i, x in enumerate(Xarr):
    Compression[i] = -Moment[i]*(-bottomY + CentroidY[i])/I[i]
print('Max compression stress {:.4g} MPa'.format(max(Compression)/1e6))


#  Draw sideview of the bridge with lines as beams
for beam in beams:
    beam.draw(ax1)
ax1.axis('equal')
ax1.set_title('Side view bridge')
ax1.set_xlabel('x [m]')
ax1.set_ylabel('y [m]')

#  Graph bridge characteristics
axs[0, 0].plot(Xarr, EI, label='EI')
axs[0, 0].set_ylabel(r'E$\cdot$I')
axs[0, 0].grid()

axs[1, 0].plot(Xarr, Load, label='Load')
axs[1, 0].plot(0, 0)
axs[1, 0].set_ylabel('distributed \nload [N/m]')
axs[1, 0].grid()

axs[2, 0].plot(Xarr, ShearForce, label='Shear')
axs[2, 0].set_ylabel('Shear force\n[N]')
axs[2, 0].set_xlabel('X-coordinate (x)')
axs[2, 0].grid()

# axs[3, 0].plot(Xarr, CentroidY, label='Shear')
# axs[3, 0].set_ylabel('Centroid')
# axs[3, 0].set_xlabel('X-coordinate (x)')
# axs[3, 0].grid()

axs[0, 1].plot(Xarr, Moment, label='Bending Moment')
axs[0, 1].set_ylabel('Bending\nmoment [N*m]')
axs[0, 1].grid()

axs[1, 1].plot(Xarr, Rotation, label='Rotation')
axs[1, 1].set_ylabel('Rotation [rad]')
axs[1, 1].grid()

axs[2, 1].plot(Xarr, Displacement, label='Displacement')
axs[2, 1].set_ylabel('Displacement [m]')
axs[2, 1].set_xlabel('X-coordinate (x)')
axs[2, 1].grid()


axs2[0].plot(Xarr, Tension/1e6, label='Tension')
axs2[0].set_ylabel('Tensile stress [MPa]')
axs2[0].grid()

axs2[1].plot(Xarr, Compression/1e6, label='Compression')
axs2[1].set_xlabel('X-coordinate [m]')
axs2[1].set_ylabel('Compression stress [MPa]')
axs2[1].grid()

plt.show()
