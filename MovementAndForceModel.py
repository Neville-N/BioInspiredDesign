# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 15:05:02 2021

@author: Neville
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox

fig3, ax3 = plt.subplots(num=3, clear=True)
fig2, ax2 = plt.subplots(num=2, clear=True)
fig1, ax1 = plt.subplots(num=1, clear=True)
ax1.set_aspect("equal")
PI = np.pi

# Initial values
initOffset = 0.48
initLength = 3.6
initLback = 5
initLengthActuator = 4.4
initHingeHeight = 3.0
angleMin = 0.9197
angleMax = 1.673

Ncrosses = 6  # Number of crosses the bridge has 
crossBeamWeight = 36.1  # Wight of a bamboo beam that forms a cross [kg]
floorPlateLength = 1.4  # [kg]
floorPlateWeight = 2.8  # [kg]
counterweight = 1.1e3  # Mass of counterweight [kg]

class Beam:
    """Class for Beam objects which stores and calculates the posistion of a beam.

    Attributes
    ----------
    A: ndarray
        1D array containing the position of the hinge on the short side [x, y]
    B: ndarray
        1D array containing the position of the hinge on the long side [x, y]
    C: ndarray
        1D array containing the position of the hinge in the middle [x, y]
    com: ndarray
        Position of the center of mass of the beam [x, y]
    length: float
        Length of the beam [m]
    offset: float
        Relative distance from hinge A to the middle hinge C.
        Generally 0<offset<=0.5
    angle : float
        Angle of vector (B-A) with X-axis [rad]
    rotation: ndarray
        Rotation array [cos(angle), sin(angle)]
    distAC: float
        Distance from hinge A to hinge C [m]. distAC = offset * length
    distBC: float
        Distance from hinge B to hinge C [m]. distBC = (1-offset) * length
    weight: float
        Weight of the beam [kg]
    color: str
        Color in which the beam is drawn in.
        Must be a color from matplotlib's list of named colors.
        https://matplotlib.org/stable/gallery/color/named_colors.html
    line: lines.Line2D
        Line2d object of matplotlib module to be updated to redraw line.

    Methods
    -------
    createFromAB(A, B):
        Update coordinates when given location of A and B.

    """

    def __init__(
        self,
        A=None,
        B=None,
        length=1,
        offset=0.45,
        angle=0,
        weight=None,
        color="black",
        drawHinge=True,
    ):
        """Construct Beam object."""
        self.angle = angle
        self.length = length
        self.offset = offset
        self.rotation = np.array([np.cos(angle), np.sin(angle)])

        self.distAC = offset * length
        self.distBC = (1 - offset) * length

        # Pick a method for calculating hinge points
        if A is not None and B is not None:
            self.createFromAB(A, B)
        elif A is not None:
            self.createFromA(A)
        elif B is not None:
            self.createFromB(B)
        else:
            print("error Both A and B are null on creating")
            return

        self.com = (self.A + self.B) / 2  # Center of gravity

        self.width = 0.1
        self.depth = 0.1
        self.density = 630
        self.volume = self.length * self.width * self.depth  # rough estimate

        if weight is not None:
            self.weight = weight
        else:
            self.weight = self.volume * self.density

        self.color = color  # Color drawn in graph
        cor = np.array([self.A, self.B, self.C])
        cor = cor.T
        if not drawHinge:
            (self.line,) = ax1.plot(cor[0], cor[1], color=self.color)
        else:
            (self.line,) = ax1.plot(cor[0], cor[1], color=self.color, marker="o")

    def createFromAB(self, A, B):
        """Update coordinates when given location of A and B."""
        self.A = A
        self.B = B
        self.C = self.offset * A + (1 - self.offset) * B
        self.angle = pointAngle(A, B)
        self.com = (self.A + self.B) / 2

    def createFromA(self, A):
        """Update coordinates when given location A and knowing the angle."""
        self.A = A
        self.B = A + self.length * self.rotation
        self.C = self.A + self.distAC * self.rotation
        self.com = (self.A + self.B) / 2  # Center of gravity

    def createFromB(self, B):
        """Update coordinates when given location B and knowing the angle."""
        self.B = B
        self.A = B - self.length * self.rotation
        self.C = self.B - self.distBC * self.rotation
        self.com = (self.A + self.B) / 2  # Center of gravity

    def updateLengthOffset(self, length, offset):
        """Update the length and offset and the params affected by this."""
        self.length = length
        self.offset = offset
        self.distAC = self.offset * self.length
        self.distBC = (1 - self.offset) * self.length
        self.volume = length * self.width * self.depth

    def updateAngle(self, angle):
        """Update the angle and rotation matrix."""
        self.angle = angle
        self.rotation = np.array([np.cos(angle), np.sin(angle)])

    def draw(self, ax):
        """Plot beam on given pyplot axes."""
        ax.plot([self.A[0], self.B[0]], [self.A[1], self.B[1]], color=self.color)

    def drawHinges(self, ax):
        """Plot beam, with hinges, on given pyplot axes."""
        ax.plot(
            [self.A[0], self.B[0], self.C[0]],
            [self.A[1], self.B[1], self.C[1]],
            color=self.color,
            marker="o",
        )

    def updateLine(self):
        """Update the line data when a slider has moved."""
        self.line.set_xdata([self.A[0], self.B[0], self.C[0]])
        self.line.set_ydata([self.A[1], self.B[1], self.C[1]])


def pointDist(p1, p2):
    """Return the pythagorean distance between two points."""
    return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def pointAngle(p1, p2, deg=False):
    """Return the angle, of the vector (p2-p1), with the x-axis.

    Returns
    -------
    Angle: Float
        Angle in radians (default.
    """
    j = 1j
    J = np.array([1, j])
    dp = p2 - p1
    angle = np.angle(np.dot(dp, J))
    if deg:
        angle *= 180 / PI
    return angle


def getAngleFromSides(oppSide, adjSides):
    """Calculate the angle of a triangle which opposes 'oppSide'.

    All sides are known, calculation based on the cosine rule.
    """
    return np.arccos((adjSides.dot(adjSides) - oppSide ** 2) / (2 * np.prod(adjSides)))


def getCircleIntersections(C1, C2, r1, r2):
    """Calculate intersection points of two cirlces.

    Circles have centers C1, C2 and radii r1, r2 respectively.

    Returns
    -------
    crossInt1 : np.array([x, y])
        Intersection option 1
    crossInt2 : np.array([x, y])
        Intersection option 2
    """
    d = pointDist(C1, C2)
    phi1 = getAngleFromSides(r2, np.array([r1, d]))
    gamma = pointAngle(C1, C2)
    rotCrossInt1 = np.array([np.cos(-phi1 + gamma), np.sin(-phi1 + gamma)])
    rotCrossInt2 = np.array([np.cos(phi1 + gamma), np.sin(phi1 + gamma)])

    crossInt1 = C1 + r1 * rotCrossInt1
    crossInt2 = C1 + r1 * rotCrossInt2

    return crossInt1, crossInt2


def updateFirstCross(
    firstAngle, length, offset, lengthBack, lengthHinge, hingeHeight, beams
):
    """Update the first two Beam objects in the list beams and returns this list.

    Parameters
    ----------
    firstAngle : Float
        Angle of the actuated beam, [rad].
    length : Float
        Length of a beam forming a cross, [m]
    offset : Float
        Fraction of the total length the middle hinge is away from hinge A.
        Generally: 0 < offset <= 0.5
    lengthBack : Float
        Distance actuator beam backhinge is placed backwards, [m].
    lengthHinge : Float
        Length of the actuator beam, [m].
    hingeHeight : Float
        Height of the actuator backhinge, [m].
    beams : [Beam, ...]
        List of all beams forming a cross on the bridge

    Returns
    -------
    beams : [Beam, ...]
        Updated list of Beam objects.

    """
    # Update beam going up
    beams[0].updateAngle(firstAngle)
    beams[0].updateLengthOffset(length, offset)
    beams[0].createFromA(np.array([0, 0]))

    # Position of the hinge of the actuator beam.
    hingeBack = np.array([-lengthBack, hingeHeight])

    # Possible positions for the connection between actuator beam and cross beam.
    option1, option2 = getCircleIntersections(
        hingeBack, beams[0].C, lengthHinge, beams[0].distBC
    )

    # Select highest of the two options
    if option1[1] > option2[1]:
        beam1B = option1
    else:
        beam1B = option2

    # Calculate the angle the first non actuated cross beam makes.
    angle0 = pointAngle(beams[0].C, beam1B)

    # Update the cross beam that goes down
    beams[1].updateLengthOffset(length, offset)
    beams[1].updateAngle(angle0)
    beams[1].createFromB(beam1B)
    return beams


def updateCross(i, length, offset, beams):
    """Update the cross beams starting at the second cross untill the end."""
    prevUp = beams[i * 2 - 2]
    prevDown = beams[i * 2 - 1]

    # Calculate options for new middle hinge point.
    option1, option2 = getCircleIntersections(
        prevDown.A, prevUp.B, prevDown.distAC, prevUp.distBC
    )

    # Pick the option that is not equal to the hinge point of the previous cross.
    if np.allclose(option1, prevUp.C):
        option = option2
    elif np.allclose(option2, prevUp.C):
        option = option1
    else:
        print(
            f"{i+1}th cross error: None of the options are corresponding to"
            "previous hinge point"
        )
        print("", option1, "\n", option2, "\n", prevUp.C)

    beamUpAngle = pointAngle(prevDown.A, option)

    # Update values of first cross beam.
    beams[i * 2 + 0].updateAngle(beamUpAngle)
    beams[i * 2 + 0].updateLengthOffset(length, offset)
    beams[i * 2 + 0].createFromA(prevDown.A)

    beamDownAngle = pointAngle(beams[i * 2 + 0].C, prevUp.B)

    # Update values of second cross beam.
    beams[i * 2 + 1].updateAngle(beamDownAngle)
    beams[i * 2 + 1].updateLengthOffset(length, offset)
    beams[i * 2 + 1].createFromB(prevUp.B)

    return beams


def updateFloorPlates(beamUp, beamDown, floorLength, floorBeams, i):
    """Update the coordinates of the floorBeams."""
    option1, option2 = getCircleIntersections(
        beamUp.A, beamDown.A, floorLength, floorLength
    )

    # Select highest option.
    if option1[1] > option2[1]:
        hingePoint = option1
    else:
        hingePoint = option2

    # Update beam objects.
    floorBeams[2 * i].createFromAB(beamUp.A, hingePoint)
    floorBeams[2 * i + 1].createFromAB(beamDown.A, hingePoint)

    return floorBeams


def updateBackBeams(lBack, lengthActuator, hingeHeight, backBeams, crossBeams):
    """Update the four beams in the back, only for visual purposes."""
    O0 = np.array([0, 0])
    O1 = np.array([-lBack, 0])
    O2 = np.array([-lBack, hingeHeight])

    backBeams[0].createFromAB(O0, O1)
    backBeams[1].createFromAB(O0, O2)
    backBeams[2].createFromAB(O1, O2)
    backBeams[3].createFromAB(O2, crossBeams[1].B)
    return backBeams


def totalWeight(beams):
    """Calculate the total weight of all beams in beams list."""
    weight = np.sum([beam.weight for beam in beams])
    return weight


def calcCOM(beams):
    """Calculate center of mass of all beams in the beams list.

    Returns
    -------
    COM: np.array([x, y])
        Center of mass.
    """
    COM = np.zeros(2)

    for beam in beams:
        COM += beam.com * beam.weight

    COM /= totalWeight(beams)
    return COM


# end functions


# Put beam objects without info in beamLists
crossBeams = []
floorBeams = []
backBeams = []
for i in range(Ncrosses * 2):
    crossBeams.append(
        Beam(
            A=np.array([0, 0]),
            B=np.array([1, 1]),
            weight=crossBeamWeight,
            length=initLength,
        )
    )
    floorBeams.append(
        Beam(
            A=np.array([0, 0]),
            B=np.array([1, 1]),
            weight=floorPlateWeight,
            length=floorPlateLength,
            color="blue",
            drawHinge=False,
        )
    )

backBeamsColors = ["yellow", "hotpink", "limegreen", "cyan"]
for i in range(4):
    backBeams.append(
        Beam(
            A=np.array([0, 0]),
            B=np.array([1, 1]),
            color=backBeamsColors[i],
            drawHinge=False,
        )
    )

minCOM = np.Inf
maxCOM = 0


# initiate slider stuff
ax3.grid()
ax3.set_xlabel("Distance spanned [m]")
ax3.set_ylabel("F [kN]")
ax2.grid()
ax2.set_xlabel("Distance spanned [m]")
ax2.set_ylabel("Potential energy [kJ]")
ax1.grid()
ax1.set_xlim((-1.1 * initLback, 6 * initLength))
ax1.set_ylim((-0.5 * initLength, 2.2 * initLength))
axcolor = "lightgoldenrodyellow"
ax1.margins(x=0.01)

# ax1.margins(y=0.25) # adjust the main plot to make room for the sliders

# axangleRange = plt.axes([0.25, 0.00, 0.60, 0.03], facecolor=axcolor)
# angleRange_slider = RangeSlider(
#     ax=axangleRange,
#     label='angle range [rad]',
#     valmin=0,
#     valmax=2/3*PI,
#     valinit=(angleMin, angleMax)
# )

axAngleMin = plt.axes([0.25, 0.0, 0.1, 0.03])
angleMin_textbox = TextBox(
    ax=axAngleMin, label="Min angle [rad]  ", initial=str(angleMin)
)

axAngleMax = plt.axes([0.55, 0.0, 0.1, 0.03])
angleMax_textbox = TextBox(
    ax=axAngleMax, label="Max angle [rad] ", initial=str(angleMax)
)

axangle = plt.axes([0.25, 0.03, 0.65, 0.03], facecolor=axcolor)
angle_slider = Slider(
    ax=axangle,
    label="Angle of first cross leg [rad]",
    valmin=0 * angleMin,
    valmax=1.2 * angleMax,
    valinit=angleMin,
)

# Make a horizontal slider to control the length
axlength = plt.axes([0.25, 0.09, 0.65, 0.03], facecolor=axcolor)
length_slider = Slider(
    ax=axlength,
    label="Length cross legs [m] (black)",
    valmin=0.01,
    valmax=4,
    valinit=initLength,
)

axoffset = plt.axes([0.25, 0.06, 0.65, 0.03], facecolor=axcolor)
offset_slider = Slider(
    ax=axoffset,
    label="Center hinge offset [%]",
    valmin=0.01,
    valmax=0.5,
    valinit=initOffset,
)

# axLback = plt.axes([0.25, 0.12, 0.1, 0.03])
# Lback_textbox = TextBox(
#     ax=axLback,
#     label='Length back [m] (yellow)  ',
#     initial=str(initLback)
# )

axLback = plt.axes([0.25, 0.12, 0.65, 0.03])
sliderLengthBack = Slider(
    ax=axLback,
    label="Length back [m] (yellow)",
    valmin=2,
    valmax=9,
    valinit=initLback,
)


# axLactuator = plt.axes([0.25, 0.15, 0.1, 0.03])
# Lactuator_textbox = TextBox(
#     ax=axLactuator,
#     label='Length actuator [m] (cyan)  ',
#     initial=str(initLengthActuator)
# )

axLactuator = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
sliderLengthActuator = Slider(
    ax=axLactuator,
    label="Length actuator [m] (cyan)",
    valmin=2,
    valmax=9,
    valinit=initLengthActuator,
)

# Make a horizontal slider to control hingeHeight
axHingeHeight = plt.axes([0.25, 0.18, 0.65, 0.03], facecolor=axcolor)
hingeHeight_slider = Slider(
    ax=axHingeHeight,
    label="Hinge Height [m] (green)",
    valmin=-1,
    valmax=3,
    valinit=initHingeHeight,
)

axItterate = plt.axes([0.25, 0.21, 0.08, 0.02])
itterateButton = Button(axItterate, "itterate", color=axcolor, hovercolor="0.975")

axClear = plt.axes([0.35, 0.21, 0.08, 0.02])
clearButton = Button(axClear, "clear", color=axcolor, hovercolor="0.975")

(COMpoint,) = ax1.plot(0, 0, marker="o", color="orange", label="COM")
(moveLinePlot,) = ax1.plot(0, 0, "g--", label="movementLine")


def updatePlus(val):
    """Reset minCOM and maxCOM."""
    global minCOM, maxCOM

    minCOM = np.Inf
    maxCOM = 0

    update(val)


def recalculate(
    firstAngle,
    length,
    offset,
    lBack,
    lengthActuator,
    hingeHeight,
    crossBeams,
    floorBeams,
    backBeams,
):
    """Calculate all positions of the beams."""
    crossBeams = updateFirstCross(
        firstAngle, length, offset, lBack, lengthActuator, hingeHeight, crossBeams
    )

    for i in range(1, Ncrosses):
        crossBeams = updateCross(i, length, offset, crossBeams)

    for i in range(Ncrosses):
        floorBeams = updateFloorPlates(
            crossBeams[2 * i], crossBeams[2 * i + 1], floorPlateLength, floorBeams, i
        )

    backBeams = updateBackBeams(
        lBack, lengthActuator, hingeHeight, backBeams, crossBeams
    )

    return crossBeams, floorBeams, backBeams


def update(val):
    """Redraw the bridge with the current values of all the sliders and input."""
    global crossBeams, floorBeams, backBeams, minCOM, maxCOM

    hingeHeight = hingeHeight_slider.val
    lengthActuator = sliderLengthActuator.val
    # lengthActuator = float(Lactuator_textbox.text)
    lBack = sliderLengthBack.val
    # lBack = float(Lback_textbox.text)
    offset = offset_slider.val
    length = length_slider.val
    firstAngle = angle_slider.val

    # angleMin = angleRange_slider.val[0]
    # angleMax = angleRange_slider.val[1]
    angleMin = float(angleMin_textbox.text)
    angleMax = float(angleMax_textbox.text)

    angle_slider.valmin = angleMin
    angle_slider.valmax = angleMax

    ax1.set_xlim((-1.1 * lBack, 6 * length))
    ax1.set_ylim((-0.5 * length, 2.2 * length))

    crossBeams, floorBeams, backBeams = recalculate(
        firstAngle,
        length,
        offset,
        lBack,
        lengthActuator,
        hingeHeight,
        crossBeams,
        floorBeams,
        backBeams,
    )

    for crossBeam in crossBeams:
        crossBeam.updateLine()

    for floorBeam in floorBeams:
        floorBeam.updateLine()

    for backBeam in backBeams:
        backBeam.updateLine()

    # Plot the movement line of top hinge
    H = np.linspace(0, 2 * PI, 120)
    moveLine = np.zeros((2, 120))

    for i, h in enumerate(H):
        moveLine[0, i] = -lBack + lengthActuator * np.cos(h)
        moveLine[1, i] = hingeHeight + lengthActuator * np.sin(h)

    moveLinePlot.set_xdata(moveLine[0])
    moveLinePlot.set_ydata(moveLine[1])

    com = calcCOM(crossBeams + floorBeams)

    COMpoint.set_xdata(com[0])
    COMpoint.set_ydata(com[1])

    minCOM = min(minCOM, com[1])
    maxCOM = max(maxCOM, com[1])

    RightSide = pointAngle(crossBeams[-1].A, crossBeams[-2].B, deg=True)
    LeftSide = pointAngle(crossBeams[0].A, crossBeams[1].B, deg=True)
    bridgeCurve = LeftSide - RightSide
    # print(crossBeams[-1].A, crossBeams[-2].B)
    # print(crossBeams[0].A, crossBeams[1].B)

    firstFloorPlateAngle = pointAngle(floorBeams[0].A, floorBeams[0].B, deg=True)

    print(
        "\ncurCOM: {3:.3g}, {4:.3g}\n"
        "minCOM: {0:.3g} m\n"
        "maxCOM: {1:.3g} m\n"
        "Weight: {2:.3g} kg\n"
        "First hinge dist: {5:.3g} m\n"
        "Last hinge: ({6:.4g}, {7:.4g})\n"
        "Bridge curvature: {8:.3g} deg\n"
        "First floor plate angle: {9:.3g} deg\n"
        "".format(
            minCOM,
            maxCOM,
            totalWeight(crossBeams + floorBeams),
            com[0],
            com[1],
            pointDist(crossBeams[0].A, crossBeams[1].A),
            crossBeams[-1].A[0],
            crossBeams[-1].A[1],
            bridgeCurve,
            firstFloorPlateAngle,
        )
    )

    fig1.canvas.draw_idle()


# angleRange_slider.on_changed(update)
angleMin_textbox.on_submit(update)
angleMax_textbox.on_submit(update)

angle_slider.on_changed(update)
length_slider.on_changed(updatePlus)
offset_slider.on_changed(updatePlus)

# Lback_textbox.on_submit(updatePlus)
# Lactuator_textbox.on_submit(updatePlus)
sliderLengthBack.on_changed(updatePlus)
sliderLengthActuator.on_changed(updatePlus)

hingeHeight_slider.on_changed(updatePlus)

update(0)


def itterateThroughAngles(event):
    """
    Itterate through all the valid positions of the bridge.

    Then calculate interesting properties to be graphed.
    """
    global crossBeams, floorBeams, backBeams
    length = length_slider.val
    offset = offset_slider.val
    hingeHeight = hingeHeight_slider.val

    lengthActuator = sliderLengthActuator.val
    lBack = sliderLengthBack.val
    # lengthActuator = float(Lactuator_textbox.text)
    # lBack = float(Lback_textbox.text)

    # angleMin = angleRange_slider.val[0]
    # angleMax = angleRange_slider.val[1]
    angleMin = float(angleMin_textbox.text)
    angleMax = float(angleMax_textbox.text)

    allAngles = np.linspace(angleMax, angleMin, 1001)

    COM = []
    firstBeamHeight = []
    spannedDist = []

    for angle in allAngles:
        crossBeams, floorBeams, backBeams = recalculate(
            angle,
            length,
            offset,
            lBack,
            lengthActuator,
            hingeHeight,
            crossBeams,
            floorBeams,
            backBeams,
        )
        movingBeams = crossBeams + floorBeams
        COM.append(calcCOM(movingBeams))
        firstBeamHeight.append(crossBeams[1].B[1])
        spannedDist.append(crossBeams[-1].A[0])

    COM = np.array(COM)
    comHeight = COM[:, 1]
    weightHeight = np.array(firstBeamHeight)
    spannedDist = np.array(spannedDist)

    g = 9.81  # Gravitational acceleration
    weigthWeight = counterweight  
    weightEnergy = weightHeight * g * weigthWeight * 1e-3  # In kilojoule
    COMenergy = comHeight * g * totalWeight(movingBeams) * 1e-3  # in kilojoule

    dspannedDist = spannedDist[:-1]
    dspannedDist += np.diff(spannedDist)[0] / 2

    dcomEnergy = np.gradient(COMenergy, spannedDist)
    dweightEnergy = np.gradient(weightEnergy, spannedDist)

    # Plot the the energies.
    ax2.plot(spannedDist, weightEnergy, label="Weight energy")
    ax2.plot(spannedDist, COMenergy, label="Bridge energy")
    ax2.legend()
    ax2.autoscale()
    ax2.set_title('Potential energy when expanding bridge')
    fig2.canvas.draw_idle()

    # Plot the derivative of the energy plot.
    ax3.plot(spannedDist[:-2], dweightEnergy[:-2], label="F_counterweight")
    ax3.plot(spannedDist[:-2], dcomEnergy[:-2], label="F_bridgeweight")
    ax3.plot(spannedDist[:-2], dcomEnergy[:-2] + dweightEnergy[:-2], label="F_total")
    ax3.legend()
    ax3.autoscale()
    ax3.set_title('Horizontal force working against expansion')
    fig3.canvas.draw_idle()


def clearAX(event):
    """Clear energy plots and redraw the grid."""
    ax2.clear()
    ax3.clear()
    ax2.grid()
    ax3.grid()
    fig2.canvas.draw_idle()
    fig3.canvas.draw_idle()


itterateButton.on_clicked(itterateThroughAngles)
clearButton.on_clicked(clearAX)

plt.show()
