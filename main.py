import matplotlib.pyplot as plt
import time as t
import sys
from math import sqrt
from numpy import exp

plt.style.use('dark_background')

def integrate(time, array):
    resArray = [0]
    for n in range(len(time) - 1):
        resArray.append(
            resArray[-1] + 0.5 * (array[n + 1] + array[n]) * (time[n + 1] - time[n])
        )
    return resArray

# Variables
dryMass = 1 # kg
wetMass = 9 # kg
burnTime = .5 # seconds
g = 9.81 # m/s^2

# adding drag
rho_0 = 1.225  # Sea level air density (kg/m^3)
H = 8500  # Scale height for Earth's atmosphere (m)
Cd = 0.5  # Drag coefficient
A = 0.25 # Rocket cross-sectional area (m^2)

propellantMass = wetMass - dryMass
massFlowRate = propellantMass / burnTime
avgThrust = massFlowRate * g  # Calculating average thrust directly

# Generating the time list
time = [round(x * 0.1, 1) for x in range(1000)]

# Finding the index where burnTime occurs in the time list
index = next(i for i, t in enumerate(time) if t >= burnTime)

# Generating the thrust list
thrust = [avgThrust if t <= burnTime else 0 for t in time]

# Generating the mass list
mass = [(wetMass - t * massFlowRate) if i < index else dryMass for i, t in enumerate(time)]

# Generating the acceleration list and taking integrals to get velocity and displacement
# No drag
acceleration = [thrust[i] / mass[i] - 9.81 for i in range(len(time))]
velocity = integrate(time, acceleration)
displacement = list(map(lambda x : 0 if x < 0 else x, integrate(time, velocity))) # 0 if 'negative val' for displacement


# Recalculate for drag after getting a model for displacement
# Calculate the average velocity during burn time
avg_velocity_burn = sum(velocity[:index + 1]) / len(velocity[:index + 1])

# Calculate the average altitude during burn time
avg_altitude_burn = sum(displacement[:index + 1]) / len(displacement[:index + 1])

# Calculate the average air density during burn time
avg_rho_burn = rho_0 * exp(-avg_altitude_burn / H)

# Calculate the average drag during burn time
avgDrag = 0.5 * avg_rho_burn * avg_velocity_burn**2 * Cd * A

# Adjust the acceleration to include drag
acceleration = [(thrust[i] - (0.5 * rho_0 * exp(-displacement[i] / H) * velocity[i]**2 * Cd * A)) / mass[i] - g for i in range(len(time))]

# Recalculate velocity and displacement with drag included
velocity = integrate(time, acceleration)
displacement = list(map(lambda x: 0 if x < 0 else x, integrate(time, velocity)))  # 0 if 'negative val' for displacement

apex_idx = displacement.index(max(displacement))
apex = round(displacement[apex_idx],1)

descent_time = round(sqrt(2 * apex / g),1)

launch_idx = next(i for i, x in enumerate(velocity) if x > 0) -1

ascent_time = (apex_idx - launch_idx) / 10

airtime = ascent_time + descent_time

landing_idx = int(airtime) * 10

time = time[:landing_idx]
acceleration = acceleration[:landing_idx]
velocity = velocity[:landing_idx]
displacement = displacement[:landing_idx]

# Plotting
plt.plot(time,acceleration)
plt.plot(time, displacement)
plt.plot(time, velocity)
plt.legend(["Acceleration","Vertical Displacement", "Velocity"])
plt.xlabel("Time")
print('Launch Apex:', apex, 'meters')
t.sleep(1)
print('Total airtime:', airtime, 'seconds') # no air resistance estimation
t.sleep(1)
print('Maximum upwards velocity:', round(max(velocity), 1), 'meters/second')
t.sleep(1)
sys.stdout.write('Graph displaying in 3 ')
t.sleep(1)
sys.stdout.write('2 ')
t.sleep(1)
sys.stdout.write('1 ')
t.sleep(1)
sys.stdout.write('...')
t.sleep(1)
plt.savefig('launch-sim.png')
plt.show()
t.sleep(3)
exit()

