import os
import itertools
import math
import gsd.hoomd
import hoomd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as npy
mpl.style.use('ggplot')

# Set the particles for the initial condition
N_particles = 1**3
spacing = 2
K = math.ceil(N_particles**(1 / 3))
L = K * spacing
position = [(0,0,0)]
#x = numpy.linspace(-L / 2, L / 2, K, endpoint=False)
#position = list(itertools.product(x, repeat=3))


#dt proptery sets the step size
dt = 0.001
# Implement the Integrator
integrator = hoomd.md.Integrator(dt=dt)


#Cell Is it necessary?
#cell = hoomd.md.nlist.Cell(buffer=0.4)

# Set force for active particle

active = hoomd.md.force.Active(filter=hoomd.filter.All())
active.active_force = [tuple((fact*np.cos(angles_bath[i]),fact*np.sin(angles_bath[i]),0)) for i in range(npart_bath)]
integrator.forces.append(active)
#activity = [tuple((fact*np.cos(angles_bath[i]),fact*np.sin(angles_bath[i]),0)) for i in range(npart_bath)]
#active_force = hoomd.md.force.active(group=ag,f_lst=activity,orientation_link=True,orientation_reverse_link=False,rotation_diff=0,seed=np.random.randint(low=1, high=2**31-1)) # DON'T CHANGE THIS!

# Integrator methods brownian
#bd = hoomd.md.integrate.brownian(group=ag,kT=args.temp,seed=np.random.randint(low=1, high=2**31-1))
bd = hoomd.md.methods.Brownian(group=ag,kT=1.0, alpha=1.0)
bd.gamma.default = 2.0  
bd.gamma_r.default = [1.0,2.0,3.0]
integrator.methods.append(bd)

# Assign the integrator to the simulation
sim.operations.integrator = integrator

# Initialize the simulation

gpu = hoomd.device.GPU()
sim = hoomd.Simulation(device=gpu,seed=1)
#sim.create_state_from_gsd(filename='inital.gsd')

# Run the simulation
sim.run(10000)
