#!/opt/miniconda3/bin/python3

import os
import itertools
import math
import gsd.hoomd
import hoomd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import freud
# mpl.style.use('ggplot')




# define the simulation box
# xy, xz, yz are tilt factors

Lx = 10
Ly = 5
Lz = 0
hoomd.Box(Lx=Lx, Ly=Ly, Lz=Lz, xy=0, xz=0, yz=0) 

# define number of particles
N_particles = 1
# define initial position of (all) particle
position = [(0, 0, 0)]

# Set the particles for the initial condition
# N_particles = 1**3
# spacing = 2
# K = math.ceil(N_particles**(1 / 3))
# L = K * spacing
# position = [(0,0,0)]

#x = numpy.linspace(-L / 2, L / 2, K, endpoint=False)
#position = list(itertools.product(x, repeat=3))


#dt proptery sets the step size
dt = 0.001
# Implement the Integrator
integrator = hoomd.md.Integrator(dt=dt)

snapshot = gsd.hoomd.Snapshot()
snapshot.particles.N = N_particles
snapshot.particles.position = position[0:N_particles]
snapshot.particles.type = ['A']
snapshot.particles.typeid = [0] * N_particles
snapshot.configuration.box = [Lx, Ly, Lz, 0, 0, 0]

with gsd.hoomd.open(name='ABP.gsd', mode='wb') as f:
    f.append(snapshot)



#Cell Is it necessary?
#cell = hoomd.md.nlist.Cell(buffer=0.4)

# Set force for active particle
# active_force_x = 0.1
# active_force_y = 0
# active_force_z = 0

# vx_0 = np.random.randint(100)  # random integer between 0 and 100
v0 = np.rand.rand(3)
v0[2] = 0  # no active force in z

active = hoomd.md.force.Active(filter=hoomd.filter.All())
active.active_force['A'] = (v0[0], v0[1], v0[2])
integrator.forces.append(active)



#activity = [tuple((fact*np.cos(angles_bath[i]),fact*np.sin(angles_bath[i]),0)) for i in range(npart_bath)]
#active_force = hoomd.md.force.active(group=ag,f_lst=activity,orientation_link=True,orientation_reverse_link=False,rotation_diff=0,seed=np.random.randint(low=1, high=2**31-1)) # DON'T CHANGE THIS!

# Integrator methods brownian
#bd = hoomd.md.integrate.brownian(group=ag,kT=args.temp,seed=np.random.randint(low=1, high=2**31-1))
# bd = hoomd.md.methods.Brownian(group=ag,kT=1.0, alpha=1.0)
# bd.gamma.default = 2.0  
# bd.gamma_r.default = [1.0,2.0,3.0]
# integrator.methods.append(bd)

bd = hoomd.md.methods.Brownian(
        filter=hoomd.filter.All(),
        kT=1,
        alpha=None,
        default_gamma=1.0,
        default_gamma_r=(1.0, 1.0, 1.0)
        )
integrator.methods.append(bd)


# define 

# Initialize the simulation
# gpu = hoomd.device.GPU()
cpu = hoomd.device.CPU()
sim = hoomd.Simulation(device=cpu, seed=42)
sim.create_state_from_gsd(filename="ABP.gsd")

# Assign the integrator to the simulation
sim.operations.integrator = integrator


#sim.create_state_from_gsd(filename='inital.gsd')

# Run the simulation
# sim.run(10000)

#snapshots = []

# snapshot = gsd.hoomd.Snapshot()
# snapshot.particles.N = N_particles
# snapshot.particles.position = position[0:N_particles]
# snapshot.particles.type = ['A']
# snapshot.particles.typeid = [0] * N_particles
# snapshot.configuration.box = [Lx, Ly, Lz, 0, 0, 0]


gsd_writer = hoomd.write.GSD(
        filename="ran.gsd",
        trigger=hoomd.trigger.Periodic(1),
        mode="wb"
        )
sim.operations.writers.append(gsd_writer)

sim.run(100)


# for producing multiple checkpoint files use following:
# steps = 100
# for i in range(steps):
#     sim.run(1)
#     hoomd.write.GSD.write(
#             state=sim.state,
#             filename="out/{}.gsd".format(i),
#             mode="wb"
#             )


arr = []

# Analyzing the trajectory with freud package
box = freud.Box.from_box(snapshot.configuration.box)
MSD = freud.msd.MSD(box)
traj = gsd.hoomd.open('ran.gsd')
for frame in traj:
    pos=frame.particles.position
    vel=frame.particles.velocity

    x = pos[0][0]
    y = pos[0][1]
    z = pos[0][2]
    vx = vel[0][0]
    vy = vel[0][1]
    vz = vel[0][2]
    arr.append([x, y, z, vx, vy, vz])

arr = np.array(arr)
arr = np.transpose(arr)

print(arr[0])  # x
print(arr[3])  # vx

r = np.sqrt(**2 + ydata**2)
diff = np.diff(r) #this calculates r(t + dt) - r(t)
diff_sq = diff**2
MSD = np.mean(diff_sq)

# um np array zu speichern
# np.safetxt("test.txt", arr)

# def set_figsize(width):
#     return (width, width)
#
# fig, ax = plt.subplots(1, 1, figsize=set_figsize(300))

# fig = plt.figure()
# ax = fig.add_subplots()

# ax.plot(arr[0], arr[1])
# plt.show()
# fig.savefig("figure.png")



        #MSD.compute(position=pos)
# Since the radial distribution function is in practice computed as a histogram, we must specify the histogram bin widths and the largest
# interparticle distance to include in our calculation



#MSD.compute(position(100,1,3),images=(100,1,3))
print('finished')

# We can now access the data through properties of the rdf object.

#r = rdf.bin_centers
#y = rdf.rdf

# Plot

#fig, ax = pltp.subplots()
#rdf.plot(ax=ax)
