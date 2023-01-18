
import hoomd
import os
import itertools
import math
import gsd.hoomd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
mpl.style.use('ggplot')

#Variables
dt = 
gamma = 
gamma_r = 
args_temp=1 
#
activity = [tuple((fact*np.cos(angles_bath[i]),fact*np.sin(angles_bath[i]),0)) for i in range(npart_bath)]
active_force = hoomd.md.force.Active(group=ag,f_lst=activity,orientation_link=True,orientation_reverse_link=False,rotation_diff=0,seed=np.random.randint(low=1, high=2**31-1)) # DON'T CHANGE THIS!
nl.update_rcut()

# Setup integrator
hoomd.md.Integrator(dt=dt)
bd = hoomd.md.methods.Brownian(group=ag,kT=args_temp,seed=np.random.randint(low=1, high=2**31-1))
bd.set_gamma('A',gamma=gamma)
bd.set_gamma_r('A',gamma_r=gamma_r)