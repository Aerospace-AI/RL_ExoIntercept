import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

def plot(traj, linewidth=0.5, axislimit_x=(-30000,30000), axislimit_y=(-30000,30000), axislimit_z=(10000,30000)):
   r_m = np.asarray(traj['position'])
   r_m -= np.mean(r_m)
   r_m_0 = r_m[0]
   r_t = r_m + np.asarray(traj['r_tm'])
   r_t_0 = r_t[0]
   print(r_m_0, r_t_0)
   plt.clf()
   fig = plt.figure()
   ax = fig.gca(projection='3d')
   ax.plot(r_m[:,0],r_m[:,1],r_m[:,2],'b',linewidth=linewidth)
   ax.plot(r_t[:,0],r_t[:,1],r_t[:,2],'r',linewidth=linewidth)
   ax.scatter(r_m_0[0], r_m_0[1], r_m_0[2] ,'b',label='Missile')
   ax.scatter(r_t_0[0], r_t_0[1], r_t_0[2] ,'r',label='Target')
   ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                  ncol=5, mode="expand", borderaxespad=0.)
   ax.set_xlabel("X (m)")
   ax.set_ylabel("Y (m)")
   ax.set_zlabel("Z (m)")
   ax.set_xlim3d(axislimit_x[0], axislimit_x[1])
   ax.set_ylim3d(axislimit_y[0], axislimit_y[1])
   ax.set_zlim3d(axislimit_z[0], axislimit_z[1])
   fontsize=9
   plt.gca().tick_params(axis='x', labelsize=fontsize)
   plt.gca().tick_params(axis='y', labelsize=fontsize)
   plt.gca().tick_params(axis='z', labelsize=fontsize)

   ax.legend
   fig.canvas.draw()
   plt.show()
