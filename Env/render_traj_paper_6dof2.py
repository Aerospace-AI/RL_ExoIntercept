import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import attitude_utils as attu
import env_utils as envu


def render_traj(traj, vf=None, scaler=None):
    t = np.asarray(traj['t'])
    t1 = t[0:-1]

    pos = np.asarray(traj['r_tm'])
    vel = np.asarray(traj['v_tm'])
    norm_pos = np.linalg.norm(pos,axis=1)
    norm_vel = np.linalg.norm(vel,axis=1)

    fig1 = plt.figure(1,figsize=plt.figaspect(0.5))
    fig1.clear()
    plt.figure(fig1.number)
    fig1.set_size_inches(8, 8, forward=True)
    gridspec.GridSpec(4,2)



    # don't plot last where we go out of FOV for done
    plt.subplot2grid( (4,2) , (0,0) )
    c0 = np.asarray(traj['pixel_coords'])[:,0] / 96 * 1.474 * 180 / np.pi
    c1 = np.asarray(traj['pixel_coords'])[:,1] / 96 * 1.474 * 180 / np.pi
    plt.plot(t[0:-1],c0[0:-1],'r',label=r'$\theta_u$' )  
    plt.plot(t[0:-1],c1[0:-1],'b',label=r'$\theta_v$' )
    plt.legend(bbox_to_anchor=(0., 0.97, 1., .102), loc=3,
            ncol=5, mode="expand", borderaxespad=0.)
    plt.gca().set_ylabel('Seeker Angles (deg)')
    plt.gca().set_xlabel("Time (s)")
    plt.grid(True)

    plt.subplot2grid( (4,2) , (0,1) ) 
    of0 = 10*np.asarray(traj['optical_flow'])[:,0] / 96 * 1.474 * 180 / np.pi
    of1 = 10*np.asarray(traj['optical_flow'])[:,1] / 96 * 1.474 * 180 / np.pi
    plt.plot(t[0:-1],of0[0:-1],'r',label=r'$d\theta_{u}/dt$')
    plt.plot(t[0:-1],of1[0:-1],'b',label=r'$d\theta_{v}/dt$')

    plt.legend(bbox_to_anchor=(0., 1.00, 1., .102), loc=3,
            ncol=5, mode="expand", borderaxespad=0.)
    plt.gca().set_xlabel("time (s)")
    #plt.gca().set_ylabel('dTheta / dt (deg/s)')
    plt.gca().set_ylabel('Seeker Angle\n Rate of Change\n (deg/s)')
    plt.grid(True)

    plt.subplot2grid( (4,2) , (1,0) )
    m = np.asarray(traj['theta_cv']) * 180 / np.pi
    plt.plot(t,m,'r',label=r'$\theta_{BV}$')#Theta_CV')
    plt.legend(bbox_to_anchor=(0., 1.00, 1., .102), loc=3,
            ncol=5, mode="expand", borderaxespad=0.)
    plt.gca().set_xlabel("Time")
    plt.gca().set_ylabel('Angle Between Missile\nVelocity and \nMissile X Axis (deg)')
    plt.grid(True)

    thrust = np.asarray(traj['bf_thrust'])
    #print(thrust.shape)
    plt.subplot2grid( (4,2) , (1,1) )
    plt.plot(t,thrust[:,0],'r',label='-Y')
    plt.plot(t,thrust[:,1],'b',label='+Y')
    plt.plot(t,thrust[:,2],'g',label='+Z')
    plt.plot(t,thrust[:,3],'k',label='-Z')
    plt.legend(bbox_to_anchor=(0., 1.00, 1., .102), loc=3,
            ncol=5, mode="expand", borderaxespad=0.)
    plt.gca().set_ylabel('Divert Thrust (N)')
    plt.gca().set_xlabel("Time (s)")
    plt.grid(True)

    missile_acc = np.asarray(traj['missile_acc'])
    target_acc = np.asarray(traj['target_acc'])

    plt.subplot2grid( (4,2) , (2,0) )
    plt.plot(t,missile_acc[:,0],'r',label='X')
    plt.plot(t,missile_acc[:,1],'b',label='Y')
    plt.plot(t,missile_acc[:,2],'g',label='Z')
 
    plt.legend(bbox_to_anchor=(0., 1.00, 1., .102), loc=3,
            ncol=5, mode="expand", borderaxespad=0.)
    plt.gca().set_ylabel('Missile Acc. ($m/s^2$)')
    plt.gca().set_xlabel("Time (s)")
    plt.grid(True)

    plt.subplot2grid( (4,2) , (2,1) )
    plt.plot(t,target_acc[:,0],'r',label='X')
    plt.plot(t,target_acc[:,1],'b',label='Y')
    plt.plot(t,target_acc[:,2],'g',label='Z')

    plt.legend(bbox_to_anchor=(0., 1.00, 1., .102), loc=3,
            ncol=5, mode="expand", borderaxespad=0.)
    plt.gca().set_ylabel('Target Acc. ($m/s^2$)')
    plt.gca().set_xlabel("Time (s)")
    plt.grid(True)

    attitude = np.asarray(traj['attitude_321'])
    plt.subplot2grid( (4,2) , (3,0) )
    colors = ['r','b','k','g']
    plt.plot(t,attitude[:,0],'r', label='yaw')
    plt.plot(t,attitude[:,1],'b', label='pitch')
    plt.plot(t,attitude[:,2],'g', label='roll')

    plt.legend(bbox_to_anchor=(0., 1.00, 1., .102), loc=3,
            ncol=5, mode="expand", borderaxespad=0.)
    plt.gca().set_ylabel('Attitude (rad)')
    plt.gca().set_xlabel("Time (s)")
    plt.grid(True)

    w = np.asarray(traj['w'])
    plt.subplot2grid( (4,2) , (3,1) )
    plt.plot(t,w[:,0],'r',label='X')
    plt.plot(t,w[:,1],'b',label='Y')
    plt.plot(t,w[:,2],'g',label='Z')
     
    plt.legend(bbox_to_anchor=(0., 1.00, 1., .102), loc=3,
            ncol=5, mode="expand", borderaxespad=0.)
    plt.gca().set_ylabel('Rot Vel (rad/s)')
    plt.gca().set_xlabel("Time (s)")
    plt.grid(True)


    plt.tight_layout(h_pad=3.0)
    fig1.canvas.draw()



