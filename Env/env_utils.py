import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import attitude_utils as attu
from scipy import signal


def make_airframe(tau=0.5, eta=0.7, w_m=30, w_z=5):
    num = -w_m**2*np.asarray([1 , 0, -w_z**2])
    den = w_z**2*np.asarray([tau , 2*tau*eta*w_m+1 , tau*w_m**2+2*eta*w_m , w_m**2])
    tf = signal.TransferFunction(num,den)
    #print(tf)
    ss = tf.to_ss()
    #print(ss.A)
    #print(ss.B)
    #print(ss.C)
    airframe = {}
    airframe['A'] = ss.A 
    airframe['B'] = np.squeeze(ss.B) 
    airframe['C'] = np.squeeze(ss.C )
    return airframe

def rad2deg(x):
    return 180/np.pi*x

def deg2rad(x):
    return np.pi/180*x

def get_vc(r_tm, v_tm):
   vc = -r_tm.dot(v_tm)/np.linalg.norm(r_tm)
   return vc
 
def get_dlos(r_tm, v_tm ):
    vc = get_vc(r_tm, v_tm) 
    #dlos       =  r_tm * vc / np.linalg.norm(r_tm) ** 2
    r = np.linalg.norm(r_tm)
    if vc > 0.01:
        dlos = v_tm / r + r_tm * vc / r**2
    else:  
        dlos = np.zeros(3)
    return dlos

 
def print_vector(s,v,f):
    v = 1.0 * v
    if isinstance(v,float): 
         v = [v]
    s1 = ''.join(f.format(v) for k,v in enumerate(v))
    s1 = s + s1
    return s1


def rk4(t, x, xdot, h ):

    """
        t  :  time
        x  :  initial state
        xdot: a function xdot=f(t,x, ...)
        h  : step size

    """

    k1 = h * xdot(t,x)
    k2 = h * xdot(t+h/2 , x + k1/2)
    k3 = h * xdot(t+h/2,  x + k2/2)
    k4 = h * xdot(t+h   ,  x +  k3)

    x = x + (k1 + 2*k2 + 2*k3 + k4) / 6

    return x



def sim_vadv(mag_vm=1500, mag_vt=800, r=(1000,10000)):
    rm = np.zeros(3)

    phi = np.random.uniform(low=-np.pi,high=np.pi)
    theta = np.random.uniform(low=-np.pi/2,high=np.pi/2)
    r = np.random.uniform(low=r[0],high=r[1]) 
    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)
    rt = np.asarray([x,y,z])
    r_tm = rt-rm

    phi = np.random.uniform(low=-np.pi,high=np.pi)
    theta = np.random.uniform(low=-np.pi/2,high=np.pi/2)
    x = mag_vt*np.sin(theta)*np.cos(phi)
    y = mag_vt*np.sin(theta)*np.sin(phi)
    z = mag_vt*np.cos(theta)
    vt = np.asarray([x,y,z])
    
    los = r_tm/np.linalg.norm(r_tm)
    vm = attu.get_lead(mag_vm, vt, los)
    

    v_tm = vt - vm
    vc = get_vc(r_tm,v_tm)
    
    tf = np.linalg.norm(r_tm)/vc
   
    rm_f = rm + vm * tf
    rt_f = rt + vt * tf
    
    rtm_f = rt_f - rm_f
    #print(vc, np.linalg.norm(vm),np.linalg.norm(vt))
    miss = np.linalg.norm(rtm_f)
    return miss

class Exact_moving_average(object):
    def __init__(self):
        pass 
    def reset(self):
        self.ave = 0.
        self.cnt = 0

    def step(self,x):
        self.ave = self.ave * (self.cnt)/(self.cnt+1) + x / (self.cnt+1)
        self.cnt += 1 
    
    def test(self, n):
        self.reset()
        vals = [] 
        for i in range(n):
            v = np.random.rand()
            self.step(v)
            vals.append(v)
        print(np.mean(vals))
        print(self.ave)
