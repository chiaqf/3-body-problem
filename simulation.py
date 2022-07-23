# Three Body Problem
# date   : 2022-07-23
# author : qinfeng
# 
#
# credits to Gaurav Deshmukh for the mathematical model taken 
# from https://towardsdatascience.com/modelling-the-three-body-problem-in-classical-mechanics-using-python-9dc270ad7767
#

import scipy as sci
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.path import Path
from matplotlib import animation
plt.style.use('seaborn')

G=6.67408e-11 #N-m2/kg2
#Reference quantities
m_nd=1.989e+30 #kg #mass of the sun
r_nd=5.326e+12 #m #distance between stars in Alpha Centauri
v_nd=30000 #m/s #relative velocity of earth around the sun
t_nd=79.91*365*24*3600*0.51 #s #orbital period of Alpha Centauri
#Net constants
K1=G*t_nd*m_nd/(r_nd**2*v_nd)
K2=v_nd*t_nd/r_nd

# orbit 1
#Define masses
#m1=0.5 #Alpha Centauri A
#m2=0.5 #Alpha Centauri B
#m3=0.5 #Third Star
##Define initial position vectors
#r1=[0.3,0,0] #m
#r2=[0,0.3,0] #m
#r3=[0,0,0.3] #m
##Define initial velocities
#v1=[0,0,0.2] #m/s
#v2=[0.2,0,0] #m/s
#v3=[0,0.2,0]

# orbit 2
#Define masses
#m1=1 #Alpha Centauri A
#m2=1 #Alpha Centauri B
#m3=1 #Third Star
##Define initial position vectors
#r1=[0,0,0] #m
#r2=[0.1,0.6,0] #m
#r3=[-0.2,-0.3,0] #m
##Define initial velocities
#v1=[0,0,0.1] #m/s
#v2=[0,0,0.2] #m/s
#v3=[0,0,0.3]

# orbit 3
#Define masses
m1=500 #Alpha Centauri A
m2=1 #Alpha Centauri B
m3=1 #Third Star
##Define initial position vectors
r1=[0,0,0] #m
r2=[2,0,0] #m
r3=[4,0,0] #m
##Define initial velocities
v1=[0,0,0.2] #m/s
v2=[0,1.5,0] #m/s
v3=[0,2,0]

# orbit 4, random
#Define masses
#import random
#m1=1 #Alpha Centauri A
#m2=1 #Alpha Centauri B
#m3=1 #Third Star
####Define initial position vectors
#r1=np.random.uniform(-0.5,0.5,[3]) #m
#r2=np.random.uniform(-0.5,0.5,[3])
#r3=np.random.uniform(-0.5,0.5,[3])
####Define initial velocities
#v1=np.random.uniform(-0.5,0.5,[3])
#v2=np.random.uniform(-0.5,0.5,[3])
#v3=np.random.uniform(-0.5,0.5,[3])
#

r1=sci.array(r1,dtype="float64")
r2=sci.array(r2,dtype="float64")
r3=sci.array(r3,dtype="float64")

v1=sci.array(v1,dtype="float64")
v2=sci.array(v2,dtype="float64")
v3=sci.array(v3,dtype="float64")

#Update COM formula
r_com=(m1*r1+m2*r2+m3*r3)/(m1+m2+m3)
#Update velocity of COM formula
v_com=(m1*v1+m2*v2+m3*v3)/(m1+m2+m3)

def ThreeBodyEquations(w,t,G,m1,m2,m3):
    r1=w[:3]
    r2=w[3:6]
    r3=w[6:9]
    v1=w[9:12]
    v2=w[12:15]
    v3=w[15:18]
    r12=sci.linalg.norm(r2-r1)
    r13=sci.linalg.norm(r3-r1)
    r23=sci.linalg.norm(r3-r2)

    dv1bydt=K1*m2*(r2-r1)/r12**3+K1*m3*(r3-r1)/r13**3
    dv2bydt=K1*m1*(r1-r2)/r12**3+K1*m3*(r3-r2)/r23**3
    dv3bydt=K1*m1*(r1-r3)/r13**3+K1*m2*(r2-r3)/r23**3
    dr1bydt=K2*v1
    dr2bydt=K2*v2
    dr3bydt=K2*v3
    r12_derivs=np.concatenate((dr1bydt,dr2bydt))
    r_derivs=np.concatenate((r12_derivs,dr3bydt))
    v12_derivs=np.concatenate((dv1bydt,dv2bydt))
    v_derivs=np.concatenate((v12_derivs,dv3bydt))
    derivs=np.concatenate((r_derivs,v_derivs))
    return derivs

init_params=sci.array([r1,r2,r3,v1,v2,v3]) #Initial parameters
init_params=init_params.flatten() #Flatten to make 1D array
time_span=sci.linspace(0,20,5000) #20 orbital periods and 500 points

import scipy.integrate
three_body_sol=sci.integrate.odeint(ThreeBodyEquations,init_params,time_span,args=(G,m1,m2,m3))

r1_sol=three_body_sol[:,:3]
r2_sol=three_body_sol[:,3:6]
r3_sol=three_body_sol[:,6:9]

fig=plt.figure(figsize=(10,10))
ax=fig.add_subplot(projection="3d")
ax.set_xlabel("x-coordinate",fontsize=14)
ax.set_ylabel("y-coordinate",fontsize=14)
ax.set_zlabel("z-coordinate",fontsize=14)
ax.set_title("Visualization of Orbits of Stars in a Three-Body System\n",fontsize=14)
ax.legend(loc="upper left",fontsize=14)


#update plot xlim ylim dynamically
def update_lines(num, paths, lines):
    max_xdata = 0
    min_xdata = 0
    max_ydata = 0
    min_ydata = 0
    max_zdata = 0
    min_zdata = 0

    for line, path in zip(lines, paths):
        data = path[:num, :2].T
        line.set_data(data)
        line.set_3d_properties(path[:num, 2])
        line.set_markevery([-1])
        line.set_markersize(20)

        try:
            x_data = data[0]
            y_data = data[1]
            z_data = path[:num,2]
            if (x_data.max() > max_xdata) : max_xdata = x_data.max()
            if (x_data.min() < min_xdata) : min_xdata = x_data.min()

            if (y_data.max() > max_ydata) : max_ydata = y_data.max()
            if (y_data.min() < min_ydata) : min_ydata = y_data.min()

            if (z_data.max() > max_zdata) : max_zdata = z_data.max()
            if (z_data.min() < min_zdata) : min_zdata = z_data.min()

        except:
            continue

    ax.set_xlim(min_xdata, max_xdata)
    ax.set_ylim(min_ydata, max_ydata)
    ax.set_zlim(min_zdata, max_zdata)


    return lines

num_steps = 100000

paths = np.array([r1_sol, r2_sol, r3_sol])
lines = [ax.plot([], [], [], 'o:')[0] for _ in paths]

# Creating the Animation object
ani = animation.FuncAnimation(
    fig, update_lines, num_steps, fargs=(paths, lines), interval=1, blit=False, repeat=False)

plt.show()

