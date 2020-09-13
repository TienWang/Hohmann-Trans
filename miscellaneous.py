import numpy as np
from RK78 import *
import matplotlib.pyplot as plt
import astropy.units as u
from astropy import constants as const

D_mars = 1.52*u.AU
M_mars = 6.4219e23*u.kg
M_earth = const.M_earth
M_jup = const.M_jup
D_jup = 5.2*u.AU
M_sat = 5.6846e26*u.kg
D_sat = 9.54*u.AU

Jupiter = np.array([-5.2,0,0])
Saturn = np.array([-9.54,0,0])
Mars = np.array([-1.52,0,0])
Earth = np.array([1,0,0])
Sun = np.array([0,0,0])

GM_sun = 1
GM_jupiter = (const.GM_jup/const.GM_sun).value
GM_saturn = (const.GM_jup/const.GM_sun * (M_sat/M_jup)).value
GM_earth = (const.GM_earth/const.GM_sun).value
GM_mars = (const.GM_earth/const.GM_sun * (M_mars/M_earth)).value

r_p = np.array([Sun,Jupiter,Saturn,Earth,Mars])
GM = np.array([GM_sun,GM_jupiter,GM_saturn,GM_earth,GM_mars])

def model_p(t,xx):
    x,y,z,vx,vy,vz = xx
    r_m = np.array([x,y,z])
    ax,ay,az = [0,0,0]
    
    for i in range(0,5):
        r = r_m-r_p[i]
        rn = np.sqrt(np.sum(np.square(r_m-r_p[i])))
        ax -= GM[i]*r[0]*rn**(-3)
        ay -= GM[i]*r[1]*rn**(-3)
        az -= GM[i]*r[2]*rn**(-3)
    return np.array([vx, vy, vz, ax, ay, az])

def model(t,xx):
    x,y,z,vx,vy,vz = xx
    r = (x**2+y**2+z**2)**(0.5)
    ax = -x*r**(-3)
    ay = -y*r**(-3)
    az = -z*r**(-3)
    return np.array([vx, vy, vz, ax, ay, az])

def model_r3body(t,xx):
    mu = 1e-6 #Earth
    x,y,z,vx,vy,vz = xx
    r1 = ((x+mu)**2 +y**2+z**2)**-1.5
    r2 = ((x+mu-1)**2+y**2+z**2)**-1.5
    omegax = x-(1-mu)*(x+mu)*r1 -mu*(x+mu-1)*r2
    omegay = y*(1-(1-mu)*r1-mu*r2)
    omegaz = -z*((1-mu)*r1+mu*r2)
    return np.array([vx, vy, vz, 2*vy+omegax, -2*vx+omegay, omegaz])

def closest_point(result):
    index = result[0][:]<0
    coord = result[:,index]
    coord[1,:]=np.square(coord[1,:])
    index = coord[1,:].argmin()
    #print(coord)
    coord = coord[:,index]
    coord[1] = coord[1]**(0.5)
    return coord

def orbit(initial, period=4000,h=np.pi/1000,err=1e-10,**kwarg):
    
    results = RKF78(0, initial, h, err, period,**kwarg)
    time = np.array(results[0])
    results = np.array(results[1]).T
    print(results.shape[-1])
    return time, results

def hohmann(r1,r2,GM):
    return np.sqrt(GM/r1)*(np.sqrt(2*r2/(r1+r2))-1), np.sqrt(GM/r2)-np.sqrt((2*GM/r2)-2*GM/(r1+r2)), np.pi*np.sqrt(((r1+r2)**3)/(8*GM))

def plot_circular_orbit(r,color,label,x0=0,y0=0):
    import matplotlib.pyplot as plt
    theta = np.arange(0, 2*np.pi, 0.01)
    x = r * np.cos(theta)+x0
    y = r * np.sin(theta)+y0
    plt.plot(x,y,c=color,linewidth=0.5,label=label)
    pass

def get_T(R,GM):
    return 2*np.pi*R/np.sqrt(GM/R)

def get_v(R):
    return np.sqrt(const.GM_sun/R).to(u.km/u.s)
