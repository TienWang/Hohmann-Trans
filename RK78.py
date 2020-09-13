import numpy as np

a = np.array([0,2/27,1/9,1/6,5/12,1/2,5/6,1/6,2/3,1/3,1,0,1])
c_bar = np.array([0,0,0,0,0,34/105,9/35,9/35,9/280,9/280,0,41/840,41/840])
# 13 columns of 'b' to satisfy the 'k_i' size
b = np.array(
    [[0,0,0,0,0,0,0,0,0,0,0,0,0],
        [2/27,0,0,0,0,0,0,0,0,0,0,0,0],
        [1/36,1/12,0,0,0,0,0,0,0,0,0,0,0],
        [1/24,0,1/8,0,0,0,0,0,0,0,0,0,0],
        [5/12,0,-25/16,25/16,0,0,0,0,0,0,0,0,0],
        [1/20,0,0,1/4,1/5,0,0,0,0,0,0,0,0],
        [-25/108,0,0,125/108,-65/27,125/54,0,0,0,0,0,0,0],
        [31/300,0,0,0,61/225,-2/9,13/900,0,0,0,0,0,0],
        [2,0,0,-53/6,704/45,-107/9,67/90,3,0,0,0,0,0],
        [-91/108,0,0,23/108,-976/135,311/54,-19/60,17/6,-1/12,0,0,0,0],
        [2383/4100,0,0,-341/164,4496/1025,-301/82,2133/4100,45/82,45/164,18/41,0,0,0],
        [3/205,0,0,0,0,-6/41,-3/205,-3/41,3/41,6/41,0,0,0],
        [-1777/4100,0,0,-341/164,4496/1025,-289/82,2193/4100,51/82,33/164,12/41,0,1,0]])
""" END """

def RKF78(t,x,h,delta,t_final,**kwargs):
    time = [0]; result = [x]
    while t < t_final:
        h, x, expand = calc(t,x,h*2,delta,**kwargs)
        t += h
        time.append(t)
        result.append(x)
        #if expand:
        #    h *= 2
    print('Done!')
    print('Function returns the time and results sequences in type of python-list.')
    print('Remember to transfer them to ndarray and transpose the results!')
    return time, result


""" Calculate and return k_i and step lenth satisfying given accuracy """

def calc(t,x,h,delta,model):
    delta2 = delta**2
    x_err = 2*delta2
    while(x_err > delta2):
        h /= 2
        k = np.zeros((13,6))
        for i in range(13):
            tt = t+a[i]*h
            xx = x+multi_row(b[i]*k.T)*h
            k[i] = model(tt,xx)

            
        tmp = (41/810)*h*(k[0]+k[10]-k[11]-k[12])
        x_err = np.sum((tmp)**2)/np.sum(x**2) # relative accuracy
    if x_err < 1e-5*delta2:
        return h, x+h*multi_row(c_bar*k.T), True
    else:
        return h, x+h*multi_row(c_bar*k.T), False

def multi_row(x):
    size = x.shape[0]
    result = np.zeros(size)
    for i in range(size):
        result[i] = np.sum(x[i])
    return result


