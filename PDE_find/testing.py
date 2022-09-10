import pickle 
import numpy as np


def Deriv_PI(u,x,t,deg,width,max_x_der):
        ders = {}
        # Need to be updated
        if width%2==0:
            return "Error: Width to be used in polynomial interpolation should be odd so that we split the left and right points evenly"
        t_leni,x_leni = u.shape
        t_len=t_leni-(width)+1 # t_len and x_len decreases because we neglect the terms at the end
        x_len=x_leni-(width)+1 

        ut = np.zeros((t_len,x_len))
        for i in range(1,max_x_der+1):
            ders["u"+"x"*i] = np.zeros((t_len,x_len))
        
        w=width//2  # number of points taken on each side
        for i in range(x_len):
            ut[:,i] = interpolate(u[:,i+w], t, diff=max_x_der,deg=deg,width=width)[:,0]
        for i in range(t_len):
            der=interpolate(u[i+w,:], x, diff= max_x_der,deg=deg,width=width)
            j = 0
            for deriv in ders:
                ders[deriv][i,:] = der[:,j]
                j+= 1
        u=u[w:t_leni-w,w:x_leni-w]
        ut = np.reshape(ut, ((t_len)*(x_len),1), order='F')

        for deriv in ders:
            ders[deriv] = np.reshape(ders[deriv],((t_len)*(x_len),1), order='F')
        print("Derivative calculation done")
        return u,ut,ders

def interpolate(u,x,deg,width,diff):
    u = u.flatten()
    x = x.flatten()

    n = len(x)
    du = np.zeros((n - width+1,diff))
    w=width//2
    
    for j in range(w, n-w):
        points = np.arange(j - w, j + w+1)
        # Polynomial interpolation: Least-squares fit
        poly = np.polynomial.polynomial.Polynomial.fit(x[points],u[points],deg)
        # Taking derivatives
        for d in range(1,diff+1):
            du[j-w, d-1] = poly.deriv(m=d)(x[j])
    return du




