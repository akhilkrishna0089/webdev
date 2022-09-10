import numpy as np
from l0bnb import fit_path
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter
from cv2 import bilateralFilter
from pycav.pde import CN_diffusion_equation
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from numpy import linalg as LA
import scipy.sparse as sparse
from scipy.sparse import csc_matrix
from scipy.sparse import dia_matrix
import itertools
import operator
import pickle

import pandas as pd

from scipy.fft import fft

import linalg as LA
import scipy.sparse as sparse
from scipy.sparse import csc_matrix
from scipy.sparse import dia_matrix
import itertools
import operator


class PDEFFT:
    def __init__(self):
        self.der = None

    def build_Theta(self,data, derivatives, derivatives_description, P, data_description = None):
    
    #builds a matrix with columns representing polynoimials up to degree P of all variables

    #This is used when we subsample and take all the derivatives point by point or if there is an 
    #extra input (Q in the paper) to put in.

    #input:
        #data: column 0 is U, and columns 1:end are Q
        #derivatives: a bunch of derivatives of U and maybe Q, should start with a column of ones
        #derivatives_description: description of what derivatives have been passed in
        #P: max power of polynomial function of U to be included in Theta

    #returns:
        #Theta = Theta(U,Q)
        #descr = description of what all the columns in Theta are

    
     n,d = data.shape
     m, d2 = derivatives.shape
     if n != m: raise Exception('dimension error')
     if data_description is not None: 
         if len(data_description) != d: raise Exception('data descrption error')
    
    # Create a list of all polynomials in d variables up to degree P
     rhs_functions = {}
     f = lambda x, y : np.prod(np.power(list(x), list(y)))
     powers = []            
     for p in range(1,P+1):
             size = d + p - 1
             for indices in itertools.combinations(range(size), d-1):
                 starts = [0] + [index+1 for index in indices]
                 stops = indices + (size,)
                 powers.append(tuple(map(operator.sub, stops, starts)))
     for power in powers: rhs_functions[power] = [lambda x, y = power: f(x,y), power]

    # First column of Theta is just ones.
     Theta = np.ones((n,1))
     descr = ['']
    
    # Add the derivaitves onto Theta
     for D in range(1,derivatives.shape[1]):
         Theta = np.hstack([Theta, derivatives[:,D].reshape(n,1)])
         descr.append(derivatives_description[D])
        
    # Add on derivatives times polynomials
     for D in range(derivatives.shape[1]):
         for k in rhs_functions.keys():
             func = rhs_functions[k][0]
             new_column = np.zeros((n,1))
             for i in range(n):
                 new_column[i] = func(data[i,:])*derivatives[i,D]
             Theta = np.hstack([Theta, new_column])
             if data_description is None: descr.append(str(rhs_functions[k][1]) + derivatives_description[D])
             else:
                 function_description = ''
                 for j in range(d):
                     if rhs_functions[k][1][j] != 0:
                         if rhs_functions[k][1][j] == 1:
                             function_description = function_description + data_description[j]
                         else:
                             function_description = function_description + data_description[j] + '^' + str(rhs_functions[k][1][j])
                 descr.append(function_description + derivatives_description[D])

     return Theta, descr
    def load(self,path):
    
        #This function loads the data

        #Inputs:
        #path= path to the pickle file that stores the data in the form of a python dictionary

        #Outputs:
        #u,x,t,dx,dt

   
        file_to_read = open(path, "rb")
        loaded_dictionary = pickle.load(file_to_read)
        u_cap = loaded_dictionary["u"]
        x = loaded_dictionary["x"]
        t = loaded_dictionary["t"]
        dx = x[2]-x[1]
        dt = t[2]-t[1]
        print("Data Loaded")
        return u_cap,x,t

    def FFT_l0bnb(self,data):
        my_csv = self.derivatives_calculator(data)
        u = my_csv["u"].values
        ut = my_csv["u_t"].values
        ux = my_csv["u_{x}"].values
        uxx = my_csv["u_{xx}"].values
        uxxx = my_csv["u_{xxx}"].values

        derivatives_description = ['','u_{x}','u_{xx}', 'u_{xxx}']

        X_ders = np.vstack([np.ones(u.shape),ux,uxx,uxxx]).T
        X_data = np.hstack([u.reshape(u.shape[0],1)])
        
        X, descr = self.build_Theta(X_data, X_ders, derivatives_description, P = 2, data_description = ['u'])

        # Applying FFT
        Y = X
        for i in range(X.shape[1]):
            Y[:,i]=fft(X[:,i])


        ut=fft(ut)

        Z = Y[:100,:]
        ut = ut[:100]

        ut_temp=np.array([])
        ut_temp=np.append(ut_temp,ut.real)
        ut_temp=np.append(ut_temp,ut.imag)


        Z_temp=np.vstack((Z.real,Z.imag))

        # Applying l0bnb

        sols = fit_path(Z_temp,ut_temp,lambda_2=0.01, max_nonzeros = 5, intercept=True,gap_tol=0.0001)
        sols

        sol_set=[]
        for i in range(len(sols)):
            b=list(np.around(sols[i]['B'],5))
            a=""
            for i in range(len(b)):
                if((b[i])!=0):
                    a=a+"+("+str(b[i])+')*'+descr[i]
            sol_set.append(a)
        print("Method 2 completed")
        return (sol_set)

    def Nohan_PolyDiff(self,u, x, deg, diff, width):
    
        """
        u = values of some function
        x = x-coordinates where values are known
        deg = degree of polynomial to use
        diff = maximum order derivative we want
        width = width of window to fit to polynomial
        """
    
        u = u.flatten()
        x = x.flatten()

        n = len(x)
        du = np.zeros((n - width+1,diff))

        if (width%2==0):
            w=width//2
            # Take the derivatives in the center of the domain
            for j in range(w, n-w+1):

                points = np.arange(j - w, j + w)
            
                # Fit to a Chebyshev polynomial
                # this is the same as any polynomial since we're on a fixed grid but it's better conditioned :)
                poly = np.polynomial.polynomial.Polynomial.fit(x[points],u[points],deg)
                # print(poly)
                # Take derivatives
                for d in range(1,diff+1):
                    du[j-w, d-1] = poly.deriv(m=d)(x[j])

        else:
            w=width//2
            # Take the derivatives in the center of the domain
            for j in range(w, n-w):

                points = np.arange(j - w, j + w+1)

                # Fit to a Chebyshev polynomial
                # this is the same as any polynomial since we're on a fixed grid but it's better conditioned :)
                poly = np.polynomial.polynomial.Polynomial.fit(x[points],u[points],deg)
                # print(poly)
                # Take derivatives
                for d in range(1,diff+1):
                    du[j-w, d-1] = poly.deriv(m=d)(x[j])

        return du


    def FindDerivatives(self,u,x,t,width=9,deg=3):
        t_len,x_len = u.shape
        # FINDING DERIVATIVE --------------------------------------
        t_len_new=t_len-(width)+1
        x_len_new=x_len-(width)+1
        ut = np.zeros((t_len_new,x_len_new))
        ux = np.zeros((t_len_new,x_len_new))
        uxx = np.zeros((t_len_new,x_len_new))
        uxxx = np.zeros((t_len_new,x_len_new))
        if (width%2==0):
            w=width//2
            for i in range(x_len_new):
                ut[:,i] = self.Nohan_PolyDiff(u[:,i+w], t, diff=3,deg=deg,width=width)[:,0]
            for i in range(t_len_new):
                der=self.Nohan_PolyDiff(u[i+w,:], x, diff=3,deg=deg,width=width)
                ux[i,:] = der[:,0]
                uxx[i,:] = der[:,1]
                uxxx[i,:] = der[:,2]
            u=u[w:t_len-w+1,w:x_len-w+1]
        else:
            w=width//2
            for i in range(x_len_new):
                ut[:,i] = self.Nohan_PolyDiff(u[:,i+w], t, diff=3,deg=deg,width=width)[:,0]
            for i in range(t_len_new):
                der=self.Nohan_PolyDiff(u[i+w,:], x, diff=3,deg=deg,width=width)
                ux[i,:] = der[:,0]
                uxx[i,:] = der[:,1]
                uxxx[i,:] = der[:,2]
            u=u[w:t_len-w,w:x_len-w]
        u = np.reshape(u, ((t_len_new)*(x_len_new),1), order='F')
        ut = np.reshape(ut, ((t_len_new)*(x_len_new),1), order='F')
        ux = np.reshape(ux, ((t_len_new)*(x_len_new),1), order='F')
        uxx = np.reshape(uxx, ((t_len_new)*(x_len_new),1), order='F')
        uxxx = np.reshape(uxxx, ((t_len_new)*(x_len_new),1), order='F')
        print("derivatives calculated")
        return u,ut,ux,uxx,uxxx,x_len_new,t_len_new

    def derivatives_calculator(self,data):
        file_to_read = open(data, "rb")
        loaded_dictionary = pickle.load(file_to_read)
        u = loaded_dictionary["u"]
        x = loaded_dictionary["x"]
        t = loaded_dictionary["t"]
        dx = x[2]-x[1]
        dt = t[2]-t[1]

        u_cap=u

        u,ut,ux,uxx,uxxx,x_len_new,t_len_new = self.FindDerivatives(u_cap,x,t)
        X_ders = np.hstack([np.ones(((t_len_new)*(x_len_new),1)),ux,uxx,uxxx])
        X_data = np.hstack([np.reshape(u, ((t_len_new)*(x_len_new),1), order='F')])
        derivatives_description = ['','u_{x}','u_{xx}', 'u_{xxx}']
        X, descr = self.build_Theta(X_data, X_ders, derivatives_description, P = 2, data_description = ['u'])  # Note that here we can add other terms like |u| if we want

        descr[0] = "constant"
        import pandas as pd
        my_df  = pd.DataFrame(X, columns =descr)
        my_df["u_t"]=ut
        return my_df



