# imports
import numpy as np
import itertools
import operator

class Lib:

    def __init__(self,method,params):
        self.method = method
        self.params = params
    def build_library(self,u,ut,ders):
        if self.method == "PC":  # PC: Polynomial combination: combining ux,uxx,uxxx... with u,u^2,u^3.......
            return self.Lib_PC(u,ut,ders,**self.params)

    def Lib_PC(self,u,ut,ders,deg,width,max_u_power):
        
        # Need to be updated
        # build theta function
        num_points = ut.shape[0]
        allders = [np.ones((num_points ,1))]
        for der in ders:
            allders.append(ders[der])
        X_ders = np.hstack(allders)
        X_data = np.hstack([np.reshape(u, (num_points ,1), order='F')])
        derivatives_description = ['']
        for i in range(len(ders.keys)):
            derivatives_description.append('u_{' + "x"*(i+1)+ "}")
        X, descr = self.build_Theta(X_data, X_ders, derivatives_description, P = max_u_power, data_description = ['u'])  # Note that here we can add other terms like |u| if we want
        y = ut.reshape(ut.shape[0])
        return X,y,descr
    def build_Theta(self,data, derivatives, derivatives_description, P, data_description = None):
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

         


