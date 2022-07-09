# imports
import numpy as np

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

         


