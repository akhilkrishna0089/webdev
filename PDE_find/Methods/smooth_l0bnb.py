import numpy as np
import itertools
import operator
from l0bnb import fit_path
import pickle
from sklearn.model_selection import train_test_split
from scipy.ndimage import gaussian_filter
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

class smoothed_l0bnb:
    def library(self,u,ut,ux,uxx):
    
        #Builds the library of terms
    
        num_points = ut.shape[0]
        X_ders = np.hstack([np.ones((num_points ,1)),ux,uxx])
        X_data = np.hstack([np.reshape(u, (num_points ,1), order='F')])
        derivatives_description = ['','u_{x}','u_{xx}', 'u_{xxx}']
        X, descr = self.build_Theta(X_data, X_ders, derivatives_description, P = 2, data_description = ['u'])  # Note that here we can add other terms like |u| if we want
        y = ut.reshape(ut.shape[0])
        return X,y,descr

    def build_Theta(self,data, derivatives, derivatives_description, P, data_description = None):
   
    
        #DIRECTLY TAKEN FROM PDE-FIND

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
    def derivatives(self,u,x,t,deg=3,width=9):


        #Finds the derivatives of u numerically using polynomial interpolation

        #Inputs:
        #u=u matrix
        #x=x values
        #t=t values
        #deg=degree of the polynomial in polynomial interpolation
        #width=no of points that are considered for interpolation (should be odd)

        #Outputs:
        #u = u matrix without the boundary points
        #ut = time derivative of u
        #ux = space derivative of u
        #uxx = 2nd space derivative of u

    
        if width%2==0:
            return "Error: Width to be used in polynomial interpolation should be odd so that we split the left and right points evenly"

        t_leni,x_leni = u.shape
        t_len=t_leni-(width)+1 # t_len and x_len decreases because we neglect the terms at the end
        x_len=x_leni-(width)+1 

        ut = np.zeros((t_len,x_len))
        ux = np.zeros((t_len,x_len))
        uxx = np.zeros((t_len,x_len))
        
        w=width//2  # number of points taken on each side
        for i in range(x_len):
            ut[:,i] = self.interpolate(u[:,i+w], t, diff=2,deg=deg,width=width)[:,0]
        for i in range(t_len):
            der=self.interpolate(u[i+w,:], x, diff=2,deg=deg,width=width)
            ux[i,:] = der[:,0]
            uxx[i,:] = der[:,1]
        u=u[w:t_leni-w,w:x_leni-w]

        ut = np.reshape(ut, ((t_len)*(x_len),1), order='F')
        ux = np.reshape(ux,  ((t_len)*(x_len),1), order='F')
        uxx = np.reshape(uxx,  ((t_len)*(x_len),1), order='F')
        print("Derivative calculation done")
        return u,ut,ux,uxx


    def interpolate(self,u,x,deg,width,diff):    
        
        #u = values of a function
        #x = x-coordinates where values are known
        #deg=degree of the polynomial in polynomial interpolation
        #width=no of points that are considered for interpolation (is odd)
        #diff=max order of derivative to be calculated
        
        
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
    def load(self,path):
    
        #This function loads the data

        #Inputs:
        #path= path to the pickle file that stores the data in the form of a python dictionary

        #Outputs:
        #u,x,t,dx,dt

   
        file_to_read = open(path, "rb")
        loaded_dictionary = pickle.load(file_to_read)
        u = loaded_dictionary["u"]
        x = loaded_dictionary["x"]
        t = loaded_dictionary["t"]
        dx = x[2]-x[1]
        dt = t[2]-t[1]
        print("Data Loaded")

        return u,x,t
    def print_pde(self,w, rhs_description, ut = 'u_t'):
        pde = ut + ' = '
        first = True
        for i in range(len(w)):
            if w[i] != 0:
                if not first:
                    pde = pde + ' + '
                pde = pde + "(%05f)" % (w[i].real) + rhs_description[i] + "\n   "
                first = False
        return(pde)

    def return_pde(self,w, rhs_description,ret, ut = 'u_t'):
        pde = ut + ' = '
        first = True
        for i in range(len(w)):
            if w[i] != 0:
                if not first:
                    pde = pde + ' + '
                pde = pde + "(%05f)" % (w[i].real) + rhs_description[i] 
                first = False
        ret.append(pde)
    
    def smooth(self,u,x,t,n=1000,sigma=0.25,m=3,sigma_k = [[3,8,0.5],[1,8,0.5],[1,8,0.5]]):
        """
        This function smooths the u matrix while keeping x and t the same.

        Inputs:
        u: The u matrix that is to be smoothed
        x: x values
        t: t values
        n: number of times initial gaussian filter is to be applied
        sigma: sigma value to be used in the initial gaussian filter
        m: number of times we need to perform KNN_Gaussian
        sigma_k: sigma_k[i][0] refers to the sigma value in the first gaussian filter of KNN_Gaussian for ith execution of KNN_Gaussian
                sigma_k[i][1] refers to the num of neighbours in the KNN regression of KNN_Gaussian for ith execution of KNN_Gaussian
                sigma_k[i][2] refers to the sigma value in the second gaussian filter of KNN_Gaussian for ith execution of KNN_Gaussian

        Output: 
        The smoothed u matrix
        """
        u_cap=u
        for i in range(n):
            u_cap=gaussian_filter(u_cap,sigma=sigma)
        for i in range(m):
            u_cap = self.KNN_Gaussian(u_cap,x,t,s1=sigma_k[i][0],k=sigma_k[i][1],s2=sigma_k[i][2])
        print("Data smoothed")
        return u_cap

    def KNN_Gaussian(self,u,x,t,s1,k,s2):
        '''
        Performs a combination of KNN and Gaussian filter smoothing
        '''

        u_cap = gaussian_filter(u, sigma=s1)
        t_len,x_len = u.shape

        # Reshaping u 
        u_reg,x_reg,t_reg = self.uxt_2D_to1D(u_cap,x,t)
        X = np.vstack([x_reg,t_reg]).T
        y = u_reg

        # Feature Scaling
        sc = StandardScaler()
        X_scaled = sc.fit_transform(X)

        # Applying KNN
        regressor = KNeighborsRegressor(n_neighbors=k)
        regressor.fit(X_scaled, y)
        y_pred = regressor.predict(X_scaled)
        u_cap = y_pred.reshape(t_len,x_len)

        u_cap = gaussian_filter(u_cap,sigma=s2)

        return u_cap

    def uxt_2D_to1D(self,u,x,t):
        '''
        Function for reshaping u,x and t so that we can perform KNN regression on it
        '''
        t_reg=[]
        x_reg=[]
        for i in range(len(t)):
            for j in range(len(x)):
                t_reg.append(t[i])
        for i in range(len(t)):
            for j in range(len(x)):
                x_reg.append(x[j])
        u_reg = u.reshape((u.size, 1))

        return u_reg,x_reg,t_reg

    def smooth_and_l0bnb(self,path):
        u,x,t = self.load(path)
        u_approx = self.smooth(u,x,t)
        u,ut,ux,uxx = self.derivatives(u_approx, x, t)
        X,y,descr = self.library(u,ut,ux,uxx)
        sols = fit_path(X,y, lambda_2 = 0.01, max_nonzeros = 5, intercept=False)
        ret = []
        for i in range(len(sols)):
            w= sols[i]["B"]
            self.return_pde(w, descr,ret)
        print("Method1 completed")
        return ret

#path = "C:/Users/user/Downloads/Dataset generation/u,x,t/1_0.pkl"
#data = smoothed_l0bnb()
#ret = data.smooth_and_l0bnb(path)
#print(type(ret))
