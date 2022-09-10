import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from helpers import *

class Smoother:

    def __init__(self,method,params):
        self.method = method
        self.params = params
    def smooth(self,u,x,t):
        if self.method == "KG":
            return self.smooth_KG(u,x,t,**self.params)


    # ===================== Option 1: KNN and Gaussian filter for smoothing =======================================================
    def smooth_KG(self,u,x,t,n,sigma,m,sigma_k):
                
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

        def KNN_Gaussian(u,x,t,s1,k,s2):
            '''
            Performs a combination of KNN and Gaussian filter smoothing
            '''

            u_cap = gaussian_filter(u, sigma=s1)
            t_len,x_len = u.shape

            # Reshaping u 
            u_reg,x_reg,t_reg = uxt_2D_to1D(u_cap,x,t)
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

        u_cap=u
        for i in range(n):
            u_cap=gaussian_filter(u_cap,sigma=sigma)
        for i in range(m):
            u_cap = KNN_Gaussian(u_cap,x,t,s1=sigma_k[i][0],k=sigma_k[i][1],s2=sigma_k[i][2])
        return u_cap

    # ==================== Option 2: ====================================================================================================
    # Can be updated when we find more methods for smoothing
