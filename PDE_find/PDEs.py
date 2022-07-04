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


from PDE_find.Methods.smooth_l0bnb import smoothed_l0bnb
from PDE_find.Methods.FFT import PDEFFT
from PDE_find.Methods.pde_find import PDE_FIND

class pdefind:
    def __init__(self,data):
        self.ret = {"method1":[],"method2": [],"method3" : [] }
        self.data = data
    def allpdes(self):
        method1 = smoothed_l0bnb()
        result1 = method1.smooth_and_l0bnb(self.data)
        self.ret["method1"]+=(result1)

        method2 = PDEFFT()
        result2 = method2.FFT_l0bnb(self.data)
        self.ret["method2"] += (result2)

        method3 = PDE_FIND()
        result3 = method3.pde_find_algo(self.data)
        self.ret["method3"].append(result3)

        self.printall()

    def printall(self):
        print("Method 1 -> Smoothing and l0bnb: ")
        for i in range(len(self.ret["method1"])):
            print(self.ret["method1"][i])

        print("Method 2 -> FFT and l0bnb: ")
        for i in range(len(self.ret["method2"])):
            print("u_t = "+self.ret["method2"][i])

        print("Method 3 -> PDEFIND: ")
        print(self.ret["method3"][0])

    def pdesby(self,type):
        if type == "smoothed_l0bnb":
            method1 = smoothed_l0bnb()
            result = method1.smooth_and_l0bnb(self.data)
            print("Method 1 -> Smoothing and l0bnb: ")
            for i in range(len(result)):
                print(result[i])
            
        elif type == "FFT":
            method2 = PDEFFT()
            result = method2.FFT_l0bnb(self.data) 
            print("Method 2 -> FFT and l0bnb: ")
            for i in range(len(result)):
                print("u_t = "+ result[i])

        elif type == "PDE_FIND":
            method3 = PDE_FIND()
            result = method3.pde_find_algo(self.data)
            print("Method 3 -> PDEFIND: ")
            print(result)




        

        





