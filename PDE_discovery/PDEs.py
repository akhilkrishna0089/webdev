from modules.smooth_data import Smoother
from modules.derivatives import Deriv
from modules.library import Lib
from modules.sparse_reg import Regr
from spmodels.deepmode import DeepMoD

class pde_discovery:

    def __init__(self,method):
        self.method = method
        self.config = self.set_def_config(method)

    def set_def_config(method):
        config={}
        if method == 'PDE_FIND':
            # Need to be updated
            pass
        elif self.method == 'FFT_method':
            # Need to be updated
            pass
        elif self.method == 'Smoothed_l0bnb':
            config["Smooth"] = {"method":"KG","params":{"n":1000,"sigma":0.25,"m":3,"sigma_k": [[3,8,0.5],[1,8,0.5],[1,8,0.5]]}}
            config["Find_derivatives"] = {"method":"PI","params":{"deg":3,"width":9,"max_x_der":2}}
            config["Library"] = {"method":"PC","params":{"max_u_power":2}}
            config["Sparse_regression"] = {"method":"L0BNB","params":{"lambda_2":0.01, "max_nonzeros":5}}
        elif self.method == 'DeepMoD':
            # config["DeepMoD"] = 
            # Need to be updated
            pass
        elif self.method == 'Custom':
            pass
        return config

    
    def predict(self,data):
        u = data["u"]
        x = data["x"]
        t = data["t"]

        #  ============ Special Methods: methods that do not follow the general methodology==========:
        # Special method 1: DeepMoD
        if self.method == "DeepMoD":
            deepmod = DeepMoD(self.config["DeepMoD"])
            pdes = deepmod.predict(u,x,t)
            return {"Method":self.method,"Config":self.config,"Predicted PDE(s)":list(pdes)}

        # ============ Methods that follow the general methodology(smoothing,finding derivatives, building theta): config is defined in set_def_config ============================================================
        # Smoothing
        if "Smooth" in self.config:
            smoother = Smoother(self.config["Smooth"]["method"],self.config["Smooth"]["params"])
            u_cap = smoother.smooth(u,x,t)
        else:
            u_cap = u
        # Finding derivatives
        deriv = Deriv(self.config["Find_derivatives"]["method"],**self.config["Find_derivatives"]["params"])
        u,ut,ders  = deriv.find_derivatives(u_cap,x,t)
        # Building library
        lib = Lib(self.config["Library"]["method"],**self.config["Library"]["params"])
        X,y,descr = lib.build_library(u,ut,ders)
        # Sparse regression
        regr = Regr(self.config["Sparse_regression"]["method"],**self.config["Sparse_regression"]["params"])
        pdes = regr.fit(X,y,descr)
        return {"Method":self.method,"Config":self.config,"Predicted PDE(s)":list(pdes)}
        
      
        



        

        





