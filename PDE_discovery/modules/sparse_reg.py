# imports
from l0bnb import fit_path

class Regr:

    def __init__(self,method,params):
        self.method = method
        self.params = params
    def fit(self,X,y,descr):
        if self.method == "L0BNB":  
            return self.Regr_L0BNB(u,x,t,**self.params)
        elif self.method == "STRidge":  
            return self.Regr_STRidge(u,x,t,**self.params)
    
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

    def Regr_L0BNB(self,X,y,descr,lambda_2,max_non_zeros):
        sols = fit_path(X,y, lambda_2 , max_non_zeros , intercept=False)
        
        ret = []
        for i in range(len(sols)):
            w= sols[i]["B"]
            self.return_pde(w, descr,ret)
        return ret
    
    def Regr_STRidge(self,X,y,descr, ): # Need to add parameters of STRidge
        u,ut,ders = [],[],[]
        # Need to be updated
        return u,ut,ders
