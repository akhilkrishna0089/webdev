# imports

class Regr:

    def __init__(self,method,params):
        self.method = method
        self.params = params
    def fit(self,X,y,descr):
        if self.method == "L0BNB":  
            return self.Regr_L0BNB(u,x,t,**self.params)
        elif self.method == "STRidge":  
            return self.Regr_STRidge(u,x,t,**self.params)

    def Regr_L0BNB(self,X,y,descr,lambda_2,max_non_zeros):
        u,ut,ders = [],[],[]
        # Need to be updated
        return u,ut,ders
    
    def Regr_STRidge(self,X,y,descr, ): # Need to add parameters of STRidge
        u,ut,ders = [],[],[]
        # Need to be updated
        return u,ut,ders
