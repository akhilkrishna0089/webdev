import PDE_find.PDEs
import PDE_find


path = "C:/Users/user/Downloads/Dataset generation/u,x,t/2_0.pkl"
data = PDE_find.PDEs.pdefind(path)
data.allpdes()
#data.pdesby("FFT")