from PDEs import pdefind

path = "C:/Users/user/Downloads/Dataset generation/u,x,t/2_0.pkl"

data = pdefind(path)
data.allpdes()
#data.pdesby("FFT")