from PDEs import pde_discovery
import pickle

path = "C:/Users/user/Downloads/Dataset generation/u,x,t/2_0.pkl"
file_to_read = open(path, "rb")
data = pickle.load(file_to_read)

# If user directly wants to apply some end-to-end method, like Smoothed_L0BNB:
pd = pde_discovery(method='Smoothed_l0bnb')
pd.predict(data)


# If user wants to apply Smoothed_l0bnb but need to change some of its hyperparameters, eg: sigma in KNN-Gaussian smoothing and max_non_zeros in L0BNB 
pd = pde_discovery(method='Smoothed_l0bnb')
pd.config["Smooth"]["params"]["sigma"] = 0.5
pd.config["max_non_zeros"]["params"]["max_non_zeros"] = 10
pd.predict(data)


# If user wants to develop a custom model by combining different modules:
pd = pde_discovery(method='Custom')
pd.config["Smooth"] = {"method":"KG","params":{"n":100,"sigma":0.25,"m":3,"sigma_k": [[3,8,0.5],[1,8,0.5],[1,8,0.5]]}}
pd.config["Find_derivatives"] = {"method":"PI","params":{"deg":3,"width":9,"max_x_der":2}}
pd.config["Library"] = {"method":"PC","params":{"max_u_power":2}}
pd.config["Sparse_regression"] = {"method":"L0BNB","params":{"lambda_2":0.01, "max_nonzeros":5}}
pd.predict(data)