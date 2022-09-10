def uxt_2D_to1D(u,x,t):
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