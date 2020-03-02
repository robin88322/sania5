#from main import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_results(DATA_FOLDER, eta_def, func_type):
    n = 4
    m = 7


    mask = np.array(pd.read_csv(DATA_FOLDER+'mask.csv',delimiter=';',header=None))
    alpha_hat_raw = np.array(pd.read_csv(DATA_FOLDER+'alpha_hat.csv',delimiter=';',header=None))
    It_hat_raw = np.array(pd.read_csv(DATA_FOLDER+'It_hat.csv',delimiter=';',header=None))
    Id_hat_raw = np.array(pd.read_csv(DATA_FOLDER+'Id_hat.csv',delimiter=';',header=None))
    Ip_hat_raw = np.array(pd.read_csv(DATA_FOLDER+'Ip_hat.csv',delimiter=';',header=None))

    def process_data(array):
        data = array.copy()
        data[data =='-']=-1
        data = data.astype(np.float64)
        return data

    alpha_hat = process_data(alpha_hat_raw)
    It_hat = process_data(It_hat_raw)
    Id_hat = process_data(Id_hat_raw)
    Ip_hat = process_data(Ip_hat_raw)

    n,m = alpha_hat.shape

    if func_type == 'Наш варіант':
        def get_alpha(i,j):
            return 1 - np.log(1+alpha_hat[i,j]* Ip_hat[i,j])

        def get_beta(i,j):
            return np.exp((Id_hat[i,j]+Ip_hat[i,j])/2) * alpha_hat[i,j]

        def get_gamma(i,j):
            return 2.5*(1+alpha_hat[i,j])*(1+Id_hat[i,j])**2

        def get_eta(i,j):
            """
            int_inf_ind - integral information index
            alpha  - one of coefs that show change dynamic of int_inf_ind
            """
            return 1 - np.log2(1+alpha_hat[i,j]*get_I(i,j)*(40+gamma[i,j]))

        def get_Ip(i,j):
            res = 0.00001*Ip_hat[i,j]*(1+alpha[i,j])*t**2*(1+(1+alpha[i,j]**2)/(Ip_hat[i,j]+Id_hat[i,j]))+.4
            if res < 1:
                return res
            return 1

        def get_Id(i,j):
            res = 0.4+0.01*Id_hat[i,j] * (1+alpha[i,j]+beta[i,j]+gamma[i,j])*np.sqrt(t)
            if res < 1:
                return res
            return 1

        def get_It(i,j):
            res = It_hat[i,j] * np.exp(-0.01*t*gamma[i,j]*beta[i,j]*alpha[i,j])
            if -0.005*t*gamma[i,j]*beta[i,j]*alpha[i,j] < 1:
                return res
            return 0
        def get_I(i,j):
            """
            I_p - level of fullness
            I_t - level of svoechasnist
            I_d - level of dostovirnist
            t - time
            """
            return get_Ip(i,j)*get_It(i,j)*get_Id(i,j)
    elif func_type == 'Задані функції':
        def get_alpha(i,j):
            return 0.5*(It_hat[i,j]+Ip_hat[i,j])*alpha_hat[i,j]

        def get_beta(i,j):
            return np.exp(It_hat[i,j]*10**-4)/(1+alpha_hat[i,j])**2
        def get_gamma(i,j):
            return 1.5*np.exp(-0.5*(It_hat[i,j]+Id_hat[i,j]))*alpha_hat[i,j]

        def get_eta(i,j):
            """
            int_inf_ind - integral information index
            alpha  - one of coefs that show change dynamic of int_inf_ind
            """
            return 1 - np.log2(1+alpha_hat[i,j]*get_I(i,j))

        def get_Ip(i,j):
            res = 10*Ip_hat[i,j]*np.log(1+alpha[i,j])*(t+1)**2
            if res < 1:
                return res
            return 1

        def get_Id(i,j):
            res = (1+0.5*beta[i,j]+gamma[i,j]*t)**2
            if res < 1:
                return res
            return 1

        def get_It(i,j):
            res = 0.05*It_hat[i,j]*(2+10**-2*alpha[i,j])*(1-3*beta[i,j]*t)
            if 3*beta[i,j]*t < 1:
                return res
            return 0
        def get_I(i,j):
            """
            I_p - level of fullness
            I_t - level of svoechasnist
            I_d - level of dostovirnist
            t - time
            """
            return get_Ip(i,j)*get_It(i,j)*get_Id(i,j)


    alpha = np.zeros((n,m))
    beta = np.zeros((n,m))
    gamma = np.zeros((n,m))

    for i in range(n):
        for j in range(m):
            if alpha_hat[i,j] != -1:
                alpha[i,j] = get_alpha(i,j)
                beta[i,j] = get_beta(i,j)
                gamma[i,j] = get_gamma(i,j)
    

    for i in range(n):
        for j in range(m):
            if alpha_hat[i,j] != -1:
                t = 10
                hist_I = []
                while True:
                    hist_I.append(get_eta(i,j))
                    if get_It(i,j) == 0.0: break
                    t+=1

    
    step = .1
    t = 10
    T = np.zeros((n,m),dtype=np.float64)
    if func_type == 'Задані функції': t = 0
    while t < 150:
        for i in range(n):
            for j in range(m):
                if alpha_hat[i,j] != -1:
                    alpha[i,j] = get_alpha(i,j)
                    beta[i,j] = get_beta(i,j)
                    gamma[i,j] = get_gamma(i,j)
                    eta = get_eta(i,j)
                    if T[i,j] == 0 and eta > eta_def:
                        T[i,j] = max(t-step,0)
                        
        t += step
    

    T1 = np.zeros((n,m),dtype=np.float64)
    T2 = np.zeros((n,m),dtype=np.float64)
    eta_plus = eta_def
    eta_minus = 0.1
    t=20
    if func_type == 'Задані функції': t = 0
    step = 0.1
    while t < 400:
        for i in range(n):
            for j in range(m):
                if alpha_hat[i,j] != -1:
                    eta = get_eta(i,j)
                    if T2[i,j] == 0 and eta > eta_plus:
                        T2[i,j] = max(t-step,0)
                    if T1[i,j] == 0 and eta > eta_minus:
                        T1[i,j] = max(t-step,0)
                        
        t += step
    Smax = []
    for i in range(n):
        T22 = T2[i]
        Smax.append(np.amax(T22[T22 != 0])) 
    Smin = []
    for i in range(n):
      T11 = T1[i]
      Smin.append(np.amin(T11[T11 != 0]))

    etta = {0.5 : [40, 80 ],
                0.6 : [22, 100 ],
                0.7 : [19, 120 ],
                0.8 : [15, 110 ],
                0.9 : [10, 110 ]
                }


    res = []
    def classify(eta):
        t_minus, t_plus = etta[round(eta, 1)]
        print("eta: ",round(eta, 1))
        for i in range(n):
            if (Smin[i] < t_minus):
                res.append(f"{i} клас особливо небезпечних ситуацій")
            elif (Smax[i] > t_plus):
                res.append(f"{i} клас майже безпечних ситуацій")
            else: res.append(f"{i} клас потенційно небезпечних ситуацій")
    

    classify(eta_def)
    matrix = []
    for i in range(n):
        for j in range(m):
            if (T2[i][j] < 0.01):
                matrix.append(" ")
            else: 
                matrix.append(f"{{{round(T1[i][j], 1)};{round(T2[i][j],1)}}}")
    return res, matrix




def show_plot(DATA_FOLDER, eta_def, func_type):
    n = 4
    m = 7 
    mask = np.array(pd.read_csv(DATA_FOLDER+'mask.csv',delimiter=';',header=None))
    alpha_hat_raw = np.array(pd.read_csv(DATA_FOLDER+'alpha_hat.csv',delimiter=';',header=None))
    It_hat_raw = np.array(pd.read_csv(DATA_FOLDER+'It_hat.csv',delimiter=';',header=None))
    Id_hat_raw = np.array(pd.read_csv(DATA_FOLDER+'Id_hat.csv',delimiter=';',header=None))
    Ip_hat_raw = np.array(pd.read_csv(DATA_FOLDER+'Ip_hat.csv',delimiter=';',header=None))

    def process_data(array):
        data = array.copy()
        data[data =='-']=-1
        data = data.astype(np.float64)
        return data

    alpha_hat = process_data(alpha_hat_raw)
    It_hat = process_data(It_hat_raw)
    Id_hat = process_data(Id_hat_raw)
    Ip_hat = process_data(Ip_hat_raw)

    n,m = alpha_hat.shape

    
    if func_type == 'Наш варіант':
        def get_alpha(i,j):
            return 1 - np.log(1+alpha_hat[i,j]* Ip_hat[i,j])

        def get_beta(i,j):
            return np.exp((Id_hat[i,j]+Ip_hat[i,j])/2) * alpha_hat[i,j]

        def get_gamma(i,j):
            return 2.5*(1+alpha_hat[i,j])*(1+Id_hat[i,j])**2

        def get_eta(i,j):
            """
            int_inf_ind - integral information index
            alpha  - one of coefs that show change dynamic of int_inf_ind
            """
            return 1 - np.log2(1+alpha_hat[i,j]*get_I(i,j)*(40+gamma[i,j]))

        def get_Ip(i,j):
            res = 0.00001*Ip_hat[i,j]*(1+alpha[i,j])*t**2*(1+(1+alpha[i,j]**2)/(Ip_hat[i,j]+Id_hat[i,j]))+.4
            if res < 1:
                return res
            return 1

        def get_Id(i,j):
            res = 0.4+0.01*Id_hat[i,j] * (1+alpha[i,j]+beta[i,j]+gamma[i,j])*np.sqrt(t)
            if res < 1:
                return res
            return 1

        def get_It(i,j):
            res = It_hat[i,j] * np.exp(-0.01*t*gamma[i,j]*beta[i,j]*alpha[i,j])
            if -0.005*t*gamma[i,j]*beta[i,j]*alpha[i,j] < 1:
                return res
            return 0
        def get_I(i,j):
            """
            I_p - level of fullness
            I_t - level of svoechasnist
            I_d - level of dostovirnist
            t - time
            """
            return get_Ip(i,j)*get_It(i,j)*get_Id(i,j)
    elif func_type == 'Задані функції':
        def get_alpha(i,j):
            return 0.5*(It_hat[i,j]+Ip_hat[i,j])*alpha_hat[i,j]

        def get_beta(i,j):
            return np.exp(It_hat[i,j]*10**-4)/(1+alpha_hat[i,j])**2
        def get_gamma(i,j):
            return 1.5*np.exp(-0.5*(It_hat[i,j]+Id_hat[i,j]))*alpha_hat[i,j]

        def get_eta(i,j):
            """
            int_inf_ind - integral information index
            alpha  - one of coefs that show change dynamic of int_inf_ind
            """
            return 1 - np.log2(1+alpha_hat[i,j]*get_I(i,j))

        def get_Ip(i,j):
            res = 10*Ip_hat[i,j]*np.log(1+alpha[i,j])*(t+1)**2
            if res < 1:
                return res
            return 1

        def get_Id(i,j):
            res = (1+0.5*beta[i,j]+gamma[i,j]*t)**2
            if res < 1:
                return res
            return 1

        def get_It(i,j):
            res = 0.05*It_hat[i,j]*(2+10**-2*alpha[i,j])*(1-3*beta[i,j]*t)
            if 3*beta[i,j]*t < 1:
                return res
            return 0
        def get_I(i,j):
            """
            I_p - level of fullness
            I_t - level of svoechasnist
            I_d - level of dostovirnist
            t - time
            """
            return get_Ip(i,j)*get_It(i,j)*get_Id(i,j)



    alpha = np.zeros((n,m))
    beta = np.zeros((n,m))
    gamma = np.zeros((n,m))

    for i in range(n):
        for j in range(m):
            if alpha_hat[i,j] != -1:
                alpha[i,j] = get_alpha(i,j)
                beta[i,j] = get_beta(i,j)
                gamma[i,j] = get_gamma(i,j)
    t = 0
    hist_Ip = []
    hist_Id = []
    hist_It = []
    while True:
        hist_Ip.append(get_Ip(0,0))
        hist_Id.append(get_Id(0,0))
        hist_It.append(get_It(0,0))
        if get_It(0,0) == 0.0: break
        t+=1

    plt.plot(hist_Ip[:],label='Ip')
    plt.show()
    plt.plot(hist_Id[:],label='Id')
    plt.show()
    plt.plot(hist_It[:],label='It')
    plt.legend()
    plt.show()