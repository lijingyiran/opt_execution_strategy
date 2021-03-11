import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

dt = 1
sigma = 0.2/np.sqrt(252) #scaled by number of trading days in a year
T = 60 #seconds
n = int(T/dt)
h = 0.1
t = np.linspace(0., T, n)
r = 1000 #initial inventory
a = 1000
k = 0.2
b = 0.3
xi = 0.05
phi = 0.01
gamma = np.sqrt(phi/k)
S = 10000*np.ones(T)
x = 20000 #initial wealth
Shat = 200 #initial execution price
delta = 5 #bid-ask spread
alpha = 0.001 #terminal liquidation penalty
s0 = 1000 #initial mid-price


def get_qt(r, t, T, phi):
    qt = np.ones(T)
    
    if phi != 0: #opt soln
        xi_num = a - 1/2 * b + np.sqrt(k * phi)
        xi_denom = a + 1/2 * b + np.sqrt(k * phi)
        xi = xi_num/xi_denom
        for i in range(T):
            num = xi * np.exp(gamma*(T-i)) - np.exp(-gamma*(T-i))
            denom = xi * np.exp(gamma*T) - np.exp(-gamma*T)
            qt[i] = num/denom * r
    
    else: #if twap        
        for i in range(T):
            qt[i] = (1-i/T)*r
            
    return qt     

    
qt = get_qt(r, t, T, phi)


def get_vt(qt, t, T, phi):
    if phi != 0:
        vt = np.ones(T)
        xi_num = a - 1/2 * b + np.sqrt(k * phi)
        xi_denom = a + 1/2 * b + np.sqrt(k * phi)
        xi = xi_num/xi_denom
        for i in range(T):
            num = xi * np.exp(gamma*(T-i)) + np.exp(-gamma*(T-i))
            denom = xi * np.exp(gamma*(T-i)) - np.exp(-gamma*(T-i))
            vt[i] = num/denom * qt[i] * gamma
            
    else: #twap
        vt = np.ones(T)*(r/T)
    
    return vt


vt = get_vt(qt, t, T, phi)


def f(v):
    return k*v

def g(v):
    return b*v


#calculate st using euler discretization
def get_st(s0, vt, h):
    st = np.zeros(T)
    st[0] = s0
    for i in range(1, T):
        st[i] = st[i-1] - g(vt[i-1]) * h + sigma * np.sqrt(h) * np.random.randn()
    return st


def get_sthat(st, vt, delta): 
    sthat = st - (delta/2 + f(vt))
    return sthat


def get_xt(sthat, vt):
    xt = np.zeros(T)
    xt[0] = x
    for i in range(1, T):
        xt[i] = xt[i-1] +sthat[i-1] * vt[i-1]*h
    return xt


def inv_penalty(t, T, phi):
    pen = np.zeros(T)
    xi_num = a - 1/2 * b + np.sqrt(k * phi)
    xi_denom = a + 1/2 * b + np.sqrt(k * phi)
    xi = xi_num/xi_denom
    f = lambda u: ((xi * np.exp(gamma * (T-u)) - np.exp(-gamma * (T-u)))/
                   (xi * np.exp(gamma * T) - np.exp(-gamma * T)) * r)**2
    for i in range(T):
        pen[i], err = integrate.quad(f, t[i], T)
    
    return phi * pen


def plot_process(process, t):
    fig, ax = plt.subplots(1,1,figsize=(8,4))
    plot = ax.plot(t, process, lw = 2)
    return plot

# visualize different processes for different phis
myphi = np.array([10e-10, 0.001, 0.01, 0.2, 1])
symb = np.array(["o", "*", "^", "1"])
labels = [r"$\phi =$" + str(i) for i in myphi]


def overlay_plot(process_name, labels, symb):
    """
    (str, np array of floats/ints, np array of str) -> plt
    """
    
    fig, ax = plt.subplots()
    fig.set_size_inches(10.5, 5.5)
    color_idx = np.linspace(0, 1, myphi.shape[0])
    
    for i, j in zip(color_idx, range(0, myphi.shape[0])):
        
        qt = get_qt(r, t, T, myphi[j])
        vt = get_vt(qt, t, T, myphi[j])
        
        if process_name == "qt":
            line = qt
        elif process_name == "vt":
            line = vt
        elif process_name == "st":
            line = get_st(s0, vt, h)
        elif process_name == "xt":
            st = get_st(s0, vt, h)
            sthat = get_sthat(st, vt, delta)
            line = get_xt(sthat, vt)
        else:
            print("Invalid process input")
            break
        
        plt1 =ax.plot(t[10:], line[10:], linestyle = '-', color = plt.cm.rainbow(i), label = labels[j])
    
    ax.legend()
    ax.set_xlabel(r"Time (seconds)", fontsize = 18)
    ax.set_ylabel(r"{}".format(process_name), fontsize = 18)
    fig.canvas.draw()


# objective function in terms of expectation
def obj(alpha, num_runs, phi):
    cash_vec = np.zeros(num_runs)
    for i in range(num_runs):
        qt = get_qt(r, t, T, phi)
        vt = get_vt(qt, t, T, phi)
        st = get_st(s0, vt, h)
        sthat = get_sthat(st, vt, delta)
        xt = get_xt(sthat, vt)
        pen = inv_penalty(t, T, phi)
        cash_vec[i] = xt[T-1] + qt[T-1]*(st[T-1] - alpha*qt[T-1]) - pen[T-1]
    cash = np.mean(cash_vec)
    return cash


cash_twap = obj(100, 100, 0) #alpha -> infty and phi = 0 opt soln is twap
cash_opt = obj(0.1, 100, 0.001)
print("Terminal wealth based on TWAP is $", round(cash_twap, 2))
print("Terminal wealth based on Optimal Execution stratgy is $", round(cash_opt, 2))
    