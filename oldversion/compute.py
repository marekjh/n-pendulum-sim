import numpy as np 
from scipy.integrate import solve_ivp
import pickle
import sys
import os

with open("symbols.pickle", "rb") as f:
    symbols = pickle.load(f)
with open("equations.pickle", "rb") as f:
    equations = pickle.load(f)
    

# Define/collect parameters + initial conditions
M1 = 1
M2 = 1
L1 = 1
L2 = 1
G = 9.81
TH1_0 = 0
TH2_0 = 0
TH1D_0 = 0
TH2D_0 = 3*np.pi
PARAMS = (M1, M2, L1, L2, G, TH1_0*180/np.pi, TH2_0*180/np.pi, TH1D_0*180/np.pi, TH2D_0*180/np.pi)

# Substitute known constant quantities
sub = dict(zip(symbols[4:], (M1, M2, L1, L2, G)))
THETA1DD, THETA2DD = equations
THETA1DD = THETA1DD.xreplace(sub)
THETA2DD = THETA2DD.xreplace(sub)

# Set time scale
interval = 0.02 # s
tmax = 20 # s
t = np.linspace(0, tmax, num=int(tmax / interval))

def main():
    # Get and save solution + parameter data
    y0 = [TH1_0, TH2_0, TH1D_0, TH2D_0]
    print("Computing trajectory...")
    sol = solve_ivp(diff, (t[0], t[-1]), y0, t_eval=t)
    data = (sol.y[0], sol.y[1], PARAMS)
    if len(sys.argv) == 2:
        filename = sys.argv[1]
    else:
        filename = "data"
    with open(os.path.join("saved", f"{filename}.pickle"), "wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    print("Done!")

def diff(t, y):
    (th1, th2, th1d, th2d) = y
    sub = dict(zip(symbols[:4], y))
    th1dd = THETA1DD.evalf(n=5, subs=sub)
    th2dd = THETA2DD.evalf(n=5, subs=sub)
    return (th1d, th2d, th1dd, th2dd)

if __name__ == "__main__":
    main()

