function asianOpt(N = 10000; S0 = 100.0, K = 100.0, ) 

# European Asian option.  
# Euler and Milstein discretization for Black-Scholes.

  r   = 0.05;     # Risk free rate
  q   = 0.0;      # Dividend yield
  v   = 0.2		  # Volatility
  tma = 0.25;     # Time to maturity

  T = 100;         # Number of time steps
  dt = tma/T;      # Time increment

  S = zeros(Float64,T);
  A = zeros(Float64,N);

  for n = 1:N
    S[1] = S0 
    dW = randn(T)*sqrt(dt);
    for t = 2:T
      z0 = (r - q - 0.5*v*v)*S[t-1]*dt;
      z1 = v*S[t-1]*dW[t];
      z2 = 0.5*v*v*S[t-1]*dW[t]*dW[t];
      S[t] = S[t-1] + z0 + z1 + z2;
    end
    A[n] = mean(S);
  end

# Define the payoff and calculate price

  P = zeros(Float64,N);
  [ P[n] = max(A[n] - K, 0) for n = 1:N ];
  price = exp(-r*tma)*mean(P);

end


-----------------------------------------------------

using PyCall
@pyimport myfinx as fx

require("asian-opt")
asianOpt(100000, K=102.0)
1.6653606411083308

asianOpt(1000000, K=102.0)
1.677428310524427

asianOpt(1000000)
2.5910315336258445

-------------------------------------------------------


using Econometrics
using EconDatasets

params = (0.4, 3.2, 0.2)
cirMod = Econometrics.CIR([params...])

cirModX = Econometrics.CIR(0.4, 3.2, 0.2)

u = 0.5; r0 = 4.5; dt = 0.004;
kk = Econometrics.quantile(cirModX, u, r0, dt)


-------------------------------------------------------

>>> import julia
>>> jl = julia.Julia()

>>> import numpy as np
>>> print np.sin(1.5)*jl.bessely0(1.5)

'''
>>> jl.using("Econometrics")
>>> cirMod = jl.eval("Econometrics.CIR(0.4, 3.2, 0.2)")
>>> u = 0.5; r0 = 4.5; dt = 0.004;
>>> kk = jl.eval("Econometrics.quantile(cirMod, u, r0, dt)")
'''

>>> jl.require("asian-opt")
>>> jl.asianOpt()
; FAILS !!!

>>> jl.eval("asianOpt()")
2.64155603283317

>>> jl.eval("asianOpt(1000000)")
2.585134104072215

>>> jl.eval("asianOpt(1000000,K=102.0)")
1.6783584092199437

>>> jl.eval("@elapsed asianOpt(1000000,K=102.0)")
3.00865038






