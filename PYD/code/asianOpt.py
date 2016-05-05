from math import sqrt, log, exp
from random import random

def box_muller(mu, sigma):
	global use_last
	global y2

	if use_last == 1:
		y1 = y2;
		use_last = 0;
	else:
		while True:
			x1 = 2.0 * random() - 1.0;
			x2 = 2.0 * random() - 1.0;
			w = x1*x1 + x2*x2;
			if w < 1.0:
				break

		w = sqrt( (-2.0 * log(w)) / w)
		y1 = x1 * w
		y2 = x2 * w
		use_last = 1
	
	return mu + sigma*y1

# Option features.
print("Setting option parameters.")

S0  = 100        # Spot price
K   = 102        # Strike price
r   = 0.05       # Risk free rate
q   = 0.0        # Dividend yield
v   = 0.2        # Volatility
tma = 0.25       # Time to maturity

PutCall = 'C'    # 'P'ut or 'C'all
Averaging = 'A'  # 'A'rithmetic or 'G'eometric

# Simulation settings.
print("Setting simulation parameters.")
N = 100000       # Number of simulations.
T = 100          # Nuber of time steps.
dt = tma/T       # Time increment

# Initialize the terminal stock price matrices 
# for the Euler and Milstein discretization schemes.

S = [0] * T
A = [0] * N
P = [0] * N

# Simulate the stock price under the Euler and Milstein schemes.
# Take average of terminal stock price.
print("Looping {} times.".format(N))

use_last = 0
for n in range(N):
	S[0] = S0
	for t in range(1,T):
		Sp = S[t-1]
		dW = box_muller(0.0,1.0)*sqrt(dt)
		z0 = (r-q-v*v*0.5)*Sp*dt
		z1 = v*Sp*dW
		z2 = 0.5*v*v*Sp*dW*dW
		S[t] = Sp + z0 + z1 + z2
		# S(n,t) = S(n,t-1) + (r-q-v^2/2)*S(n,t-1)*dt + v*S(n,t-1)*dW + 0.5*v^2*S(n,t-1)*dW^2;
		
#	A = mean(S(n,:))
	if Averaging =='A':
		Asum = 0.0
		for t in range(T):
			Asum = Asum + S[t]
		A[n] = Asum / float(T)

#	A = exp(mean(log(S(n,:))))
	if Averaging == 'G':
		Asum = 0.0
		for t in range(T):
			Asum = Asum + log(S[t])
		A[n] = exp(Asum / float(T))
	
if PutCall == 'C':
	for n in range(N):
		P[n] = A[n] - K
		if P[n] < 0.0:
			P[n] = 0.0
elif PutCall == 'P':
	for n in range(N):
		P[n] = K - A[n]
		if P[n] < 0.0:
			P[n] = 0.0
			
Psum = 0.0;
for n in range(N):
    Psum = Psum + P[n]
	
Pmean = Psum / float(N)
z0 = exp(-r*tma);
z1= z0*Pmean;

print("Option price = {}".format(z1))
print("Done.")
