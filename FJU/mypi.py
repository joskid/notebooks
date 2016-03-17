import numpy as np
import numpy.random as nr
import time

N = 10000000
K = 0

t0=time.time()
for i in range(N):
  x = nr.random()
  y = nr.random()
  z = x*x + y*y
  if (z < 1.0):
    K += 1

mypi = 4.0*float(K)/float(N)
t1 =   time.time() - t0
print "PI ~= %f after %d trials, took %f sec." % (mypi, N, t1)

