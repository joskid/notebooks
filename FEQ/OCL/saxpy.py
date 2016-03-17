import numpy as np
import numpy.random as nr
import time as tm

n = 10000000
a = 2.0
x = nr.randn(n)
y = nr.randn(n)
z = np.zeros(n)

tic = tm.time()
for i in range(n):
    z[i] = a*x[i] + y[i]

toc = tm.time()
print "elapsed: %7.4f seconds. \n\n " % (toc - tic)
