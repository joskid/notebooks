import numpy as np

def vander(x,N):
  x = np.asarray(x)
  if x.ndim != 1:
    raise ValueError("x must be a 1-D array or sequence")
  v = np.empty((len(x), N), dtype=np.promote_types(x.dtype, int))
  if N > 0:
    v[:,0] = 1
  if N > 1:
    v[:, 1:] = x[:, None]
    np.multiply.accumulate(v[:, 1:], out=v[:, 1:], axis=1)
  return v

