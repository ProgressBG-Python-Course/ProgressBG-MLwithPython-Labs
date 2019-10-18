import numpy as np
import time
import sys

SIZE = 1_000_000

l1 = range(SIZE)
l2 = range(SIZE)

a1=np.arange(SIZE)
a2=np.arange(SIZE)

def python_lists():
  result = [(x+y) for x,y in zip(l1,l2)]

def numpy_arrays():
  result = a1 + a2




# python list
start = time.time()
result = [(x+y) for x,y in zip(l1,l2)]
print("python list e2e sum time: {} ms".format((time.time()-start)*1000))

# numpy array
start= time.time()
result = a1 + a2
print("numpy e2e sum time: {} ms", (time.time()-start)*1000)