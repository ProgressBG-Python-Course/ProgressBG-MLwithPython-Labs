import numpy as np

a1 = np.array([[1,2,3], [4,5,6]],dtype="int8")
print(a1)

# shape - tuple of array dimensions
print("{:12s}{}".format("a1.shape:",a1.shape))

# ndim - number of array dimensions
print("{:12s}{}".format("a1.ndim:",a1.ndim))

# dtype - data-type of the array’s elements
print("{:12s}{}".format("a1.dtype:",a1.dtype))

# size - number of elements in the array
print("{:12s}{}".format("a1.size:",a1.size))

# itemsize - length of one array element in bytes
print("{:12s}{}".format("a1.itemsize:",a1.itemsize))

# nbytes - total bytes consumed by the elements of the array
print("{:12s}{}".format("a1.nbytes:",a1.nbytes))
