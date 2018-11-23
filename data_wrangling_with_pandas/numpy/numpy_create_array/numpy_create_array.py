import numpy as np

def from_object():
  # from Python list object
  a1 = np.array([1,2,3])

  # set explicit the data type
  a2 = np.array([1,2,3], dtype=float)
  a3 = np.array([1,2,3], dtype=bool)


  # type coercion
  a4 = np.array([1, 2, 3.0])

  print(a1)
  print(a2)
  print(a3)
  print(a4)

def from_range():
  # from Python list object
  a1 = np.arange(5)
  a2 = np.arange(5,10)
  a3 = np.arange(1,10,2)

  print("\nnp.arange(5):",a1)
  print("\nnp.arange(5,10):",a2)
  print("\nnp.arange(1,10,2):",a3)

def fill_with_value():
  a1 = np.full( (3,2), 2)
  print(a1)


def multi_dim():
  a1 = np.arange(10).reshape(2,5)

  print(a1)

fill_with_value()