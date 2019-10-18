import numpy as np

def indexing():
  a1 = np.arange(1,5)

  print(a1[0])

def multi_index_proper():
  a1 = np.arange(1,10).reshape(3,3)
  # print(a1)
  a2 = a1[1,2]
  print(a2)

def multi_index_wrong():
  a1 = np.arange(1,10).reshape(3,3)
  # print(a1)
  a2 = a1[1][2]
  print(a2)

multi_index_proper()
multi_index_wrong()
