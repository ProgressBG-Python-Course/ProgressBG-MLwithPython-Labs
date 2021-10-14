"""Summary

    Test and measure different ways to count
    unique values in a list/ndarray
"""
import numpy as np
from collections import defaultdict,Counter
import time
import sys


def _statit(f):
  """ simple stat decorator

      prints time taken and the returned object size
  """
  def timed(*args, **kw):
    time_start = time.time()
    result = f(*args, **kw)
    time_end = time.time()

    time_taken = (time_end - time_start) * 1000

    result_size = sys.getsizeof(result)
    docstr = f.__doc__;

    print('{:35s}:{:2.2f} ms. \n "{}" \n\tresult size: {} bytes \n\tresult type {} \n\n'
          .format(f.__name__, time_taken, docstr,result_size, type(result)))

    return result

  return timed


@_statit
def get_values_counts_as_dict_1(atribute_list):
  """unique values count with python dictionary (defaultdict)"""
  values_counts = defaultdict(int)

  for val in atribute_list:
      values_counts[val] += 1

  return values_counts


@_statit
def get_values_counts_as_dict_2(attribute_list):
  """unique values count with python list.count() method"""
  values_counts = {}

  for val in set(attribute_list):
      values_counts[val] = attribute_list.count(val)

  return values_counts

@_statit
def get_values_counts_as_dict_3(attribute_list):
  """unique values count with python Counter dict subclass"""
  return Counter(attribute_list)


@_statit
def get_values_counts_as_2d_array(attribute_list):
    """unique values count with np.unique method"""
    values, counts = np.unique(attribute_list, return_counts=True)

    return [values, counts]


def timing_values_counts(arr):
    print('Testing on Python list with len: {}'.format(len(arr)))

    # print('\n{}:'.format(get_values_counts_as_dict_1.__doc__))
    # %timeit -n1 get_values_counts_as_dict_1(the_list)

    get_values_counts_as_dict_1(arr)
    get_values_counts_as_dict_2(list(arr))
    get_values_counts_as_dict_3(arr)
    get_values_counts_as_2d_array(arr)

if __name__ == '__main__':
  length_factor = 1000000

  arr = np.repeat(np.array(['red', 'green', 'blue']), length_factor)
  np.random.shuffle(arr)

  timing_values_counts(arr)
