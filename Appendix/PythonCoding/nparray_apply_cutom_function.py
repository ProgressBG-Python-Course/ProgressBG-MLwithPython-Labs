# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 0.8.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.7.0
# ---

"""Summary

    Test and measure different ways of applying a custom
    function (in the examples - mapit()) to a numpy array
"""
import numpy as np
import pandas as pd
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

    print('{:35s}:{:2.2f} ms. \n\tresult size: {} bytes \n\tresult type {} \n\n'
          .format(f.__name__, time_taken, result_size, type(result)))

    return result

  return timed


def mapit(i):
  """ Intentionally not used lambdas, in order to keep code readable
      for non-lambdas programmers. But it's perfectly ok to use
      lambda i: mappings[i]

      For binary mappings is also ok not to use a function at all,
      but just the statement: 'red' if i == 0 else 'green'
  """
  return mappings[i]


@_statit
def mapping_with_list_comprehension(y):
  return [mapit(i) for i in y]


@_statit
def mapping_with_map_function(y):
  return list(map(mapit, y))


@_statit
def mapping_with_npvectorize(y):
  # note, that we include the np.vectorize(f) init time here
  return np.vectorize(mapit)(y)


@_statit
def mapping_with_pandas_map(y):
  s = pd.Series(y)
  return s.map(mappings)


@_statit
def mapping_with_pandas_apply(y):
  s = pd.Series(y)
  return s.apply(mapit)


if __name__ == '__main__':
  mappings = {
    0: 'red',
    1: 'green',
    2: 'blue'
  }
  ARRAY_SIZE = 10
  y = np.random.choice([0, 1, 2], size=ARRAY_SIZE)

  print
  mapping_with_list_comprehension(y)
  mapping_with_map_function(y)
  mapping_with_npvectorize(y)
  mapping_with_pandas_map(y)
  mapping_with_pandas_apply(y)
