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

import numpy as np

def target_colors(y):
  """Shows 3 ways of binary mapping to numpy array

    Args:
        y (TYPE): numpy arr

    Returns:
        TYPE: mapped values arr
  """

  # with python list comprehensions
  # return [CHILD_COLOR if i == 0 else ADULT_COLOR for i in y]

  # with python map function
  # return map(lambda i:CHILD_COLOR if i == 0 else ADULT_COLOR , y)

  # with numpy vectorize
  # return np.vectorize(lambda i: CHILD_COLOR if i==0 else ADULT_COLOR)(y)

  # if we need more than binary mapping:
  c_map = {
    1: ADULT_COLOR,
    0: CHILD_COLOR
  }

  return np.vectorize(lambda i: c_map[i])(y)

CHILD_COLOR = 'green'
ADULT_COLOR = 'red'

y = np.random.choice([0, 1], size=(10,))
print(y)

c = target_colors(y)

print(list(zip(y,c)))
