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
    '1': ADULT_COLOR,
    '0': CHILD_COLOR
  }

  return np.vectorize(lambda i: c_map[str(i)])(y)



CHILD_COLOR = 'green'
ADULT_COLOR = 'red'

y = np.random.choice([0, 1], size=(10,))
print(y)

c = target_colors(y)

print(list(zip(y,c)))