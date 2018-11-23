---
jupyter:
  jupytext:
    formats: ipynb,py:light
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.0'
      jupytext_version: 0.8.4
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.7.0
---

# This IS A NEW TEST

## the imports

```python
import numpy as np
import matplotlib.pyplot as plt
```

## the code

```python
def simple_plot():
  #On X axis => random numbers from 1 to 15
  x = np.arange(1,16)
  #On Y axis => the power of x-es
  y = x**2

  # lets plot:
  plt.plot(x,y)

  # if we want to see the plot, we must:
  # plt.show()


simple_plot()
```

```python

```
