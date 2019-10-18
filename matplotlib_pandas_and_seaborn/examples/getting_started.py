# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 0.8.4
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
import matplotlib.pyplot as plt

def simple_plot():
  #On X axis => random numbers from 1 to 15
  x = np.arange(1,16)
  #On Y axis => the power of x-es
  y = x**2

  # lets plot:
  plt.plot(x,y)

# +
# if we want to see the plot (outside of the interactive environment), we must:
#   plt.show()
# -

simple_plot()

def multiple_subplots():
  x = np.array([1,2,3,4])
  y1 = x**2
  y2 = x**3

  axes_names = ['Axes 1', 'Axes 2']

  fig, axes = plt.subplots(nrows=2)
  fig.set_facecolor("#CCCCCC") # redish

  for ax, y, name,color in zip(axes, [y1, y2], axes_names, ["#00FF80","#0080FF"]):
      ax.plot(x, y, color='black')
      ax.set_title(name)
      ax.set_facecolor("#BA5E5E")
      ax.set_ylabel('Y Axis')
      ax.set_xlabel('X Axis')
      # ax.set_xticklabels([1,2,3,4])

  plt.setp(axes[0].spines.values(), color="#F3FF00")
  plt.setp(axes[1].spines.values(), color="#00F3FF")
  plt.tight_layout()
  plt.show()

def empty_figure():
  fig = plt.figure()
  fig.set_facecolor("#B15A5A")

  plt.show()

def add_axes():
  fig = plt.figure()
  fig.set_facecolor("#B15A5A")
  fig.add_subplot(211)
  fig.add_subplot(212)

  plt.show()

def add_subplots():
  fig, axes = plt.subplots(nrows=2)
  fig.set_facecolor("#CCCCCC")

  plt.show()


def axes_labels():
  fig, axes = plt.subplots()
  fig.set_facecolor("#B15A5A")

  axes.set_title("y=sqrt(x)")
  axes.set_xlabel("X",fontsize=14, color="yellow")
  axes.set_ylabel("Y", rotation=0,fontsize=14, color="yellow")
  axes.xaxis.set_label_coords(1.05, 0.02)
  axes.yaxis.set_label_coords(0, 1.05)

  plt.show()


def task_simple_plot():
  x = np.array([4, 9, 25])
  y = np.sqrt(x)

  labels_color = "#FF004F"
  fig_facecolor = "#CCCCCC"

  fig, axes = plt.subplots()
  fig.set_facecolor(fig_facecolor)
  axes.plot(x, y, color='black')
  axes.set_title("y=sqrt(x)")
  axes.set_xlabel("X", color=labels_color, fontsize="14")
  axes.set_ylabel("Y", rotation=0,color=labels_color, fontsize="14")
  axes.xaxis.set_label_coords(1.02, 0.02)
  axes.yaxis.set_label_coords(0, 1 )

  plt.show()

