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

### imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

FIGSIZE=(10,7)
CHILD_COLOR = '#55FF55'
ADULT_COLOR = '#990000'


def target_colors(y):
  return np.vectorize(lambda i: CHILD_COLOR if i==0 else ADULT_COLOR)(y)
  # return [CHILD_COLOR if i == 0 else ADULT_COLOR for i in y]

def get_data():
  """Get/simulate the data

    Returns:
        TYPE: tupple (X, y)
        where:
          X[:,0] -> heights
          X[:,1] -> weights
          y[]    -> class values
  """

  # 40 children -> Gaussian([120, 50], [[90,10 ],[10, 40]])
  # 120 adults -> Gaussian([150, 80], [[60, 10],[10, 40]])

  # test X
  np.random.seed(1111)
  children = np.random.multivariate_normal([120, 50], [[90,10 ],[10, 40]], 40)
  adults = np.random.multivariate_normal([150, 80], [[60, 10],[10, 40]], 120)
  X = np.concatenate((children ,adults),axis=0)

  # test y
  y1 = np.zeros(len(children))
  y2 = np.ones(len(adults))
  y = np.concatenate((y1,y2),axis=0)

  return  (X, y)


def plot_predicted(X,y):
  colors = target_colors(y)
  plt.scatter(X[:,0].tolist(), X[:,1].tolist(),  c=colors, s=50);
  lim = plt.axis()

  colors = target_colors(y)
  plt.scatter(xnew[:, 0], xnew[:, 1], c=colornew, s=30,  alpha=0.3)
  plt.axis(lim);

  plt.show()

def custom_plot(*args, tick_spacing=5, figsize=FIGSIZE, alpha=1, **kw):
  print("figsize: ", figsize)
  # exit();


  fig, ax = plt.subplots(1,1, figsize=figsize)

  colors = target_colors(args[1])
  ax.scatter(args[0][:,0], args[0][:,1],c=colors, s=50, alpha=alpha)

  ax.set_xlabel(kw['xlabel'])
  ax.set_ylabel(kw['ylabel'])
  plt.title(kw['title'], size=14)

  ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
  ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

  return plt


def main():
  # prepare data
  X, y = get_data()
  X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=1111)


  # view data
  c_plt = custom_plot(X,y,
                      xlabel="heights", ylabel="widths",
                      title="Adults/Children",figsize=FIGSIZE,
                      tick_spacing=5,
                      alpha=1)
  c_plt.show()


  # train the model
  model = MultinomialNB()
  model.fit(X_train, y_train);

  # see it (plot decision boundaries):
  # gen lot's of dots:
  heights = np.random.randint(80,190, 5001).reshape(-1,1)
  weights = np.random.randint(20,120, 5001).reshape(-1,1)

  X = np.append(heights, weights, axis=1)
  y_pred= model.predict(X)
  colors = target_colors(y_pred)
  c_plt = custom_plot(X,y_pred,
                      xlabel="heights", ylabel="widths",
                      title="decision boundaries",figsize=FIGSIZE,
                      tick_spacing=5,
                      alpha=1)
  c_plt.show()



  # test model
  # simple test:
  x_pred_simple = [[180,90],[110,20], [130, 60]]
  y_pred_simple = model.predict(x_pred_simple)

  print('X=', x_pred_simple)
  print('y=', np.vectorize(lambda x: 'Child' if x==0 else 'Adult')(y_pred_simple))

  yprob = model.predict_proba(x_pred_simple)
  yprob.round(2)

  tags = ['heights', 'weights']
  res = [zip(i, tags) for i in yprob]
  for a, b in res:
    print(a,b)


  # y_pred = model.predict(X_test)
  # # (X_test, y_test) = test_model(model)


  # c_plt = custom_plot(X,y,
  #                     xlabel="heights", ylabel="widths",
  #                     title="NEW",figsize=FIGSIZE,
  #                     tick_spacing=5,
  #                     alpha=1)


  # colornew= target_colors(y_pred)
  # c_plt.scatter(X_test[:, 0], X_test[:, 1], c='b', alpha=0.4)

  # c_plt.show()

if __name__ == '__main__':
  main()


