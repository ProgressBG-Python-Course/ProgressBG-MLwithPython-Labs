import matplotlib.pyplot as plt
import numpy as np

#On X axis => random numbers from 1 to 15
x = np.arange(1,16)
#On Y axis => square of x-es
y = x**2

# lets plot:
plt.plot(x,y)

# if we want to see the plot, we must:
plt.show()

# if we want to save the plot, we must:
fig.savefig("test.png")


