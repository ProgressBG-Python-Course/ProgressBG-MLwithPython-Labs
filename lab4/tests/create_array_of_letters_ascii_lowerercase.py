import string
import numpy as np

for _ in range(2_000_000):
	arr = np.array( list(string.ascii_lowercase)[:10] )

print(arr)

