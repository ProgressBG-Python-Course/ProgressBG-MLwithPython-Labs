import numpy as np

for _ in range(2_000_000):
	arr = np.array( list(map(chr, range(97, 107))) )

print(arr)

