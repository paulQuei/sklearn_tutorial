
import numpy as np

data = np.arange(0, 100)
np.random.seed(59)
print(data[np.random.permutation(100)[90:]])