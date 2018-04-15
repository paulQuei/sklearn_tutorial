import numpy as np
from sklearn.model_selection import train_test_split

data = np.arange(0, 100)
train_set, test_set = train_test_split(data, test_size=0.1, random_state=59)
print("test_set: \n {} \n".format(test_set))