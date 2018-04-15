import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.DataFrame({
    "Year": [1999,2000,2001,2002,2003,2004,2005,2006,
        2007,2008,2009,2010,2011,2012,2013,2014],
    "Price": [3800,3900,4000,4200,4500,5500,6500,7000,
        8000,8200,10000,14000,13850,13000,16000,18500]})
data.plot(kind="scatter", x="Year", y="Price", c="B", s=100)

plt.plot([1999, 2020], [3800, 20000], c="coral")
plt.plot([1999, 2020], [5500, 30000], c="yellowgreen")
plt.plot([1999, 2020], [2000, 50000], c="blueviolet")

plt.show()

