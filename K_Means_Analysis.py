import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('seattle-weather.csv')

sns.pairplot(df)
plt.show(block=True)
