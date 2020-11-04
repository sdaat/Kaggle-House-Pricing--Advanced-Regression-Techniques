#Initialy necessery modules imported.

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

#Firstly, train and test sets uploaded.
train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")

#Id hasn't any affect on the "SalePrice". So it can be dropped.
train=train.drop(["Id"],axis=1)
test=test.drop(["Id"],axis=1)


sns.heatmap(test.isnull())
