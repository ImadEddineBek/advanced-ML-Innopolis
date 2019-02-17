import pandas as pd
from patsy import dmatrix
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import train_test_split

boston = load_boston()
print(boston.DESCR)

bos = pd.DataFrame(boston.data)
bos.columns = boston.feature_names
bos['PRICE'] = boston.target

data_x = bos['CRIM']
data_y = bos['PRICE']

train_x, valid_x, train_y, valid_y = train_test_split(data_x, data_y, test_size=0.33, random_state=1)

transformed_x = dmatrix("bs(train, knots=(2,25,60), degree=3, include_intercept=False)", {"train": train_x},
                        return_type='dataframe')

fit1 = sm.GLM(train_y, transformed_x).fit()

transformed_x2 = dmatrix("bs(train, knots=(10,25,50,65),degree =3, include_intercept=False)", {"train": train_x},
                         return_type='dataframe')

fit2 = sm.GLM(train_y, transformed_x2).fit()

pred1 = fit1.predict(
    dmatrix("bs(valid, knots=(2,25,60), include_intercept=False)", {"valid": valid_x}, return_type='dataframe'))
pred2 = fit2.predict(dmatrix("bs(valid, knots=(10,25,50,65),degree =3, include_intercept=False)", {"valid": valid_x},
                             return_type='dataframe'))

rms1 = sqrt(mean_squared_error(valid_y, pred1))
print(rms1)
rms2 = sqrt(mean_squared_error(valid_y, pred2))
print(rms2)

xp = np.linspace(valid_x.min(), valid_x.max(), 70)

pred1 = fit1.predict(dmatrix("bs(xp, knots=(2,25,60), include_intercept=False)", {"xp": xp}, return_type='dataframe'))
pred2 = fit2.predict(
    dmatrix("bs(xp, knots=(10,25,50,65),degree =3, include_intercept=False)", {"xp": xp}, return_type='dataframe'))

plt.scatter(data_x, data_y, facecolor='None', edgecolor='k', alpha=0.1)
plt.plot(xp, pred1, label='3 knots')
plt.plot(xp, pred2, color='r', label='4 knots')
plt.legend()
plt.xlabel('per capita crime rate by town')
plt.ylabel('Price')
plt.show()
