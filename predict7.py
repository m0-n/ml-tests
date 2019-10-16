from sklearn.datasets import load_boston
boston = load_boston()

import pandas as pd
import numpy as np

# START ENCODING
data = pd.read_csv("ourdata.csv") 
from sklearn.preprocessing import LabelEncoder

data = pd.get_dummies(data, columns=['Gender','Country', 'Profession', 'University Degree'])

data.to_csv("data_out.csv", index=False)
# END ENCODING

import xgboost as xgb
from sklearn.metrics import mean_squared_error


#X, y = data.iloc[:,:-1],data.iloc[:,-1]
x_cols = [x for x in data.columns if x != 'Income']
x = data[x_cols]
y = data["Income"]

X = x.iloc[0:111993] 
y = y.iloc[0:111993] 

X = X.drop(['source'], axis=1)
X = X.drop(['Instance'], axis=1)
X = X.drop(['Hair Color'], axis=1)


data_dmatrix = xgb.DMatrix(data=X,label=y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=5)

xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 10)

xg_reg.fit(X_train,y_train)

preds = xg_reg.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))

# cv
#
#params = {"objective":"reg:linear",'colsample_bytree': 0.3,'learning_rate': 0.1,
#                'max_depth': 5, 'alpha': 10}
#cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,
#                    num_boost_round=10,early_stopping_rounds=5,metrics="rmse", as_pandas=True, seed=123)
#cv_results.head()

# visualize
#xg_reg = xgb.train(params=params, dtrain=data_dmatrix, num_boost_round=10)
#import matplotlib.pyplot as plt

#xgb.plot_importance(xg_reg)
#plt.rcParams['figure.figsize'] = [5, 5]
#plt.show()

# making the actual predictions
x_cols = [x for x in data.columns if x != 'Income']
x = data[x_cols]
X_real = x.iloc[111993:] 
X_real = X_real.drop(['source'], axis=1)
X_real = X_real.drop(['Hair Color'], axis=1)
X_real = X_real.drop(['Instance'], axis=1)
preds = xg_reg.predict(X_real)



