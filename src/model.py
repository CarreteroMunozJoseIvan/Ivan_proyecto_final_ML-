import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import os
import pickle

#libraries for preprocessing
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

#libraries for evaluation
from sklearn.metrics import mean_squared_log_error,r2_score,mean_squared_error
from sklearn.model_selection import train_test_split


#libraries for models
from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge
from sklearn.linear_model import LassoCV,RidgeCV


from sklearn.linear_model import Lasso

from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR



import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

#Loading Dataframe

df=pd.read_csv("./data/processed_data.csv")
df=df.drop('id',axis=1)

df2=df.copy()
df.head()
df.drop(columns='posting_date', inplace=True)

num_col=['year','odometer','lat','long']
cat_cols=['region','manufacturer','model','condition','cylinders','fuel','title_status', 'transmission', 'drive', 'size', 'type', 'paint_color', ]

le=preprocessing.LabelEncoder()
df[cat_cols]=df[cat_cols].apply(le.fit_transform)

norm = StandardScaler()
df['price'] = np.log(df['price'])
df['odometer'] = norm.fit_transform(np.array(df['odometer']).reshape(-1,1))
df['year'] = norm.fit_transform(np.array(df['year']).reshape(-1,1))
df['model'] = norm.fit_transform(np.array(df['model']).reshape(-1,1))

#scaling target variable
q1,q3=(df['price'].quantile([0.25,0.75]))
o1=q1-1.5*(q3-q1)
o2=q3+1.5*(q3-q1)
df=df[(df.price>=o1) & (df.price<=o2)]

df['region'] = norm.fit_transform(np.array(df['region']).reshape(-1,1))
df['lat'] = norm.fit_transform(np.array(df['lat']).reshape(-1,1))
df['long'] = norm.fit_transform(np.array(df['long']).reshape(-1,1))

#function to split dataset int training and test


def trainingData(df,n):
    X = df.iloc[:,n]
    y = df.iloc[:,-1:].values.T
    y=y[0]
    X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.9,test_size=0.1,random_state=0)
    folder_path = 'data'


    train_data_path = os.path.join(folder_path, "train.csv")
    X.to_csv(train_data_path, index=False)

    print("Train data saved to", train_data_path)
    folder_path = 'data'
    test_data_path = os.path.join(folder_path, "test.csv")
    y = pd.DataFrame(y)
    y.to_csv(test_data_path, index=False)

    print("Test data saved to", test_data_path)

    return (X_train,X_test,y_train,y_test)

X_train,X_test,y_train,y_test=trainingData(df,list(range(len(list(df.columns))-1)))

#model implementation and fitting data
xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', learning_rate = 0.4,
                max_depth = 24, alpha = 5, n_estimators = 200)
xg_reg.fit(X_train,y_train)
params = xg_reg.get_params()
y_pred = xg_reg.predict(X_test)

models_folder = 'models'
model_path = os.path.join(models_folder, "train_model.pkl")
with open(model_path, 'wb') as file:
    pickle.dump(xg_reg, file)

label_encoder_path = os.path.join(models_folder, "label_encoder.pkl")
with open(label_encoder_path, 'wb') as file:
    pickle.dump(le, file)

ruta_parametros = os.path.join(models_folder, 'model_config.yaml')

# Guarda los parámetros en el archivo pickle
with open(ruta_parametros, 'wb') as archivo:
    pickle.dump(params, archivo)

