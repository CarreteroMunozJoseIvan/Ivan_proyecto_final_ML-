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

from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR



import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

#Loading Dataframe

#df=pd.read_csv("vehiclesFinal2.csv")
df=pd.read_csv("./data/processed_data.csv")
#df=df.drop('Unnamed: 0',axis=1)
df=df.drop('id',axis=1)

"""df=df.drop('lat',axis=1)
df=df.drop('long',axis=1)
df=df.drop('region',axis=1)"""

df2=df.copy()
df.head()

num_col=['year','odometer','lat','long']
cat_cols=['region','manufacturer','model','condition','cylinders','fuel','title_status', 'transmission', 'drive', 'size', 'type', 'paint_color', 'posting_date' ]

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

# def remove_neg(y_test,y_pred):
#     ind=[index for index in range(len(y_pred)) if(y_pred[index]>0)]
#     y_pred=y_pred[ind]
#     y_test=y_test[ind]
#     y_pred[y_pred<0]
#     return (y_test,y_pred)

# #function for evaluation of model
# def result(y_test,y_pred):
#     r=[]
#     r.append(mean_squared_log_error(y_test, y_pred))
#     r.append(np.sqrt(r[0]))
#     r.append(r2_score(y_test,y_pred))
#     r.append(round(r2_score(y_test,y_pred)*100,4))
#     return (r)

# accu=pd.DataFrame(index=['MSLE', 'Root MSLE', 'R2 Score','Accuracy(%)'])


LR=LinearRegression()
LR.fit(X_train,y_train)
y_pred=LR.predict(X_test)

# coef = pd.Series(LR.coef_, index = X_train.columns)
# imp_coef = coef.sort_values()
# matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)
# imp_coef.plot(kind = "barh")
# plt.title("Feature importance using Linear Regression Model")
# plt.savefig('Linear-Regression-Feature-Importance.jpg')
# plt.show()

# #estimating MSLE for k=1-9
# R_MSLE=[]
# for i in range(1,10):
#     KNN=KNeighborsRegressor(n_neighbors=i)
#     KNN.fit(X_train,y_train)
#     y_pred=KNN.predict(X_test)
#     error=np.sqrt(mean_squared_log_error(y_test, y_pred))
#     R_MSLE.append(error)
#     print("K =",i," , Root MSLE =",error)

# curve = pd.DataFrame(R_MSLE) #elbow curve 
# plt.figure(figsize=(8,4))
# plt.xticks(list(range(1,10)), list(range(1,10)), rotation='horizontal')
# plt.plot(list(range(1,10)),R_MSLE)
# plt.xlabel('K')
# plt.ylabel('MSLE')
# plt.title('Error Plot for Each K')
# plt.savefig('KNN-Error-Plot.jpg')
# plt.show()


KNN=KNeighborsRegressor(n_neighbors=5) 
KNN.fit(X_train,y_train)
y_pred=KNN.predict(X_test)

# r4_knn=result(y_test,y_pred)
# print("MSLE : {}".format(r4_knn[0]))
# print("Root MSLE : {}".format(r4_knn[1]))
# print("R2 Score : {} or {}%".format(r4_knn[2],r4_knn[3]))
# accu['KNN']=r4_knn

RFR = RandomForestRegressor(n_estimators=180,random_state=0, min_samples_leaf=1, max_features=0.5, n_jobs=-1, oob_score=True)
RFR.fit(X_train,y_train)
y_pred = RFR.predict(X_test)

# r5_rf=result(y_test,y_pred)
# print("MSLE : {}".format(r5_rf[0]))
# print("Root MSLE : {}".format(r5_rf[1]))
# print("R2 Score : {} or {}%".format(r5_rf[2],r5_rf[3]))
# accu['RandomForest Regressor']=r5_rf

# df_check = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
# df_check = df_check.head(25)
# #round(df_check,2)
# df_check.plot(kind='bar',figsize=(10,5))
# plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
# plt.title('Performance of Random Forest')
# plt.ylabel('Mean Squared Log Error')
# plt.savefig('Random-Forest-Performance.jpg')
# plt.show()

#model implementation and fitting data
xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', learning_rate = 0.4,
                max_depth = 24, alpha = 5, n_estimators = 200)
xg_reg.fit(X_train,y_train)
params = xg_reg.get_params()
y_pred = xg_reg.predict(X_test)

#··························· Do I have to do the de-escalate again? 
# y_test_1,y_pred_1=remove_neg(y_test,y_pred)
# r8_xg=result(y_test_1,y_pred_1)
# print("MSLE : {}".format(r8_xg[0]))
# print("Root MSLE : {}".format(r8_xg[1]))
# print("R2 Score : {} or {}%".format(r8_xg[2],r8_xg[3]))


# xgb.plot_importance(xg_reg)
# plt.rcParams['figure.figsize'] = [5, 5]
# plt.tight_layout()
# plt.savefig('XGBoost-Features-Importance.jpg')
# plt.show();

# accu['XGBoost Regressor']=r8_xg

models_folder = 'models'
model_path = os.path.join(models_folder, "train_model.pkl")
with open(model_path, 'wb') as file:
    pickle.dump(xg_reg, file)

ruta_parametros = os.path.join(models_folder, 'model_config.yaml')

# Guarda los parámetros en el archivo pickle
with open(ruta_parametros, 'wb') as archivo:
    pickle.dump(params, archivo)