from model import *

#model implementation and fitting data
xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', learning_rate = 0.4,
                max_depth = 24, alpha = 5, n_estimators = 200)
xg_reg.fit(X_train,y_train)
y_pred = xg_reg.predict(X_test)
# Store the scaling parameters used for the target variable
target_scaling_params = {'mean': df['price'].mean(), 'std': df['price'].std()}

# Perform predictions on your X_test data
y_pred = xg_reg.predict(X_test)

# Inverse scaling on the predicted values to obtain the actual price
y_pred_actual = (y_pred * target_scaling_params['std']) + target_scaling_params['mean']

y_pred_actual = np.exp(y_pred_actual)
print(y_pred_actual)

#some of models will predict neg values so this function will remove that values
def remove_neg(y_test,y_pred):
    ind=[index for index in range(len(y_pred)) if(y_pred[index]>0)]
    y_pred=y_pred[ind]
    y_test=y_test[ind]
    y_pred[y_pred<0]
    return (y_test,y_pred)
#function for evaluation of model
def result(y_test,y_pred):
    y_test_shifted = y_test - y_test.min() + 1
    y_pred_shifted = y_pred - y_test.min() + 1
    
    r=[]
    r.append(mean_squared_log_error(y_test_shifted, y_pred_shifted))
    r.append(np.sqrt(r[0]))
    r.append(r2_score(y_test_shifted,y_pred_shifted))
    r.append(round(r2_score(y_test_shifted,y_pred_shifted)*100,4))
    return (r)

#dataframe that store the performance of each model
accu=pd.DataFrame(index=['MSLE', 'Root MSLE', 'R2 Score','Accuracy(%)'])

y_test_1,y_pred_1=remove_neg(y_test,y_pred)
r8_xg=result(y_test_1,y_pred_1)
print("MSLE : {}".format(r8_xg[0]))
print("Root MSLE : {}".format(r8_xg[1]))
print("R2 Score : {} or {}%".format(r8_xg[2],r8_xg[3]))

xgb.plot_importance(xg_reg)
plt.rcParams['figure.figsize'] = [5, 5]
plt.tight_layout()
plt.savefig('XGBoost-Features-Importance.jpg')
plt.show();

accu=pd.read_csv('./errors.csv',index_col=0)
model_accuracy=accu.loc['Accuracy(%)']

x=list(range(len(model_accuracy)))
y=list(range(0,101,10))
props = dict(boxstyle='round', facecolor='white', alpha=0.8)
plt.figure(figsize=(20,6))
plt.plot(model_accuracy)
plt.yticks(y)
plt.xticks(fontsize=20)
plt.xticks(rotation = (10))
plt.xlabel("Models",fontsize=30)
plt.ylabel("Accuracy(%)",fontsize=30)
plt.title("Performance of Models")
for a,b in zip(x,y):
    b=model_accuracy[a]
    val="("+str(round(model_accuracy[a],2))+" %)"
    plt.text(a, b+4.5, val,horizontalalignment='center',verticalalignment='center',color='green',bbox=props)
    plt.text(a, b+3.5, '.',horizontalalignment='center',verticalalignment='center',color='red',fontsize=50)
plt.tight_layout()
plt.savefig('Overall-Performance.jpg',dpi=600)
plt.show();