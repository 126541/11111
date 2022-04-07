import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import seaborn as sns

data = pd.read_csv('Advertising.csv',engine='python')
#创建特征列表
feature_cols=['TV', 'radio', 'newspaper']
#使用列表选择data的子集
X=data[feature_cols]
# print(X.head())
y=data['sales']
# print(y.head())
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=1,test_size=0.2)#分为训练集和测试集，训练集占0.8
# print(X_train.shape)
# print(y_test.shape)
linear_reg=LinearRegression()#尝试了下不能直接使用LinearRegression()来进行拟合
model=linear_reg.fit(X_train,y_train)
print('截距是：',model.intercept_)
print('系数是：',model.coef_)
#预测结果
y_predict=linear_reg.predict(X_test)



print ('RMSE',np.sqrt(metrics.mean_squared_error(y_test, y_predict)))#squared先求差的平方再求平均数，再求开平方
plt.figure()
plt.plot(range(len(y_predict)),y_predict,'b',label='predict')
plt.plot(range(len(y_test)),y_test,'r',label='test')
plt.legend()#显示图中的标签
plt.xlabel('the number of sales')#横坐标
plt.ylabel('value of sales')#纵坐标
plt.show()