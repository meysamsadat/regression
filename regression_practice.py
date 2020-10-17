import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.decomposition import PCA


boston = load_boston()
x = boston.data
y = boston.target
des_bos = boston.DESCR
boston.feature_names

df = pd.DataFrame(boston.data,columns=boston.feature_names)
df['Target'] = boston.target
df_des = df.describe()
df.isnull().sum()
min_max_norm = MinMaxScaler(feature_range=(0,1))
df.shape
feature = boston.feature_names
min_max_norm.fit_transform(df[feature])

selected_features = ['LSTAT','RM']
pca = PCA()

my_pca = pca.fit_transform(df[selected_features])

df_scaled_des = df.describe()
sns.set()
sns.distplot(df.Target,bins=30)
corellation = df.corr().round(2)
ax = sns.heatmap(corellation,annot=True)

selected_features = ['LSTAT','RM']
for i,col in enumerate(selected_features):
    plt.subplot(1,2,i+1)
    plt.scatter(df[col],df['Target'],marker='o')
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('Target')

X = df[selected_features].values
Y =df['Target'].values
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.4,random_state=142)

reg = LinearRegression(n_jobs= -1)
reg.fit(x_train,y_train)
reg.score(y_test,y_pred)
y_pred = reg.predict(x_test)
reg.score(y_test,y_pred)

r_sq = reg.score(x_test, y_test)
print('coefficient of determination:', r_sq)

r2_score(y_test,y_pred)

mean_squared_error(y_test,y_pred)
mean_absolute_error(y_test,y_pred)