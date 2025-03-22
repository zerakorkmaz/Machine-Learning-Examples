
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

df=pd.read_csv("winequality-red.csv")
print(df.head())

print(df.isnull().sum()) ,
#eksik değer yok

#CORELATİON
#Şarap kalitesine etki eden faktörleri tespit edelim.
print(df.corr()["quality"].sort_values(ascending=False))
plt.figure(dpi=100)
sns.heatmap(np.round(df.corr(),2),annot=True,cmap="coolwarm")
plt.show()

#alcohol                 0.476166
#sulphates               0.251397
#citric acid             0.226373 bunlar en çok etkileyen 3 faktör çıkar.

cdf=df[["alcohol","sulphates","citric acid","quality"]]

#split to train&test
split_data=np.random.rand(len(cdf))<0.8
train_df=cdf[split_data]
test_df=cdf[~split_data]

#train
reg=LinearRegression()
train_x=np.asarray(train_df[["sulphates"]])
train_y=np.asarray(train_df[["quality"]])
reg.fit(train_x,train_y)
print("cofficient: %.2f" %reg.coef_[0][0])
print("intercept: %.2f" %reg.intercept_[0])

#model testi

test_x = np.asarray(test_df[["alcohol"]])
test_y = np.asarray(test_df[["quality"]])
test_prediction = reg.predict(test_x)
print(test_prediction)
print('r2 score: %.2f' % r2_score(test_prediction, test_y))

y_pred = reg.predict(test_x)


# Dağılım Grafiği
plt.figure(figsize=(8, 6))
plt.scatter(test_x,test_y, color='blue', alpha=0.5, label='Gerçek Değerler')
plt.scatter(test_x, y_pred, color='red', alpha=0.5, label='Tahmin Edilen Değerler')
plt.xlabel('Alcohol')
plt.ylabel('Quality')
plt.title('Simple Linear Regression - Alcohol vs. Quality')
plt.legend()
plt.show()
