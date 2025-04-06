import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore, skew, boxcox
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, StandardScaler, RobustScaler,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score, ConfusionMatrixDisplay,roc_auc_score, roc_curve
from sklearn.linear_model import  LogisticRegression

df=pd.read_csv("bank-additional-full.csv",sep=";")

#gereksiz sütunları çıkaracağım böylece verimi temizleyeceğim.
columns_to_drop = ['duration', 'pdays', 'contact', 'month', 'day_of_week', 'default']
df = df.drop(columns=columns_to_drop)
print(df.head())

#kategorik verileri sayısal verilere çevireceğim
df['y'] = df['y'].map({'yes': 1, 'no': 0})

cols = df.select_dtypes(include='object').columns
for col in cols:
    df[col]=LabelEncoder().fit_transform(df[col])

print(df.head())

#burada hedef değişkenim y sütunu, X ve y olarak ayıracağım
#train ve test verileri oluşturacağım

X = df.drop('y', axis=1)
y = df['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#X'te bulunan sütunlar farklı ölçeklerde olabilir. bu yüzden değerleri strandart hale getireceğim.
X_train=StandardScaler().fit_transform(X_train)
X_test=StandardScaler().fit_transform(X_test)

#Logistic Refresyon yapacağım bunun için modeli oluşturuyourm.
model=LogisticRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Doğruluk Oranı:", accuracy)

#Görselleştirelim bir de

#Burada, confusion_matrix fonksiyonunu kullanarak doğru ve yanlış tahminlerin sayısını hesaplıyorum ve ConfusionMatrixDisplay ile bunu görselleştiriyorum
matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=["Hayır", "Evet"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

#Eğitim ve test setlerinin doğruluk oranlarını karşılaştıran bir çubuk grafik oluşturuyorum
print(classification_report(y_test, y_pred))
accuracies = [accuracy_score(y_train, model.predict(X_train)), accuracy]
labels = ['Eğitim Verisi', 'Test Verisi']

plt.figure(figsize=(6, 4))
sns.barplot(x=labels, y=accuracies, palette="Blues_d",hue=labels)
plt.ylim(0, 1)
plt.title("Eğitim ve Test Doğruluk Oranları")
plt.ylabel("Doğruluk Oranı")
plt.show()
