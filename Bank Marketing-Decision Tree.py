import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

df = pd.read_csv('bank-additional-full.csv', sep=';')

# kategorik verileri sayısala çeviriyorum
df_encoded = df.copy()
label_encoders = {}
for column in df_encoded.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df_encoded[column] = le.fit_transform(df_encoded[column])
    label_encoders[column] = le


X = df_encoded.drop("y", axis=1)
y = df_encoded["y"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# decision tree kullanacağım için bu modeli oluşturdum
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

#görselleştireceğim-ancak grafiğe "max depth" ile sınır koyuyorum çünkü ağaç çok büyük oluyor
plt.figure(figsize=(20, 10))
plot_tree(
    model,
    max_depth=5,
    feature_names=X.columns,
    class_names=label_encoders['y'].classes_,
    filled=True,
    rounded=True,
    fontsize=8
)

plt.title("Decision Tree - Bank Term Deposit Prediction")
plt.show()