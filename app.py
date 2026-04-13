import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from kNN import Knn

df = pd.read_csv('data/breast-cancer.csv')

# Drop ID column
df = df.drop(columns=['id'])

X = df.iloc[:,1:]
y = df.iloc[:,0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print(accuracy_score(y_test, y_pred))

apnaKnn = Knn(k=5)

apnaKnn.fit(X_train, y_train)
y_pred1 = apnaKnn.predict(X_test)
print(accuracy_score(y_test, y_pred1))

