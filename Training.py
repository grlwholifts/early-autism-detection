import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json

df = pd.read_csv(r'/kaggle/input/innerve-ads/ADScry.csv')
X = df.iloc[:, 0:19].values
y = df.iloc[:, 20]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = Sequential()
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=19))
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

classifier.fit(X_train, y_train, batch_size=10, epochs=100)

y_pred = classifier.predict(X_test)

y_p = []

for index, item in enumerate(y_pred):
    if item >= 0.5:
        y_p.append(1)
    else:
        y_p.append(0)

y_t=y_test.tolist()

count = 0
for i in range(634):
    if y_p[i] == y_t[i]:
        count = count+1


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_t, y_p)
print(cm)