from sklearn import linear_model

from data import *

dataset = get_dataset("adults")


X_train, y_train, X_val, y_val = dataset.get_split_data()


clf = linear_model.RidgeClassifier()
clf.fit(X_train, y_train)
clf.predict(X_val)
