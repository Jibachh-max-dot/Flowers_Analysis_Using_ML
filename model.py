import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib

#random seed
seed=42

#read original dataset
iris_df= pd.read_csv("data/iris.csv")
iris_df.sample(frac=1, random_state=seed)

#selsecting features and target data
X =iris_df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
y=iris_df[['Species']]

#split data intp train and test sets
#70% training and 30% teseting
X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.3, random_state=seed, stratify=y)

#create an instance of the random forest classifer
clf = RandomForestClassifier(n_estimators=100)

#train the classifier on the training  dataa
clf.fit(X_train,y_train)

#predict on the test set
y_predict=clf.predict(X_test)

#calculate accuracy
accuracy=accuracy_score(y_test,y_predict)
print(f"Accuracy: {accuracy}") #accuracy: 0.88

#save the model to disk
joblib.dump(clf, "output_model/rf_model.sav")
