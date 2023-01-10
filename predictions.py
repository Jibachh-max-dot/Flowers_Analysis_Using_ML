import joblib 
def predict(data):
    clf = joblib.load("output_model/rf_model.sav")
    return clf.predict(data)