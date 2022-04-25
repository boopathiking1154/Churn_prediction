import pickle

# we are loading the model using pickle
model = pickle.load(open('model_churn.pkl', 'rb'))

def predict_churn(df):
    dataset = df.drop(columns=['Customer id'])
    predictions = model.predict(dataset)
    df['ALR'] = predictions
    return (df)
