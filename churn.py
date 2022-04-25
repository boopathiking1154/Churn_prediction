import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

dataset = pd.read_csv('customer_li_train.csv')

x_train = dataset.drop(columns=['Customer id', 'ALR'])
y_train = dataset['ALR']

classifier = LogisticRegression(solver='liblinear', random_state=0)
classifier.fit(x_train, y_train)

test_data = pd.read_csv('customer_li_test.csv')
y_test = test_data['ALR']

y_pred = classifier.predict(test_data.drop(columns=['Customer id', 'ALR']))

# Saving model to disk
pickle.dump(classifier, open('model_churn.pkl', 'wb'))

model = pickle.load(open('model_churn.pkl', 'rb'))
print(model.predict([[257, 113, 406, 1.41, 3.94, 4.86, 1, 17, 15]]))
