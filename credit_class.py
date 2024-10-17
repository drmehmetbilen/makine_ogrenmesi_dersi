import pandas as pd

df = pd.read_csv("credit_risk_dataset.csv")

df.fillna(0, inplace=True)

df.drop(['cb_person_default_on_file', 'person_home_ownership', 'loan_intent', 'loan_grade'], axis=1, inplace=True)

df

y = df["loan_status"]
x = df[["person_age", "person_income", "person_emp_length", 
        "loan_amnt", "loan_int_rate", 
        "loan_percent_income", "cb_person_cred_hist_length"]]

from sklearn.svm import SVC
model = SVC()

model.fit(x, y)

prediction = model.predict([[23, 60000, 3, 13000, 7.66, 0.22 , 4]])
print("Prediction:", prediction)