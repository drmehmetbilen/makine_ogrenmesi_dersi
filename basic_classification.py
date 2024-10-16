import pandas as pd
from sklearn.svm import SVC

class BasicClassification():
    def __init__(self):
        self.df = pd.read_csv("penguins_size.csv")       
            
        self.df.replace("NA", 0, inplace=True)        
        
        self.df["island"] = self.df["island"].map({'Biscoe': 1, 'Dream': 2, 'Torgersen': 3})
        self.df["sex"] = self.df["sex"].map({'MALE': 1, 'FEMALE': 2})
        
    def fit(self):
        y = self.df["species"]
        X = self.df[["island", "culmen_length_mm", "culmen_depth_mm", "flipper_length_mm", "body_mass_g", "sex"]]
                        
        X.fillna(0, inplace=True)        
                
        self.model = SVC()
        self.model.fit(X, y)
                
        return self.model
    
    def predict(self, sample):
        prediction = self.model.predict([sample])
        print("Guess:", prediction)
        return prediction

classifier = BasicClassification()
model = classifier.fit()

sample = [3, 38.7, 19, 195, 3450, 2]  
classifier.predict(sample)
