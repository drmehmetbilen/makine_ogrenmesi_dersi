import numpy as np
from sklearn.neighbors import KNeighborsRegressor



class BasicClassification():
     def __init__(self):
        # data set : [room count, for buy or rent? (1: rent, 0: buy), price]
        self.train_data = np.array([
            [3, 1, 14000], 
            [3, 1, 18000],   
            [3, 1, 15000],  #3+1 for rent 
            [3, 1, 7000],
            [3, 1, 8500],

            [2, 1, 11000],
            [2, 1, 14000],  # 2+1 for rent
            [2, 1, 12000],
            [2, 1, 15000],
            [2, 1, 8000],
            [2, 1, 12500],
            [2, 1, 13000],


            [3, 0, 4980000],
            [3, 0, 2750000],
            [3, 0, 4250000],  # 3+1 in sale
            [3, 0, 4575000],
            [3, 0, 6500000],

            [2, 0, 2750000],
            [2, 0, 7600000],
            [2, 0, 1900000],  # 2+1 in  sale 
            [2, 0, 3850000],
            [2, 0, 2400000],    
          
            
        ])
        
        self.X_train = self.train_data[:, :2]  
        self.y_train = self.train_data[:, 2]    
        
        
        self.knn = KNeighborsRegressor(n_neighbors=2)
        
       
        self.knn.fit(self.X_train, self.y_train)
    
              
     def fit(self, room_count, transaction_type):       
        
       
        sample = np.array([[room_count, transaction_type]])
        predicted_price = self.knn.predict(sample)
        
        return predicted_price[0]

     def predict(self, room_count, transaction_type):
   
        sample = np.array([[room_count, transaction_type]])
        predicted_price = self.knn.fit(sample)
        
        return predicted_price[0]

real_estate_knn = BasicClassification()

user_room_count = int(input("enter room count"))    
user_transaction_type = int(input("if you want to buy press 0 or want to rent press 1"))  # for rent  1,for sale  0


predicted_price = real_estate_knn.fit(user_room_count, user_transaction_type)
print(f"{user_room_count} room's {'for rent' if user_transaction_type == 1 else 'for buy'} guesed price : {predicted_price} TL")
     


