class BasicClassification():
    def __init__(self, low_threshold, high_threshold) -> None:
        self.low_threshold = 600000 #ucuz max
        self.high_threshold = 1500000 #pahalı min
    def fit(train_data):
        pass #if else kullanarak yapacağım için eğitim verisi kullanmadım 
    def predict(self, sample):
        
        if sample<self.low_threshold:
            return "ucuz"
        elif self.low_threshold<sample<self.high_threshold:
            return "makul"
        else: 
            return "pahali"
    
print(BasicClassification(600000, 1500000).predict(800000))