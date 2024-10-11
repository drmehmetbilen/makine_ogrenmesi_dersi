class BasicClassification():
    def __init__(self) -> None:
        pass
    def fit(self,train_data):
        self.train_data = train_data

    def predict(self,sample):
        distance_list = []
        for key, item in self.train_data.items():
            mean_ = sum(item)/len(item)
            distance = abs(mean_ - sample)
            new_distnce = {"class":key, "distance":distance, "mean":mean_, "sample":sample}
            distance_list.append(new_distnce)
        
        distance_list = sorted(distance_list, key=lambda x: x["distance"])
        return distance_list[0]["class"]