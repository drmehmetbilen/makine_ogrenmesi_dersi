from basic_classification import BasicClassification

if __name__ == "__main__":
    
    classifier = BasicClassification()

    data ={
        "big":[5,7,9,12,21,6],
        "small":[1,2,3,4]
    }


    classifier.fit(data)
    print(classifier.predict(100))