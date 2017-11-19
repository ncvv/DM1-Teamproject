import pandas as pd
import io_util as io
from sklearn.neighbors import KNeighborsClassifier

def naive_bayes(dataset):
    '''Trains and tests a Naive Bayes Classifier with selected features'''
    ###können wir das nicht für alle methoden verwenden bis zu data_train, data_test,...?
    from sklearn.naive_bayes import GaussianNB
    from sklearn.model_selection import train_test_split
    from category_encoders.ordinal import OrdinalEncoder
    from sklearn.metrics import accuracy_score
    
    encoder = OrdinalEncoder()
    dataset_encoded = encoder.fit_transform(dataset)
    listings_data=dataset_encoded.drop(columns=['id','perceived_quality'])
    listings_target= dataset_encoded['perceived_quality']
    
    data_train, data_test, target_train, target_test = train_test_split(listings_data, listings_target, test_size=0.2, random_state=42, stratify=listings_target)
    
    naive_bayes = GaussianNB()
    naive_bayes.fit(data_train, target_train)
    prediction = naive_bayes.predict(data_test)

    #nbresults=pd.DataFrame(data_test)
    #quality_predicted = nbresults.assign(predicted_quality=prediction)
    #io.write_csv(quality_predicted, '../data/playground/naivebayes.csv')
    
    accuracy=accuracy_score(target_test, prediction)
    print('Accuracy of Naive Bayes Classifier:{}'.format(accuracy))

dataset = io.read_csv('../data/playground/dataset.csv')
naive_bayes(dataset)

def knn(dataset, knn_estimator, data_train, target_train, data_test, target_test):
    '''KNN'''
    knn_estimator = KNeighborsClassifier(4)
    knn_estimator.fit(data_train, target_train)

    predict = knn_estimator.predict(data_test)
    print('Prediction of KNN Classifier:{}'.format(predict))

    accuracy = knn_estimator.score(target_test, predict)
    print('Accuracy of KNN Classifier:{}'.format(accuracy))
