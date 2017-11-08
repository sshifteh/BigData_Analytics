
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
import numpy as np
from module import create_dataframe, string_to_int_mapping, convert_string_to_int, split_data
from module import roc_auc_SVM
import matplotlib.pyplot as plt





def k_nearest_neighbour(k,X,y, X_train,X_test, y_train, y_test):
    

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # How sensitive is k-NN classification accuracy to the train/test split proportion?

    t = [0.6, 0.5, 0.4, 0.3]
    knn = KNeighborsClassifier(n_neighbors=5)

    for s in t:

        scores = []
        for i in range(1, 100):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - s)
            knn.fit(X_train, y_train)
            scores.append(knn.score(X_test, y_test))
        plt.plot(s, np.mean(scores), 'bo')

    plt.xlabel('Training set proportion (%)')
    plt.ylabel('accuracy');
    plt.show()



    # 3-fold cross validation
    from sklearn.model_selection import cross_val_score

    classifier = KNeighborsClassifier(n_neighbors=5)
    X = X.as_matrix()
    y = y.as_matrix()
    cv_scores = cross_val_score(classifier, X, y)

    print('Cross-validation scores (3-fold):', cv_scores)
    print('Mean cross-validation score (3-fold): {:.3f}'
          .format(np.mean(cv_scores)))




    # Negative class (0) is most frequent
    dummy_majority = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
    y_dummy_predictions = dummy_majority.predict(X_test)

    print 'Test set accuracy with dummy classifier: ', dummy_majority.score(X_test, y_test)
    print 'Most frequent class (dummy classifier)' 
    print confusion_matrix(y_test, y_dummy_predictions)

    knn = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)

    knn_predicted = knn.predict(X_test)
    print 'Test set accuracy with knn classifier: ', knn.score(X_test, y_test)
    print 'k-nearest neighbour classifier'
    print confusion_matrix(y_test, knn_predicted)






def main():

    filename = "bank.csv"
    df = create_dataframe(filename)
    df = string_to_int_mapping(df)
    df = convert_string_to_int(df)
    X, y, X_train, X_test, y_train, y_test = split_data(df)
    k = 100
    k_nearest_neighbour(k,X, y, X_train, X_test, y_train, y_test)





if __name__ == '__main__':
        main()

