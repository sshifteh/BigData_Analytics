import pandas
from pandas.plotting import scatter_matrix
from adspy_shared_utilities import plot_fruit_knn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import cm
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score





def create_dataframe(filename):

    df= pandas.read_table(filename)
    df.rename(columns={'Unnamed: 0': 'labels'}, inplace=True)
    X = df[['height', 'width', 'mass']]
    y = df['labels']
    return df,X,y






def k_nearest_neighbour(X,y, vizualize = False):
    X_train, X_test,y_train, y_test = train_test_split(X,y, random_state=0)

    # Examine data
    cmap = cm.get_cmap('gnuplot')
    #scatter = pandas.plotting.scatter_matrix(X_train, c=y_train, marker = 'o', s = 40, hist_kwds = {'bins':15}, figsize = (9,9),cmap=cmap)

    # plotting a 3D scatter plot
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(X_train['width'], X_train['height'], X_train['mass'], c = y_train, marker = 'o', s=100)
    ax.set_xlabel('width')
    ax.set_ylabel('height')
    ax.set_zlabel('mass')
    plt.show(vizualize)

    plot_fruit_knn(X_train, y_train, 5, 'uniform')   # we choose 5 nearest neighbors


    # K-nearest neighbour method:
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    # accuracy
    print 'knn Accuracy, Training data: %.2f' % knn.score(X_train, y_train)
    print 'knn Accuracy, Test data: %.2f' % knn.score(X_test, y_test)







def knn_sensitivity(X,y, vizualize):


    # How sensitive is k-NN classification accuracy to the train/test split proportion?
    proportion_list = [0.6, 0.5, 0.4, 0.3]
    knn = KNeighborsClassifier(n_neighbors=5)
    for s in proportion_list:
        scores = []
        for i in range(1, 100):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - s)
            knn.fit(X_train, y_train)
            scores.append(knn.score(X_test, y_test))
        plt.plot(s, np.mean(scores), 'bo')

    plt.xlabel('Training set proportion (%)')
    plt.ylabel('accuracy');
    plt.show(vizualize)


    # cross validation
    clf = KNeighborsClassifier(n_neighbors = 10)
    X = X.as_matrix()
    y = y.as_matrix()
    cv_scores = cross_val_score(clf, X, y)

    print('Cross-validation scores (3-fold):', cv_scores)
    print('Mean cross-validation score (3-fold): {:.3f}'.format(np.mean(cv_scores)))






def linear_SVM(C,X,y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    clf = SVC(kernel='linear', C=C).fit(X_train, y_train)

    print '\n'
    print 'Linear Kernel: Accuracy, Training data: %.2f' %clf.score(X_train,y_train)
    print 'Linear Kernel: Accuracy, Test data: %.2f' % clf.score(X_test, y_test)
    print '\n'





def kernalized_SVM(C,X,y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    clf = SVC(kernel = 'rbf', C=C).fit(X_train, y_train)

    print '\n'
    print('Fruit data set (unnormalized features)')
    print('RBF kernel: Accuracy: Training data: {:.2f}'
          .format(clf.score(X_train, y_train)))
    print('RBF kernel: Accuracy: Test data: {:.2f}'
          .format(clf.score(X_test, y_test)))
    print '\n'


    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    clf = SVC(kernel = 'rbf', C=C).fit(X_train_scaled, y_train)
    print('Fruit data set (normalized with MinMax scaling)')
    print('RBF-kernel SVC (with MinMax scaling) training set accuracy: {:.2f}'
          .format(clf.score(X_train_scaled, y_train)))
    print('RBF-kernel SVC (with MinMax scaling) test set accuracy: {:.2f}'
          .format(clf.score(X_test_scaled, y_test)))
    print '\n'







def main():
    filename = 'fruit_data_with_colors.txt'
    df, X, y = create_dataframe(filename)

    k_nearest_neighbour(X,y)
    knn_sensitivity(X,y, True)

    C = 10
    linear_SVM(C,X,y)
    kernalized_SVM(C,X,y)




if __name__ == '__main__':
        main()
