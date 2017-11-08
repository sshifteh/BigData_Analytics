import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import cm
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import validation_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc



def create_dataframe(filename):
    """
    create_dataframe structures the input csv file into a dataframe using pandas

    :param filename: csv file  
    :return: dataframe object
    """

    # Create and structure the dataframe
    csvreader = csv.reader(open(filename, 'rb'), delimiter=';', quotechar='"')
    data = []
    for line in csvreader:
        data.append(line)

    labels = data[0]
    data.pop(0)
    df = pd.DataFrame.from_records(data, columns=labels)

    return df





def string_to_int_mapping(df):
    """
       string_to_int_mapping takes the dataframe object and makes a maps 
       from string values to integer representations for features that contain string values. 
       If a feature is not binary, that is it has more than two categories, these are represented with 1,2,3 etc. 
       If the feature is binary, like 'yes' or 'no', these are represented with 1 and 0 respectively. 

       :param df: dataframe object 
       :return: dataframe object 
       """


    label_dict = {'yes': 1, 'no': 0}
    df['y'] = df['y'].apply(lambda x: label_dict[x])

    job_dict = {"admin.": 1,"unknown":2,"unemployed":3,"management":4,"housemaid":5,"entrepreneur":6,"student":7,
                                       "blue-collar":8,"self-employed":9,"retired":10,"technician":11,"services":12}
    df['job'] = df['job'].apply(lambda x: job_dict[x])

    marital_dict = {"married":1,"divorced":2,"single":3}
    df['marital'] = df['marital'].apply(lambda x: marital_dict[x])

    education_dict = {"unknown":1 ,"secondary":2,"primary":3,"tertiary":4}
    df['education'] = df['education'].apply(lambda x: education_dict[x])

    default_dict = {'yes':1, 'no':0}
    df['default'] = df['default'].apply(lambda x: default_dict[x])

    housing_dict = {'yes':1, 'no':0}
    df['housing'] = df['housing'].apply(lambda x: housing_dict[x])

    loan_dict = {'yes': 1, 'no': 0}
    df['loan'] = df['loan'].apply(lambda x: loan_dict[x])

    contact_dict = { "unknown":1,"telephone":2,"cellular":3}
    df['contact'] = df['contact'].apply(lambda x: contact_dict[x])

    month_dict = {'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6, 'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}
    df['month'] = df['month'].apply(lambda x: month_dict[x])

    poutcome_dict = {"unknown":3,"other":2,"failure":0,"success":1}
    df['poutcome'] = df['poutcome'].apply(lambda x: poutcome_dict[x])


    return df






def examine_data(df, filename):
    """
    examine data takes the dataframe and the file and examines reports about the data
    in the article such as sample count and success rate. 
    The article reports that the instance count is 45211 for the full data set, and 4521 for the subset. 
    It also reports a that the count of successes are 5289 for the full set (11.7 percent).
    
    
    In addition min and max values are calculated for features where 
    this is of interest. 


    :param df: dataframe object  
    """

    bank_instance_count = 4521
    bank_success_rate = 521

    print   '\n'
    print  'Examine datafile: ' + filename
    print   '\n'
    print 'Check for instances with missing entries: '
    print np.where(pd.isnull(df)) # Nan
    print np.where(df.applymap(lambda x: x == ''))  # finner rad og kolonne index med tom string
    print   '\n'
    print   'Check the reported number of instances: '
    features = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration',
            'campaign', 'pdays', 'previous', 'poutcome']


    value= []
    for i in features:
            if len(df[i]) == bank_instance_count:
                  value.append(True)
            else:
                  value.append(False)
    if False in value:
            print 'There is an error in the sample count!'
    else:
            print 'The sample count is as reported'
    print   '\n'
    print 'Check the reported number of successes:'

    if len(df.loc[df['y'] == 1]) == bank_success_rate:
          print 'The number of succes is as noted in the paper'
    else:
            print 'Reported number of success is not as in paper'
    print   '\n'





def convert_string_to_int(df):
    """
    The feature values of several features are in fact objects not ints. 
    convert_string_to_int converts these feature values to ints to make calculation possible. 
    
    :param df: 
    :return: 
    """

    #print df.dtypes

    object_list = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous', ]
    for i in object_list:
        df[i] = df[i].astype(np.int32)


    return df






def binning_age(df):
    """
    :param df: dataframe
    """
    age_list = [18, 30,50, 87 ]
    group_names = ['18-30', '30-50', '50-87']
    categories = pd.cut(df['age'], age_list, labels=group_names)
    df['categories'] = pd.cut(df['age'], age_list, labels=group_names)
    df['age_binned'] = pd.cut(df['age'], age_list)
    # print categories
    print "%5s %10s" % ('age group', 'frequency')
    print pd.value_counts(df['categories'])
    pd.value_counts(df['categories']).plot.bar()  # hist() doesnt show something nice

    plt.legend(['Frequency'])
    plt.xlabel('age group')
    plt.ylabel('count')
    plt.show()






def statistic(df):

    object_list = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous', ]
    for i in object_list:
            print '%s: min: %g, max: %g, mean: %g, std: %g ' \
                  %(i, df[i].min(), df[i].max(), round(df[i].mean()), round(df[i].std()))





def split_data(df):

    X = df[['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration',
            'campaign', 'pdays', 'previous', 'poutcome']]
    y = df['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


    return X,y,X_train,X_test, y_train, y_test








def gamma_C_test(X,y, X_train,X_test, y_train, y_test):

    # Finding the best combination of the kernel width parameter gamma and the regularization parameter C

    gamma_list = [0.01, 0.1, 1, 5]
    C_list = [0.1, 1, 10, 15]
    for i in gamma_list:
            for j in C_list:
                    print 'gamma: ', i, 'C: ', j
                    clf = SVC(kernel='rbf', gamma=i, C = j).fit(X_train, y_train)
                    print('Accuracy of RBF-kernel SVC on training set: {:.2f}'
                          .format(clf.score(X_train, y_train)))
                    print('Accuracy of RBF-kernel SVC on test set: {:.2f}'
                          .format(clf.score(X_test, y_test)))





def kernalized_SVM_Gaussian(X,y, X_train,X_test, y_train, y_test):

    """
    kernalized_SVM_Gaussian uses a SVM learning model with a Gaussian kernel. 
    
    
    :param X: Feature values  
    :param y: label values 
    :param X_train: training set, 75 percent of the data set 
    :param X_test:  test test, 25 percent of the data set 
    :param y_train: target values used for training  
    :param y_test:  target values used for testing 
    :return: 
    """

    # Chose parameters
    this_gamma = 1
    this_C = 0.1

    # Train the model
    clf = SVC(kernel ='rbf',gamma = this_gamma, C= this_C).fit(X_train, y_train)

    # 5-fold cross validation
    scores = cross_val_score(clf, X, y, cv=5)
    print '***********************************'
    print scores
    print np.mean(scores)


    # Accuracy for unscaled data
    print('Bank data set (unnormalized features)')
    print('Accuracy of RBF-kernel SVC on training set: {:.2f}'
          .format(clf.score(X_train, y_train)))
    print('Accuracy of RBF-kernel SVC on test set: {:.2f}'
          .format(clf.score(X_test, y_test)))

    # Accuracy for scaled data
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    clf = SVC(kernel= 'rbf', gamma = this_gamma, C = this_C).fit(X_train_scaled, y_train)
    print('Bank data set (normalized with MinMax scaling)')
    print('RBF-kernel SVC (with MinMax scaling) training set accuracy: {:.2f}'
          .format(clf.score(X_train_scaled, y_train)))
    print('RBF-kernel SVC (with MinMax scaling) test set accuracy: {:.2f}'
          .format(clf.score(X_test_scaled, y_test)))


    #FIXME ! To use cross validation on normalized data, we get leakage of information about the whole dataset into the training data
    #FIXME ! Instead scaling must be performed for each fold separately.
    #FIxme! Can be done with pipelines.



    # Plotting validation curves to assess the varians-bias tradeoff
    param_range = np.logspace(-3, 3, 4)
    train_scores, test_scores = validation_curve(SVC(), X, y,
                                                 param_name='gamma',
                                                 param_range=param_range, cv=5)

    plt.figure()

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title('Validation Curve with SVM')
    plt.xlabel('$\gamma$ (gamma)')
    plt.ylabel('Score')
    plt.ylim(0.85, 1.05)
    lw = 2

    plt.semilogx(param_range, train_scores_mean, label='Training score',
                 color='darkorange', lw=lw)

    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color='darkorange', lw=lw)

    plt.semilogx(param_range, test_scores_mean, label='Cross-validation score',
                 color='navy', lw=lw)

    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color='navy', lw=lw)

    plt.legend(loc='best')
    plt.show()





def evaluation_SVM(X_train,X_test, y_train, y_test):

    # Negative class (0) is most frequent
    dummy_majority = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
    # Therefore the dummy 'most_frequent' classifier always predicts class 0
    y_dummy_predictions = dummy_majority.predict(X_test)

    #print  y_dummy_predictions
    print dummy_majority.score(X_test, y_test)
    print 'Test set accuray with dummy classifier: ', dummy_majority.score(X_test, y_test)


    # Negative class (0) is most frequent
    confusion = confusion_matrix(y_test, y_dummy_predictions)
    print('Most frequent class (dummy classifier)')
    print confusion

    svm = SVC(kernel='rbf', gamma = 1, C = 0.1).fit(X_train, y_train)
    svm_predicted = svm.predict(X_test)
    confusion = confusion_matrix(y_test, svm_predicted)
    print('Support vector machine classifier (gaussian kernel, C=0.1)')
    print confusion


    print(classification_report(y_test, svm_predicted, target_names=['0', '1']))






def roc_auc_SVM(X_train,X_test, y_train, y_test):

    y_score_svm = SVC(kernel='rbf', gamma = 1, C = 0.1).fit(X_train, y_train).decision_function(X_test)
    falsePositiveRate_svm, truePositiveRate_svm, _ = roc_curve(y_test, y_score_svm)
    roc_auc_svm = auc(falsePositiveRate_svm, truePositiveRate_svm)

    plt.figure()
    plt.xlim([-0.01, 1.00])
    plt.ylim([-0.01, 1.01])
    plt.plot(falsePositiveRate_svm, truePositiveRate_svm, lw=3, label='ROC curve (area = {:0.2f})'.format(roc_auc_svm))
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('ROC curve ', fontsize=16)
    plt.legend(loc='lower right', fontsize=13)
    plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
    plt.axes().set_aspect('equal')
    plt.show()




def decision_trees(X,y,X_train,X_test, y_train, y_test):

    clf = DecisionTreeClassifier().fit(X_train, y_train)

    print('Accuracy of Decision Tree classifier on training set: {:.2f}'
          .format(clf.score(X_train, y_train)))
    print('Accuracy of Decision Tree classifier on test set: {:.2f}'
          .format(clf.score(X_test, y_test)))

    #Result is overfitted. Try pre-prunning methods:
    clf2 = DecisionTreeClassifier(max_depth=3).fit(X_train, y_train)

    print('Accuracy of Decision Tree classifier on training set: {:.2f}'
          .format(clf2.score(X_train, y_train)))
    print('Accuracy of Decision Tree classifier on test set: {:.2f}'
          .format(clf2.score(X_test, y_test)))


    #print ('Feature importances: {}'.format(clf2.feature_importances_))

    feature_importance_values = clf2.feature_importances_

    plt.bar(range(1,17), feature_importance_values, color='b')
    plt.legend(['Feature importances '])
    plt.xlabel('Features')
    plt.ylabel('Value')
    plt.show()





def evaluation_DT(X_train,X_test, y_train, y_test):


    # Dummy accuracy
    dummy_majority = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
    print 'Test set accuray with dummy classifier: ', dummy_majority.score(X_test, y_test)

    dummy_predicted = dummy_majority.predict(X_test)
    confusion = confusion_matrix(y_test, dummy_predicted)
    print 'Dummy classifier confusion matrix: '
    print confusion

    # Tree accuracy
    clf2 = DecisionTreeClassifier(max_depth=3).fit(X_train, y_train)
    print('Accuracy of Decision Tree classifier on test set: {:.2f}'
          .format(clf2.score(X_test, y_test)))


    tree_predicted = clf2.predict(X_test)
    confusion = confusion_matrix(y_test, tree_predicted)
    print 'Decision tree classifier (max_depth = 3) confusion matrix: '
    print confusion






def main():
    filename = "bank.csv"
    df = create_dataframe(filename)
    df = string_to_int_mapping(df)
    examine_data(df,filename)
    df = convert_string_to_int(df)

    binning_age(df)
    statistic(df)
    X, y, X_train, X_test, y_train, y_test = split_data(df)

    gamma_C_test(X,y, X_train,X_test, y_train, y_test)
    kernalized_SVM_Gaussian(X,y, X_train, X_test, y_train, y_test)
    evaluation_SVM(X_train,X_test, y_train, y_test)
    roc_auc_SVM(X_train, X_test, y_train, y_test)

    decision_trees(X,y,X_train,X_test, y_train, y_test)
    evaluation_DT(X_train,X_test, y_train, y_test)










if __name__ == "__main__":
    main()

