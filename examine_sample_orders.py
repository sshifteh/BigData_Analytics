import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def create_dataframe(filename):
    """
    create_dataframe structures the input csv file into a dataframe using pandas

    :param filename: csv file  
    :return: dataframe object
    """

    # Create and structure the dataframe
    csvreader = csv.reader(open(filename, 'rb'), delimiter=',', quotechar='"')
    data = []
    for line in csvreader:
        data.append(line)

    labels = data[0]
    data.pop(0)
    df = pd.DataFrame.from_records(data, columns=labels)

    # Replace "" zip_code entries with 0
    df['zip_code'] = df['zip_code'].replace(to_replace=[""], value='0')

    # Convert string to int or float so they can be sorted when plotting
    df['birth_year'] = df['birth_year'].astype(np.int32)
    df['zip_code'] = df['zip_code'].astype(np.int32)
    df['net_amount'] = df['net_amount'].astype(np.float)
    # df['delivery_date'] = df['delivery_date'].astype(np.string)


    df = df.set_index(['delivery_date'])
    print df

    return df


def zero_zip_code(df):
    return


def statistics(df):
    """

    :param df: Datafram  
    :return: prints the count of each column, min and max, and
    mean and standard deviation of the parameters birth year and net amount spent. 

    """
    print    "\n"
    print "%20s" % "count"
    print "user_id:       ", df["user_id"].count()
    print "birth_year:    ", df["birth_year"].count()
    print "zip_code:      ", df["zip_code"].count()
    # print "delivery_date: ", df["delivery_date"].count()
    print "net_amount:    ", df["net_amount"].count()
    print "\n"
    print "%17s %5s" % ('min', 'max')
    print "net amount: %5.0f, %5.0f" % (round(df["net_amount"].min()), round(df["net_amount"].max()))
    print "birth year: %5.0f, %5.0f" % (round(df["birth_year"].min()), round(df["birth_year"].max()))
    print "\n"

    print "%17s %5s" % ('mean', 'standard deviation')
    print "net amount: %5.0f, %5.0f" % (round(df["net_amount"].mean()), round(df["net_amount"].std()))
    print "birth year: %5.0f, %5.0f" % (round(df["birth_year"].mean()), round(df["birth_year"].std()))
    print "\n"


def zero_net_amount(df):
    return df.loc[df['net_amount'] == 0]


def frequency(df, column, vizualize=False):
    """

    :param df: Dataframe 
    :param column: The column of interest in the dataframe 
    :param vizualize: Boolean paramter for plotting, default is True  
    :return: Plot of frequency of entries in the column of interest


     frequency() stores the entries in the column in a dictionary as keys. 
     The count of each key is its value. plt 
     Finally it plots the counts of each key. 

    """

    column = np.array((df.loc[:, column]))
    dict = {}

    for i in column:
        if i in dict:
            dict[i] += 1
        else:
            dict[i] = 1

    plt.bar(dict.keys(), dict.values(), color='b')
    plt.legend(['Frequency'])
    plt.xlabel('variable')
    plt.ylabel('count')
    plt.show(vizualize)

    # FIXME! connect to 2nd table, of which zip code-ranges correspond to kommune


def range_calculator(age_list):
    """
    :return: list of birth years from 18 to 25, +every 10 years until 65 
    """
    return [2017 - i for i in age_list]


def binning_age(df, bin_list, vizualize=False):
    """
    :param df: dataframe 
    :param bin_list: age groups
    :return: returns the frequency of buyers in each age group 

    """
    bins = bin_list
    group_names = ['65-98', '55-65', '45-55', '35-45', '25-35', '18-25']
    categories = pd.cut(df['birth_year'], bins, labels=group_names)
    df['categories'] = pd.cut(df['birth_year'], bins, labels=group_names)
    df['age_binned'] = pd.cut(df['birth_year'], bins)
    # print categories
    print "%5s %10s" % ('age group', 'frequency')
    print pd.value_counts(df['categories'])
    pd.value_counts(df['categories']).plot.bar()  # hist() doesnt show something nice

    plt.legend(['Frequency'])
    plt.xlabel('age group')
    plt.ylabel('count')
    plt.show(vizualize)


    # print df
    # FIXME! also want to check if user_id is used before, because I want to see new users.


def binning_age_2(df, bin_list, vizualize=False):
    """
    :param df: dataframe 
    :param bin_list: age groups
    :return: returns the frequency of buyers in each age group 

    """
    bins = bin_list
    group_names = ['65-98', '55-65', '45-55', '35-45', '25-35', '18-25']
    categories = pd.cut(df['birth_year'], bins, labels=group_names)
    df['categories'] = pd.cut(df['birth_year'], bins, labels=group_names)
    df['age_binned'] = pd.cut(df['birth_year'], bins)

    plt.bar(df['categories'], df['net_amount'])


    # print df
    # FIXME! also want to check if user_id is used before, because I want to see new users.


def binning_amount(df, vizualize=False):
    """
    :param df: dataframe 
    :param bin_list: amount groups
    :return: returns the frequency of buyers in each age group 

    """
    bins = [0, 50, 200, 400, 600, 800, 1000, 5000]  # [5000,1000, 800, 600, 400, 200, 50, 0]
    group_names = ['1000 - 5000', '800-1000', '600-800', '400-600', '200-400', '50-200', '0-50']
    categories = pd.cut(df['net_amount'], bins, labels=group_names)
    df['categories'] = pd.cut(df['net_amount'], bins, labels=group_names)
    df['age_binned'] = pd.cut(df['net_amount'], bins)
    # print categories
    print "%5s %10s" % ('net amount', 'frequency')
    print pd.value_counts(df['categories'])
    pd.value_counts(df['categories']).plot.bar()  # hist() doesnt show something nice

    plt.legend(['Frequency'])
    plt.xlabel('net amount')
    plt.ylabel('count')
    plt.show(vizualize)


# flest kjoep over 1000
# def threshold_data(df, threshold  = 1000):
# fx if there is an incentive program for buying over 1000 or 500 or something for companies that buy alot.
#        return df.loc[(df['net_amount'] > threshold)]


def crosstab_data(df):
    df_rounded = df.round({'net_amount': 0})
    print pd.crosstab(df_rounded['net_amount'], df_rounded['zip_code'], margins=True)


def test(df):
    year = np.array((df.loc[:, "birth_year"]))
    amount = np.array((df.loc[:, "net_amount"]))

    assert len(year) == len(amount)

    # unknown how it caluclates its bars. Its is not average values.
    plt.plot(x=df['birth_year'], y=df['net_amount'])
    plt.show()


def main():
    vizualize = True
    df = create_dataframe("SampleOrders.csv")
    statistics(df)
    print "___zero zip code______"
    print zero_zip_code(df)
    print "___zero net amount____"
    print zero_net_amount(df)

    for i in ['birth_year', 'zip_code']:
        frequency(df, i, vizualize)

    age_list = [98, 65, 55, 45, 35, 25, 18]
    bin_list = range_calculator(age_list)
    binning_age(df, bin_list, vizualize)
    binning_amount(df, vizualize)
    # binning_age_2(df, bin_list, vizualize)


    # threshold_data(df)
    # crosstab_data()
    # test(df)


if __name__ == "__main__":
    main()


# questions:
# is there a correlation between zip code and amount bought  ?
# a would like to see a plot of the average amount spend based on age, and zip code
# also based on age groups
# I would like to connect zip code to kommune
# i would like to set a threshold value and see in which kommune their net amount is above
# something to do with the delivery dates. How often ( in one month) does one user_id buy?







