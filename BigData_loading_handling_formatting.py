
import csv, json
import numpy as np
import pandas as pd
#import xlutils



def reader(filename):

    csvreader = csv.reader(open(filename, 'rb'), delimiter='\t', quotechar='"')
    data = []
    for line in csvreader:
        r = []
        for entry in line:
            r.append(entry)
        data.append(r)

    return data





def alternative(data):

    json_structure = {
        "header": data[0],
        "data": data[1:]
    }
    return json_structure







def list_to_dict(data):

    data_array = np.array(data)
    dict = {}

    for i in range(len(data_array[0])):
            dict[str(data_array[:,i][0])]= list(data_array[:,i][1:])
    return dict






def main():

    filename = "WEOApr2017all.csv" #'data.txt'
    data = reader(filename)
    #json_structure = alternative(data)
    json_structure = list_to_dict(data)
    open('data6.json', 'wb').write(json.dumps(json_structure))


    with open("data6.json", "r") as f:
        info = json.load(f)

    df = pd.DataFrame(info)

    #print df.head()
    df_years = df.loc[:, '2000':'2015']
    print df_years

    df_years.to_excel('year2000-2015.xls', sheet_name='sheet1', index=False)
    df.to_excel('all_data.xls', sheet_name='sheet1', index=False)



if __name__ == "__main__":
    main()
