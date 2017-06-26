



def nested_list():

    """
    Make nested list for each row like [[row1], [row2]].
    :return: nested list
    """

    with open("data.txt", "r") as file:
        all_text = file.read()

    text_list =  all_text.split("\n")
    categories = text_list[0].split("|")
    data = []
    for i in range(1,len(text_list)):
        data.append(text_list[i].split("|"))
    return data



def make_table(data):
    """
    :param data: take the nested list with the data  
    :return: prints the table 
    """

    for i in data:
        print i[0], " "*(17-len(i[0])), \
            i[1], " "*(3 - len(i[1])), \
            i[2], " "*(16 - len(i[2])), \
            i[3], " "*(11 - len(i[3])), \
