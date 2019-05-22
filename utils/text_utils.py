import os
import string
import pickle


def get_file_text(dir):
    f = open(dir, 'r')
    t = f.read()
    f.close
    return t

def get_file_list(dir):
    f = open(dir,'r')
    l = f.readlines()
    data_list = []

    for item in l:
        if len(item) < 1:
            continue

        id_, _ = os.path.splitext(item)
        data_list.append(id_)
    
    return set(data_list)

#compiles dictiionary of descriptions, indexed by the image name they are correlated to, and removes erronious 
# characters from description
#returns a description dictionary but also saves dictionary as a .pkl for unpickling later
def compile_description_list(dir):
    f = open(dir, 'r')
    l = f.readlines()

    trans_table_punct = str.maketrans('', '', string.punctuation)
    desc_list = dict()
    to_file_list = []

    for item in l:
        t = item.split()
        if len(item) < 2:
            continue

        id_ = t[0]
        description = t[1:]

        #remove punctuation, uppercase letters, numbers, and trailing letters
        description = [w.lower() for w in description]
        description = [w.translate(trans_table_punct) for w in description]
        description = [w for w in description if w.isalpha()]
        description = [w for w in description if len(w)>1]

        id_, _ = os.path.splitext(id_)
        description = ' '.join(description)

        description = 'startcap ' + description + ' stopcap'

        if id_ not in desc_list:
            desc_list[id_] = [] 
        
        desc_list[id_].append(description) 

    pickle.dump(desc_list, open('descriptions.pkl', 'wb'))

    return desc_list

    #removes any description not in the data set from a description list returned
def filter_desc_list(data):
    desc_list = pickle.load(open('descriptions.pkl', 'rb'))
        
    filtered_desc_list = desc_list.copy()

    for item in desc_list:
        if item not in data:
            del filtered_desc_list[item]


    return filtered_desc_list



def unpickle_desc_list(dir):
    return pickle.load(open(dir, 'rb'))
