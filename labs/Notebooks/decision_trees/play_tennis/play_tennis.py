# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 0.8.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.7.0
# ---

import numpy as np
import pandas as pd
import inspect

# + {"code_folding": [0]}
DEBUG = 1
def debug(msg, data, level=DEBUG):
    if level==1:
        print('/'*60)
        print("({})>{}: {}".format(inspect.stack()[1][3], msg, data))
        print('\\'*60)

# + {"code_folding": [0]}
class Node:
    def __init__(self, data):
        self.data = data
        self.children = []
        self.title = ''
        self.branch_name = ''

    def add_child(self, obj):
        self.children.append(obj)

    def add_node_title(self, title):
        self.title = title

    def get_node_title(self):
        return self.title if self.title != '' else None

    def add_branch_name(self, name):
        self.branch_name = name

    def get_branch_name(self):
        return self.branch_name if self.branch_name != '' else None

    def get_data(self):
        return self.data

# + {"code_folding": [0]}
def entropy(attribute):
    ''' Calculates entropy of the specified attribute list'''
    entropy = 0

    values, amounts = np.unique(attribute, return_counts=True)
    fraction_amounts = amounts / len(attribute)

    for p in fraction_amounts:
        if p != 0:
            entropy -= p*np.log2(p)

    return entropy

# + {"code_folding": [0, 18]}
def gain(target,  attribute):
    ''' returns a gain value for the attribute list.'''
    values, amounts = np.unique(attribute, return_counts=True)
    fraction_amounts = amounts / len(attribute)

    ''' calculate entropy for each branch.
        that are split apart depending upon the value at each position, so if windy is true it finds all p and n's for that value and multiplies it by i, which is the total amount of times the value appears in the list.
    '''
    result = entropy(target)
    for frac, val in zip(fraction_amounts, values):
        result -= frac * entropy(target[attribute == val])


    # debug("target:", target)
    # debug("attribute:", attribute)
    # debug("gain:", result)
    return result

def gain_longer(target,  attribute):
    '''
        gain(S,A) = Entropy(S) - sum_on_values_A( pt * Entropy(St) )
        gain(S,A) = Entropy(S) - Entropy(S,A)

        i.e. S = PlayTennis, A = Outlook

        pt = |St|/|S| - number of elements in t / number of elements in S

    '''
    # get values/counts for attribute
    attr_vals, attr_counts = np.unique(attribute, return_counts=True)

    # Calculate entropy for the class(target) set.
    entropy_S = entropy(target)
    entropy_S_A = 0

    # calculate entropy for each branch and add it proportionaly to the total entropy of the potential split:
    for i,v in enumerate(attr_vals):
        pt =  attr_counts[i]/len(target)
        entropy_Sv = entropy(target[attribute==v])
        entropy_S_A += pt * entropy_Sv

    # calculate and return the gain:
    gain_S_A = entropy_S - entropy_S_A;
    return gain_S_A

# + {"code_folding": [0]}
def isPure(attribute):
    unique_vals = np.unique(attribute)

    if len(unique_vals) == 1:
        return 1
    return None

def split(attribute):
    '''stores each values' indices from the attribute list in a dictionary structure.'''
    indices ={}

    for i in np.unique(attribute):
        indices[i] = np.where(attribute == i)[0]

    return indices

# + {"code_folding": [0]}
def create_tree(attributes, class_attribute, titles):
    #base case
    if(len(class_attribute) == 0 or isPure(class_attribute)):
        return Node(class_attribute)

    all_gains = np.zeros(len(attributes))

    for i in range (0, len(attributes)):
        all_gains[i] = gain(class_attribute, attributes[i])

    if np.all(all_gains < 1e-6):
       return Node(class_attribute)


    selected_attribute_indexes = np.argmax(all_gains)

    dict = split(attributes[selected_attribute_indexes])

    Title = titles[selected_attribute_indexes]

    #deletes the selected attribute from the data
    attributes = np.delete(attributes, selected_attribute_indexes, axis=0)
    titles = np.delete(titles, selected_attribute_indexes, axis=0)

    # make the new Root node
    root = Node(class_attribute)
    root.title = Title

    for i, j in dict.items():
        class_attribute_subset = class_attribute[j]

        # create a new data array with the selected attribute column removed.
        new_data = np.empty([len(attributes), len(j)])
        new_data = np.ndarray.tolist(new_data)

        for t in range(0, len(attributes)):
            new_data[t] = attributes[t][j]

        new_data = np.array(new_data)

        #recursive step to create tree and pass in split data after.
        child = create_tree(new_data, class_attribute_subset, titles)
        child.add_branch_name(i)
        root.add_child(child)
    return root

# + {"code_folding": [0]}
def print_tree(node, counter = 0):
    '''prints each node and its values. Labels terminal nodes and returns total amount of nodes.'''

    # indent = "  " * depth

    #base case and for terminal nodes printing
    if len(node.children) == 0:
        print("Branch: {}".format(node.branch_name))

        node.data = np.unique(node.data)
        print("\tValues:", node.data)
        print("\tTerminal Node", "\n")
        return counter

    if(node.branch_name != ''):
        print("Branch: ", node.branch_name)

    if(node.title != ''):
        print("\t"+node.title)

    print("Values:", node.data,"\n")
    counter+=1
    for node in node.children:
        counter = print_tree(node, counter)
        counter+=1
    return counter


# def classify(node, example):
#     debug("example", example)

#     if len(node.children) == 0:
#         print("Branch: {}".format(node.branch_name))


#         node.data = np.unique(node.data)
#         print("\tValues:", node.data)
#         print("\tTerminal Node", "\n")

#         answer = node.data
#         return answer

#     if(node.branch_name != ''):
#         print("Branch: ", node.branch_name)
#         classify(node.children)

#     else:
#         classify(node.children[0], example)
#         debug("@@@@@", node.children[0])
#         # for attr, value in node.__dict__.items():
#         #     print(attr, value)

#     if(node.title != ''):
#         example = example[,:]
#         node = node.children[0]
#         classify(node, example)
#         debug("node.title",node.title)
#         debug("example[0,0]",example[1:][:])

# + {"code_folding": [0]}
def prepare_data(data_path):
    ## load data:
    data = pd.read_csv(data_path)
    data.head()

    titles = data.columns[:-1].values
    decisions = data['play'].values
    data = data.drop(labels='play', axis=1).values.T


    # debug("data", data)
    # debug("decisions", list(decisions))
    # debug("titles", list(titles))

    return (data,decisions,titles)

# + {"code_folding": [0]}
def main():
    data_path = "../../datasets/play_tennis_demo.csv"
    (data,decisions,titles) = prepare_data(data_path)

    print(data[0])
    ent_0 = entropy(data[0])
    ent_1 = entropy(data[1])
    ent_2 = entropy(data[2])

    print(titles[0],ent_0)
    print(titles[1],ent_1)
    print(titles[2],ent_2)

    # print('gain_longer: ', gain_longer(decisions,data[0]))
    # print('gain shortest: ', gain(decisions,data[0]))

    # root = create_tree(data, decisions, titles)
    # node_counter = print_tree(root)
    # print('Total number of nodes:', node_counter)

    # print(type(root))

    # classify:
    # example = [titles,data[:,0]]
    # classify(root, example)


# + {"code_folding": [0]}
if __name__ in ("__main__", "__console__"):
    main()
# -


