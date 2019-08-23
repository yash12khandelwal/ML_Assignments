# 17CH10051
# Yash Khandelwal
# Assignment Number 1
# python3 <program name>

import pandas as pd
import numpy as np
import math

# structure to save Decision Tree
class Node :
    def __init__(self ,key): 
        self.key = key  
        self.child = []

# utility function to caluclate the entropy of dataframe
# for calcuation of entropy the formula of entropy is used in which once we take the probablity of occurence of yes and in other probablity of occurence of no
# only required attribute is the last column of the dataframe
def entropy(df):
    yes_count = 0
    no_count = 0

    for i in range(len(df.iloc[:, -1])):
        if df.iloc[:, -1][i] == "yes":
            yes_count += 1
        
        if df.iloc[:, -1][i] == "no":
            no_count += 1
        continue

    if yes_count == 0 or no_count == 0:
        return 0
    
    entropy = -1 * (((yes_count/(yes_count + no_count))*math.log((yes_count/(yes_count + no_count)), 2)) + ((no_count/(yes_count + no_count))*math.log((no_count/(yes_count + no_count)), 2)))

    return entropy

# utility function to calculate the information gain from DataFrameDict and parent information
# information gain is the difference of parent_entropy and weighted sum of children entropy
# required attributes are dictionary of splited dataframe and the parent entropy
def information_gain(DataFrameDict, parent_entropy, l):
    child_entropy = 0

    for key in DataFrameDict.keys():
        child_entropy += (len(DataFrameDict[key])/l)* entropy(DataFrameDict[key].reset_index(drop = True))

    return parent_entropy - child_entropy

# utility function to split the dataframe on the basis of attribute
def split(df, attribute):
    temp = list(set(df[attribute]))

    DataFrameDict = {elem : pd.DataFrame for elem in temp}

    for key in DataFrameDict.keys():
        DataFrameDict[key] = df[:][df[attribute] == key]

    return DataFrameDict

# utility function to traverse the tree and print the tree hierarchialy
def traverse_tree(root, indent): 
   
    # Stack to store the nodes  
    nodes=[] 
  
    # push the current node onto the stack  
    nodes.append(root)  
    
    # loop while the stack is not empty  
    while (len(nodes)):   
    
        # store the current node and pop it from the stack  
        curr = nodes[0]  
        nodes.pop(0)
        # current node has been travarsed

        for i in range(len(indent)):
            if curr.key in indent[i]:
                print((i + 1)*"\t", curr.key)
                # print(curr.key)
         
        # store all the childrent of current node from  
        # right to left.  
        for it in range(len(curr.child)-1,-1,-1):   
            nodes.insert(0,curr.child[it])

# utility function to create a decision tree
# order stores the order in which the tree is being splitted in the recursion
# attributes has all the attributes in the total dataset
# current dataset is the parent node for the recursion
def tree(orignal_data, current_data, attributes, root, order):
    
    if len(set(current_data.iloc[:, -1])) == 1:
        
        root.child.append(Node(str(list(set(current_data.iloc[:, -1]))[0])))
        return root
    
    if len(attributes) == 0:
        
        temp = np.unique(current_data.iloc[:, -1], return_counts=True)
        index = np.argmax(temp[1])
        root.child.append(Node(temp[0][index]))
        return root
    
    parent_entropy = entropy(current_data.reset_index())
        
    max_gain = -10000
    for i in attributes:    
        DataFrameDict = split(current_data, i)

        gain = information_gain(DataFrameDict, parent_entropy, len(current_data))

        if gain >= max_gain:
            max_gain = gain
            element = i
    
    order.append(element)
    attributes.remove(element)
    
    DataFrameDict = split(current_data, element)

    for key in DataFrameDict.keys():
        root.child.append(Node(str(key)))

    for i in range(len(root.child)):
        root.child[i] = tree(orignal_data, DataFrameDict[root.child[i].key], attributes[:], root.child[i], order)
    
    return root

# main function to read the data and call other utility functions
def main():
    data = pd.read_csv("data1_19.csv")

    root = Node("data")

    attributes = list(data.columns.values)
    target = attributes[-1]
    attributes = attributes[:-1]

    order = []
    root = tree(data, data, attributes, root, order)
    order = order[:3]
    indent = []
    for i in order:
        indent.append(list(set(data[i])))
    indent.append(list(set(data[target])))
    traverse_tree(root, indent)

if __name__=="__main__":
    main()