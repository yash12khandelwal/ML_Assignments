# 17CH10051
# Yash Khandelwal
# Assignment Number 3
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

# utility function to create a decision tree
# attributes has all the attributes in the total dataset
# current dataset is the parent node for the recursion
def tree(current_data, attributes, root):
	
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
	
	attributes.remove(element)
	DataFrameDict = split(current_data, element)

	for key in DataFrameDict.keys():
		root.child.append(Node(str(key)))

	for i in range(len(root.child)):
		root.child[i] = tree(DataFrameDict[root.child[i].key], attributes[:], root.child[i])
	
	return root

def initialize_weight(df):
	df['weight'] = [1/len(df)]*len(df)
	df['check'] = ['correct']*len(df)

	return df

def calculate_total_error(node, df, attributes):
	error = 0
	for i in range(len(df)):
		temp = tree_predict(node, list(df.iloc[i, :4]), attributes)
		if df.iloc[i,3] == temp:
			error += 0
		else:
			error += df.iloc[i, 4]
	return error

def calculate_significance(total_error):
	significance = 0.5 * math.log10((1 - total_error) / total_error)

	return significance

def update_weights(df, significance):
	df.loc[df['check']=='incorrect', ['weight']] *= math.exp(significance)
	df.loc[df['check']=='correct', ['weight']] *= math.exp(-1 * significance)

	df['weight'] /= np.sum(list(df['weight']))
	return df

def adaboost(weighted_data, attributes, no_of_rounds):

    tree_dict = {}
    data_dict = {}
    significance_dict = {}

    data_dict[0] = weighted_data
    for i in range(no_of_rounds):
        tree_dict[i] = Node("data")

        if i >= 1:
            data_dict[i] = data_dict[i - 1].sample(n = len(weighted_data),replace = True, weights= 'weight', random_state = 1)

        tree_dict[i] = tree(data_dict[i].iloc[:,:4], attributes[:], tree_dict[i])

        total_error = calculate_total_error(tree_dict[i], data_dict[i], attributes)
        significance_dict[i] = calculate_significance(total_error)

        data_dict[i] = update_weights(data_dict[i], significance_dict[i])

    return (data_dict, significance_dict, tree_dict)

def traverse(node, path = []):
    path.append(node.key)
    if len(node.child) == 0:
        print(path)
        path.pop()
    else:
        for child in node.child:
            traverse(child, path)
        path.pop()

def get_accuracy(prediction, labels):
    count = 0

    for i in range(len(labels)):
        if prediction[i] == labels[i]:
            count += 1

    accuracy = (count / len(labels)) * 100

    return accuracy

def get_results(test_data, tree_dict, significance_dict, attributes):
    results = []
    for i in test_data.values:
        prediction = 0
        for j in range(3):
            temp = tree_predict(tree_dict[j], list(i[:3]), attributes)
            if temp == 'yes':
                temp = 1
            if temp == 'no':
                temp = -1

            temp *= significance_dict[j]
            prediction += temp
        
        if prediction >= 0:
            prediction = 'yes'
        else:
            prediction = 'no'

        results.append(prediction)

    return results

def tree_predict(node, row, attributes):
    # print(row)

    j = 0
    while j <= len(attributes):
        # print("#########", len(node.child))
        k = 0

        if node.child[k].key == 'yes' or node.child[k].key == 'no':
            return node.child[k].key

        while k < len(node.child):

            if node.child[k].key == 'yes' or node.child[k].key == 'no':
                return node.child[k].key

            if node.child[k].key in row:
                node = node.child[k]
                break
            else:
                k += 1
        j += 1

# main function to read the data and call other utility functions
def main():
	data = pd.read_csv("data3_19.csv")
	test_data = pd.read_csv("test3_19.csv", header = None)

	attributes = list(data.columns.values)

	target = attributes[-1]
	attributes = attributes[:-1]

	weighted_data = initialize_weight(data)
	print("################ Training Started #################")
	data_dict, significance_dict, tree_dict = adaboost(weighted_data, attributes[:], 3)
	print("################ Training Completed #################")

	results = get_results(test_data, tree_dict, significance_dict, attributes)
	print("\n", "TEST ACCURACY: ", get_accuracy(results, list(test_data.iloc[:, 3])))

if __name__ == "__main__":
    main()
