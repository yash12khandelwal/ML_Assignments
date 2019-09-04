# 17CH10051
# Yash Khandelwal
# Assignment Number 2
# python3 <program name>

import pandas as pd
import numpy as np

# utility function to prepare the dataframe
# in the orignal data, there were "" in the data which were not getting removed by changing delimiter
def prepare_dataframe(df):
    temp = []
    for j in range(len(df)):
        string = ""
        for i in range(len(df.iloc[j,:])):
            string += str(df.iloc[j,i])
        temp.append([int(char) for char in string.split(',')])

    df = pd.DataFrame(temp, columns = df.columns.values[0].split(","))

    return df

# utility function to create a dictionary of all the probablities from training data

# laplacian smoothing has been used while calculating the conditional probablities
# in laplacian smoothing we create one fake example of all the possible values an attribute can take
# so for every example it increases the value of numerator by 1 and the value of denominator by the no. of fake examples generated
# this is used so that while calculating the probablities, case doesnt occur in which the conditional probablity becomes 0 which will lead 
# the product of probablities becoming 0 
def train(df, target, attributes):

    classes = list(set(df[target[0]]))

    # dictionary to store the probablities
    dictionary = {}

    for p in attributes:
        
        temp = list(set(df[p]))
        temp1 = {}
        
        for k in temp:
            temp3 = []
            for j in classes:
                
                numerator = (df[df[p] == k].index & df[df[target[0]] == j].index).shape[0] + 1
                denominator = df[df[target[0]] == int(j)].shape[0] + len(set(df[p]))
                probability = numerator / denominator
                
                temp3.append(probability)

            temp1[k] = temp3

        dictionary[p] = temp1

    return dictionary

# utility function to predict the labels
def predict(df, target, attributes, test, dictionary):

    classes = list(set(df[target[0]]))
    temp = []

    for k in range(len(test)):

        # declaring the probablity of previous class as 0
        prev_probability = 0
        
        # iterating over both the target classes [0 , 1]
        for j in classes:

            # declairing the probablity for current class as 1
            probability = 1
        
            # caluclating the probablity of current class, by multiplying the probablity of each independent attribute
            for i in attributes:

                probability *= dictionary[i][test[i][k]][j]

            # probablity of target class is also multiplied
            probability *= (df[df[target[0]] == j].shape[0] / df.shape[0])
            
            # if probablity of current class is more than probablity of previous class then the label is class giving more probablity
            if probability > prev_probability:
                
                prev_probability = probability
                label = j

        temp.append(label)

    return temp

# utility function to print the accuracy of predicted labels
def get_accuracy(prediction, labels):
    count = 0

    for i in range(len(labels)):
        if prediction[i] == labels[i]:
            count += 1

    accuracy = (count / len(labels)) * 100

    return accuracy

def main():

    # train file
    df = pd.read_csv("data2_19.csv")
    df = prepare_dataframe(df)

    # test file
    test = pd.read_csv("test2_19.csv")
    test = prepare_dataframe(test)

    # target attribute
    target = df.columns.values[0]
    # features
    attributes = df.columns.values[1:]

    dictionary = train(df, target, attributes)

    train_prediction = predict(df, target, attributes, df, dictionary)

    test_prediction = predict(df, target, attributes, test, dictionary)
    print("Test example predicted classes: ", test_prediction)

    train_accuracy = get_accuracy(train_prediction, df[target[0]])
    print("Train Accuracy: ", train_accuracy, "%")

    test_accuracy = get_accuracy(test_prediction, test[target[0]])
    print("Test Accuracy: ", test_accuracy, "%")

if __name__=="__main__":
    main()
