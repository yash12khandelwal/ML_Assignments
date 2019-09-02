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

# utility function to predict the labels
def predict(df, target, attributes, test):

    # list of classes in the target set [0 , 1]
    classes = list(set(df[target[0]]))
    
    # list to store the predicted labels
    temp = []

    # iterating over all the test dataset
    for k in range(len(test)):

        # declaring the probablity of previous class as 0
        prev_probability = 0
        
        # iterating over both the target classes [0 , 1]
        for j in classes:

            # declairing the probablity for current class as 1
            probability = 1
        
            # caluclating the probablity of current class, by multiplying the probablity of each independent attribute
            for i in attributes:

                # numerator is the count of intersection of target class and test element attribute
                numerator = (df[df[i] == test[i][k]].merge(df[df[target[0]] == int(j)])).shape[0] + 1
                
                # denominator is the count of the target class in the total dataset
                denominator = df[df[target[0]] == int(j)].shape[0] + len(classes)
                probability *= (numerator / denominator)

            # probablity of target class is also multiplied
            probability *= (df[df[target[0]] == int(j)].shape[0] / df.shape[0])
            
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

    prediction = predict(df, target, attributes, test)

    print("Predicted classes: ", prediction)
    accuracy = get_accuracy(prediction, test[target[0]])

    print("Accuracy: ", accuracy, "%")

if __name__=="__main__":
    main()
