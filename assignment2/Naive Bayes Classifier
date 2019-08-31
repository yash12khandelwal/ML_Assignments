# 17CH10051
# Yash Khandelwal
# Assignment Number 2
# python3 <program name>

import pandas as pd
import numpy as np

def prepare_dataframe(df):
    temp = []
    for j in range(len(df)):
        string = ""
        for i in range(len(df.iloc[j,:])):
            string += str(df.iloc[j,i])
        temp.append([int(char) for char in string.split(',')])

    df = pd.DataFrame(temp, columns = df.columns.values[0].split(","))

    return df

def predict(df, target, attributes, test):
    classes = list(set(df[target[0]]))
    temp = []
    for k in range(len(test)):

        prev_probability = 0
        
        for j in classes:
            probability = 1
        
            for i in attributes:

                numerator = (df[df[i] == test[i][k]].merge(df[df[target[0]] == int(j)])).shape[0] + 1
                denominator = df[df[target[0]] == int(j)].shape[0] + len(classes)
                probability *= (numerator / denominator)

            probability *= (df[df[target[0]] == int(j)].shape[0] / df.shape[0])
            if probability > prev_probability:
                
                prev_probability = probability
                label = j

        temp.append(label)

    return temp

def get_accuracy(prediction, labels):
    count = 0

    for i in range(len(labels)):
        if prediction[i] == labels[i]:
            count += 1

    accuracy = (count / len(labels)) * 100

    return accuracy

def main():
    df = pd.read_csv("data2_19.csv")
    df = prepare_dataframe(df)

    test = pd.read_csv("test2_19.csv")
    test = prepare_dataframe(test)

    target = df.columns.values[0]
    attributes = df.columns.values[1:]

    prediction = predict(df, target, attributes, test)

    print("Predicted classes: ", prediction)
    accuracy = get_accuracy(prediction, test[target[0]])

    print("Accuracy: ", accuracy, "%")

if __name__=="__main__":
    main()
