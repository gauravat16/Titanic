#!/usr/bin/python

import os


# ../dataset/train.csv

def getDataFromCSV(path):
    testData=open(path,"r").readlines();

    data = []

    for line in testData:
        if(not line.__contains__("PassengerId")):
            data.append([x.strip() for x in line.split(',')])

    return data


def getIndexArray(index,path):
    data = getDataFromCSV(path)

    returnArr =[]

    for arr in data:
        returnArr.append(arr[index])

    return returnArr


