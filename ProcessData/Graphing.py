#!/usr/bin/python

import matplotlib.pyplot as plt
import ReadCSV
def showGraph():



    x = formatAge()
    y = formatGender()

    print x
    print y
    plt.plot(x,y,"o")
    plt.show()


def formatGender(path):
    data = []
    if path=="../dataset/test.csv":
        csvData= ReadCSV.getIndexArray(4,path)
    else:
        csvData= ReadCSV.getIndexArray(5,path)



    for val in csvData:
        if (val == "male"):
            data.append(0)
        else:
            data.append(1)
    return data


def formatSiblings(path):
    data = []
    if path=="../dataset/test.csv":
        csvData= ReadCSV.getIndexArray(7,path)
    else:
        csvData= ReadCSV.getIndexArray(6,path)



    for val in csvData:
           data.append(int(val))
    return data


def formatClass(path):
    data = []
    if path=="../dataset/test.csv":
        csvData= ReadCSV.getIndexArray(1,path)
    else:
        csvData= ReadCSV.getIndexArray(2,path)



    for val in csvData:
        data.append(int(val))
    return data


def formatAge(path):
    data = []

    if path == "../dataset/test.csv":
        csvData = ReadCSV.getIndexArray(5, path)
    else:
        csvData = ReadCSV.getIndexArray(6, path)

    for val in csvData:
        if (val!=""):
            data.append(float(val))
        else:
            data.append(0)


    return data

def formatSurvival(path):
    data = []
    for val in ReadCSV.getIndexArray(1,path):
        if (val == "0"):
            data.append(0)
        else:
            data.append(1)
    return data

def getPids(path):

    data = []
    for val in ReadCSV.getIndexArray(0,path):
        data.append(val)
    return data




def add2Features(feature1,feature2):
    combined =[]
    for index in range(0,len(feature1)):

        combined.append([feature1[index],feature2[index]])

    return combined


def add3Features(feature1,feature2,feature3):
    combined =[]
    for index in range(0,len(feature1)):

        combined.append([feature1[index],feature2[index],feature3[index]])

    return combined


def add4Features(feature1,feature2,feature3,f4):
    combined =[]
    for index in range(0,len(feature1)):

        combined.append([feature1[index],feature2[index],feature3[index], f4[index]])

    return combined



def getOPCSV(data):
    combined =[]
    combined = add2Features(getPids("../dataset/test.csv"),data)
    combined.insert(0,["PassengerId","Survived"])

    import csv
    with open('result.csv', 'wb') as f:
        wtr = csv.writer(f, delimiter=',')
        wtr.writerows(combined)


def getPrunedFeatures(mode):
    test_path="../dataset/test.csv"
    train_path="../dataset/train.csv"

    if (mode==1):
        age=ReadCSV.getIndexArray(6,train_path)
        pClass=ReadCSV.getIndexArray(2,train_path)
        gender=formatGender(train_path)
        survival=ReadCSV.getIndexArray(1,train_path)
        siblings=ReadCSV.getIndexArray(7,train_path)

    else:
        age = ReadCSV.getIndexArray(5, test_path)
        pClass = ReadCSV.getIndexArray(1, test_path)
        gender = formatGender(test_path)
        survival = ReadCSV.getIndexArray(1,train_path)
        siblings=ReadCSV.getIndexArray(6,test_path)


    # print age
    # print pClass
    # print gender
    # print survival
    # print siblings


    combinedFeatures = []
    newLables = []
    for index in range(0, len(age)):
        if(age[index]!="" and float(age[index]) != 0):
            combinedFeatures.append([float(age[index]), int(pClass[index]), gender[index], int(siblings[index])])
            newLables.append(survival[index])

    return [combinedFeatures,newLables]

