from sklearn.preprocessing import StandardScaler
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report

import pandas as pd

def retrieveData():
    trainingData = pd.read_csv("training-data.csv", header = 0).as_matrix()
    testData = pd.read_csv("test-data.csv", header = 0).as_matrix()

    return trainingData, testData

def separateFeaturesAndCategories(trainingData, testData):
  trainingFeatures = trainingData[:, :-1]
  trainingCategories = trainingData[:, -1:]
  testFeatures = testData[:, :-1]
  testCategories = testData[:, -1:]

  return trainingFeatures, trainingCategories, testFeatures, testCategories

def scaleData(trainingFeatures, testFeatures):
    scaler = StandardScaler()
    scaler.fit(trainingFeatures)

    scaledTrainingFeatures = scaler.transform(trainingFeatures)
    scaledTestFeatures = scaler.transform(testFeatures)

    return scaledTrainingFeatures, scaledTestFeatures

def classifyTestSamples(trainingFeatures, trainingCategories, testFeatures):
    clf = SGDClassifier()

    clf.fit(trainingFeatures, trainingCategories)
    predictedCategories = clf.predict(testFeatures)

    return predictedCategories

def gatherClassificationMetrics(testCategories, predictedCategories):
    accuracy = accuracy_score(testCategories, predictedCategories)
    metrics_report = classification_report(testCategories, predictedCategories)

    print("Accuracy rate: " + str(round(accuracy, 2)) + "\n")
    print(metrics_report)

def main():
    trainingData, testData = retrieveData()

    trainingFeatures, trainingCategories, testFeatures, testCategories = separateFeaturesAndCategories(trainingData, testData)

    scaledTrainingFeatures, scaledTestingFeatures = scaleData(trainingFeatures, testFeatures)

    predictedCategories = classifyTestSamples(trainingFeatures, trainingCategories, testFeatures)

    gatherClassificationMetrics(testCategories, predictedCategories)

if __name__ == "__main__": main()
