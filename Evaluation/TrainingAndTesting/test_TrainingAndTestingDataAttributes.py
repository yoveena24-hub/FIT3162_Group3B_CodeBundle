import pytest
import pandas as pd

import sys
sys.path.insert(0, '../../')

from Evaluation.TrainingAndTesting import TrainingAndTestingDataAttributes

import numpy as np

def testclassOfscore():
    data = {'score': [5, 7, 10,19,20]}
    df = pd.DataFrame(data)
    assert TrainingAndTestingDataAttributes.classOfScore(df) == ['Low','Low','Low','Low','High']

def testratioofactivity():
    List = [5,6,6,6,2,3,2,1,4]
    assert TrainingAndTestingDataAttributes.RatioOfActivity(List,5) == (5/4)

def testcheckifduplicates():
    List1 = [3, 4, 1, 4, 5]
    List2 = [1,2,9,6,7]
    assert TrainingAndTestingDataAttributes.checkIfDuplicates(List1) == True, "Check duplicates test failed"
    assert TrainingAndTestingDataAttributes.checkIfDuplicates(List2) == False, "Check duplicates test failed"

def testgetchange():
    assert TrainingAndTestingDataAttributes.getChange(0,2,'5984') == 2, "Get change test failed"
    assert TrainingAndTestingDataAttributes.getChange(1, 2, '5984') == 2, "Get change test failed"

def testchangelist():
    assert TrainingAndTestingDataAttributes.changeList('590',[1,1,2]) == [0,1,2], "Change list test failed"
    assert TrainingAndTestingDataAttributes.changeList('590', [2, 1, 2]) == [0, 0, 1], "Change list test failed"
    assert TrainingAndTestingDataAttributes.changeList('590', [2, 2, 2]) == [0, 1, 2], "Change list test failed"

def testgetclass():
    scoreList = [0,100,50,25,100,200,20,10,55,13]
    userscore1 = [10,39,60]
    userscore2 = [0,10,13]
    quantileList =[]
    quantileList.append(np.percentile(scoreList, 25))
    quantileList.append(np.percentile(scoreList, 50))
    quantileList.append(np.percentile(scoreList, 90))
    assert TrainingAndTestingDataAttributes.getClass('590',userscore1,quantileList) == "positive"
    assert TrainingAndTestingDataAttributes.getClass('590', userscore2, quantileList) == "neutral"

