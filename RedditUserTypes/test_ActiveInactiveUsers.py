import pytest
import pandas as pd
import ActiveInactiveUsers
import SentimentAnalysisOfActiveUsers

def testactivitybasedongaps():
    quantiles = [0.25, 0.5, 0.75]
    data = {'Gap': [5, 7, 10]}
    df = pd.DataFrame(data,index = quantiles)
    list = [4,5,5,9]
    assert ActiveInactiveUsers.activityBasedOnGaps(list,df) == True, "Activity based on gaps failed"

def testgetinfluencers():
    quantiles = [0.25, 0.5, 0.75]
    data = {'Gap': [5, 7, 10]}
    data2= {'No_Of_Gaps': [2,5,6]}
    df = pd.DataFrame(data, index=quantiles)
    df2 = pd.DataFrame(data2,index=quantiles)
    hashtable = {'ivan': [4,4,5]}
    hashtable1 = ActiveInactiveUsers.getInfluencers(hashtable,df,df2)
    assert hashtable1['ivan'] == False, "Get influencers test failed"

def testfindinggaps():
    data = {'author': ['Ivan','Ivan','Ivan'],'year':[2018,2018,2018],'month':[1,5,6],'day':[1,1,30]}
    df = pd.DataFrame(data)
    hashtable = ActiveInactiveUsers.findingGaps(df)
    assert hashtable['Ivan'] == [120,60], "Finding gaps test failed"

def testfindingquantiles():
    hashtable = {'ivan': [5,4,3,2,2], 'Loic' : [2,4,5,5] }
    quantiledf = SentimentAnalysisOfActiveUsers.findingQuantiles(hashtable)
    assert quantiledf['Compound_Score'][0.50] == 4, "finding quantiles test failed"
    assert quantiledf['Compound_Score'][0.75] == 5, "finding quantiles test failed"

def testpositivityofsentiment():
    hashtable = {'ivan': [0.909, 0.706, -0.5, -0.1, 0.8],'Loic':[0.509, 0.306, -0.5, 0]}
    quantiledf = SentimentAnalysisOfActiveUsers.findingQuantiles(hashtable)
    assert (SentimentAnalysisOfActiveUsers.PositivityOfSentiments(hashtable['ivan'],quantiledf,'Compound_Score')) == True, "Positivity of sentiments test failed"

def testpositityoftext():
    hashtable = {'ivan': [0.909, 0.706, -0.5, -0.1, 0.8], 'Loic': [-0.509, -0.306, -0.5, 0]}
    assert (SentimentAnalysisOfActiveUsers.positivityOfText(hashtable['ivan'],0)) == True,"Positivity of text test failed"
    assert (SentimentAnalysisOfActiveUsers.positivityOfText(hashtable['Loic'],0)) == False, "Positivity of text test failed"

def testpotentialclients():
    data = {'author':['Ivan', 'Loic'],'description':['I am very happy about my dell laptop','I am very very disappointed']}
    df = pd.DataFrame(data)
    potentialClientsList, hashtable = SentimentAnalysisOfActiveUsers.potentialClients(df)
    assert len(potentialClientsList)== 1, "Potential clients test failed"

