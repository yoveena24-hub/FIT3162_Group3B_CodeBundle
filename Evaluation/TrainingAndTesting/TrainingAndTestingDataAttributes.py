"""
Author: Team 3B
Monash IDs: 29201128, 29201128, 29019834, 27467058
Last updated by: Yoveena Vencatasamy (29019834)
Last updated date: 30.01.2020
-------------------------------------------------------------
This python program creates the training and testing dataset. From the training and testing dataset,
attributes to be fed in machine learning models are created and written to a csv file. The attributes
are as follows: ratio of gaps of author of post, number of gaps of author, sentiment score of post, sentiment
ratio of comments, score of post, score ratio of comments.

Note:
- A gap is the interval of time between two consecutive post of a user (one gap == two posts, time interval between the 2 posts)
- Each score value is classified as 'neutral', 'Low' or 'High'
"""
import pandas as pd
import sys
sys.path.insert(0, '../../RedditUserTypes')

import ActiveInactiveUsers
import SentimentAnalysisOfActiveUsers

import numpy as np

def createnewdf(df,type):
    """
    This function creates a dataframe for all attributes required to feed in machine learning
    models. Depending on the data frame input, it will either create csv file for training or
    testing data attributes.
    :param df: dataframe can either be testing or training data
    :param type: train/test based on the file that needs attributes to be created. Since file names
    for the attributes need to be different for training and testing datasets.
    :return: NIL
    """
    df2 = df[['created_utc', 'id', 'author', 'num_comments', 'month']]
    gapRatioList, noOfGaps = findingActivityRatios(df)  # Author: [ratio of gaps, no of gaps]
    postsSentimentScoreList, postIdList = sentimentAnalysisOfPosts(df)
    commentSentimentScoreList, classCommentScore = sentimentAnalysisOfComments(df, postIdList)
    postScoreList = classOfScore(df)

    df2 = df2.assign(Ratio_Of_Gaps=gapRatioList)
    df2 = df2.assign(Number_of_Gaps=noOfGaps)
    df2 = df2.assign(Sentiment_Of_Post=postsSentimentScoreList)
    df2 = df2.assign(Sentiment_Of_Comments=commentSentimentScoreList)
    df2 = df2.assign(Score_Of_Comments=classCommentScore)
    df2 = df2.assign(Score=postScoreList)
    if type == "train":
        df2.to_csv("TrainingDataAttributes.csv")
    else:
        df2.to_csv("TestingDataAttributes.csv")

def classOfScore(df):
    """
    This function classifies a score as 'neutral', 'Low' or 'High'. It analyses the scores of all
    posts in the data frame and then sets the 90th percentile to create the classifiers
    :param df: data frame with a 'score' column
    :return classOfScoreList: returns the list of scores after classification
    """
    scoreList = []
    for ind in df.index:
        if df['score'][ind] != 0:
            scoreList.append(df['score'][ind])
    percentile90th = np.percentile(scoreList, 90)

    classOfScoreList = []
    for ind in df.index:
        if df['score'][ind] == 0:
            classOfScoreList.append('neutral')
        elif df['score'][ind] >= percentile90th:
            classOfScoreList.append('High')
        else:
            classOfScoreList.append('Low')
    return classOfScoreList

def RatioOfActivity(list, value):
    """
    This function gives the ratio of gaps based on a value set. For eg, if you have a list of gaps
    for a user. It calculates the number of values above and below the value set. The ratio is then
    given by (number of values below/number of values above). This is so because the smaller the gap
    the more active the author.
    :param list: List of gaps for an author
    :param value: value is the benchmark set to determine active
    :return upCount / downCount: ratio of active gaps against inactive gaps
    """
    upCount = 0
    downCount = 0
    for item in list:
        if item < value:
            upCount += 1
        else:
            downCount += 1
    if downCount == 0:
        return upCount
    return (upCount / downCount)

def positivity(list, value):
    """
    This function determines whether a sentiment score is positive. For eg if you have a list of sentiment scores, if
    item in list is greater than benchmark (value) set, then it is considered as positive.
    :param list: list of sentiment scores of a user
    :param value: benchmark set for sentiment score to be positive
    :return upCount / downCount:  ratio of positivity against negativity
    """
    upCount = 0
    downCount = 0
    for item in list:
        if item > value:
            upCount += 1
        else:
            downCount += 1
    if downCount == 0:
        return upCount
    return (upCount / downCount)


def findingActivityRatios(df):
    """
    This function creates 2 lists. One lists has the gap ratio of the author for each post and the other
    list contains the number of gaps of the author for each post. Those are the attributes that will be
    written to the csv file.
    :param df: data frame can either be training data or testing data
    :return gapRatioList, noOfGaps:
    """
    hashtable, inactive_list, quantileOfGaps, quantileOfNoOfGaps = ActiveInactiveUsers.findingquantiles(df)

    for author in hashtable:
        length = len(hashtable[author])
        gapList = hashtable[author]
        tuple = [RatioOfActivity(gapList, quantileOfGaps['Gap'][0.50]), length]
        hashtable[author] = tuple

    gapRatioList = []
    noOfGaps = []
    for ind in df.index:
        gapRatioList.append((hashtable[df['author'][ind]])[0])
        noOfGaps.append((hashtable[df['author'][ind]])[1])

    return gapRatioList, noOfGaps


def checkIfDuplicates(alist):
    """
    This function is simple and just checks whether a list has duplicates. This is achieved by using
    the set function
    :param alist:
    :return True or False: If the list in parameter has duplicates, it returns true, else returns false
    """
    if len(alist) == len(set(alist)):
        return False
    else:
        return True

def sentimentAnalysisOfPosts(df):
    """
    This function gets the compound score of each post in a data frame and creates 2 lists. Those will be used
    as attributes for the csv file.
    :param df: Data frame that has the column 'description' for the post description
    :return post_sentiment_score, post_id: The first list contains the sentiment score of each post and the
    second list contains each post id
    """

    #using vader from nltk library
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    current_analyzer = SentimentIntensityAnalyzer()

    post_id = []
    post_sentiment_score = []
    for ind in df.index:
        post_id.append(df['id'][ind])
        # use sentiment analyser function from imported file
        score = SentimentAnalysisOfActiveUsers.obtain_sentiment(current_analyzer, str(df['description'][ind]))
        post_sentiment_score.append(score['compound'])
    return post_sentiment_score, post_id

def getChange(i, j, id):
    """
    This function returns the index of the value that should be changed in a list
    :param i:
    :param j:
    :param id:
    :return 1 or 2:
    """
    if i == 0:
        if j == 1:
            return 1
        elif j == 2:
            return 2
    if i == 1:
        if j == 2:
            return 2

def changeList(id, alist):
    """
    This function changes duplicate values of a list to another value
    :param id:
    :param alist: The list that contains duplicates
    :return alist: The list as parameter has been modified
    """
    for i in range(len(alist)-1):
        for j in range(i+1, len(alist)):
            if alist[i] != 0 and alist[i] == alist[j]:
                changeIndex = getChange(i, j, id)
                alist[i] = 0
                alist[j] = 0
                alist[changeIndex] += 1
    return alist


def getClass(id, scoreList, quantileList):
    """
    This function categorises a list of sentiment scores of comments to a post to one of the following classifiers:
    'neutral', 'positive' and 'highly_positive'. This is achieved by first comparing each value to a benchmark obtained
    from the quantile distribution of all the sentiment scores of all comments in the data frame.
    :param id: The id of the post to the comments
    :param scoreList: A list of sentiment scores of all the comments to a post
    :param quantileList: The values for the 25th, 50th and 90th percentile in the sentiment score distribution of comments
    :return 'neutral', 'positive' and 'highly_positive': overall classifier for the comments
    """
    # list has counts as - neutral, positive, highly positive
    countList = [0, 0, 0]
    for score in scoreList:
        if score == 0:
            # neutral
            countList[0] += 1
        elif quantileList[1] <= score < quantileList[2]:
            # positive
            countList[1] += 1
        elif score >= quantileList[2]:
            countList[2] += 1

    if checkIfDuplicates(countList):
        countList = changeList(id, countList)

    # get the ratios
    total = sum(countList)
    ratio = 0
    placeValue = -1
    if total != 0:
        for count in countList:
            if ratio < count/total:
                ratio = count/total
                placeValue = countList.index(count)
    #classify based on the ratios
        if placeValue == 0:
            return "neutral"
        if placeValue == 1:
            return "positive"
        if placeValue == 2:
            return "highly_positive"
    else:
        return "neutral"


def sentimentAnalysisOfComments(df, post_id):
    """
    This function analyses each comment to a post; the factors that it takes into consideration are
    the score and sentiment score of a comment. For each post it gets the classifier for the score and sentiment
    score of its comments. It creates 2 lists; the first list for sentiment of comments and the second list for
    score of comments. Those are used as attributes for the csv file.
    :param df: data frame obtained from preprocessed dataset of posts
    :param post_id: List of post ids in data frame
    :return ratio_comment_sentiment_score, classCommentScore: list of for sentiment of comments and score of
    comments after classification
    """
    df_comments = pd.read_csv("../Preprocessing/preprocessed_comments.csv")

    #using vader library for sentiment analysis
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    current_analyzer = SentimentIntensityAnalyzer()

    postHashtable = {}
    commentScoreTable = {}

    #creating hashtables with post id as key
    for ind in df.index:
        postHashtable[df['id'][ind]] = []    #hastable for sentiment score of comments
        commentScoreTable[df['id'][ind]] = []  #hastable for score of comments

    #creating a new data frame for only the comments to available post ids
    comment_id_list = []
    for ind in df_comments.index:
        comment_id_list.append((df_comments['parent_id'][ind])[3:])
    df_comments['parent_id'] = comment_id_list
    df2 = df_comments[(df_comments['parent_id']).isin(post_id)]

    for ind in df.index:
        for ind1 in df2.index:
            parent_id = df2['parent_id'][ind1]
            if parent_id == df['id'][ind]:
                score1 = SentimentAnalysisOfActiveUsers.obtain_sentiment(current_analyzer, str(df2['body'][ind1]))
                scorecomment = score1['compound']
                # getting the sentiment score of the comments to each post
                postHashtable[df['id'][ind]].append(scorecomment)
                # getting the score of the comments to each post
                commentScoreTable[df['id'][ind]].append(df2['score'][ind1])

    #creating a list for the scores only
    commentScoreList = []
    for id in commentScoreTable:
        for score in commentScoreTable[id]:
            if score != 0:
                commentScoreList.append(score)

    #getting quantiles for scores of comments distribution
    quantileList = []
    quantileList.append(np.percentile(commentScoreList, 25))
    quantileList.append(np.percentile(commentScoreList, 50))
    quantileList.append(np.percentile(commentScoreList, 90))

    #classifying scores of comments to each post
    for id in commentScoreTable:
        commentScoreTable[id] = getClass(id, commentScoreTable[id], quantileList)

    # classifying sentiment scores of comments to each post
    for id in postHashtable:
        postHashtable[id] = positivity(postHashtable[id], 0)

    #creating the list of for ratio of comments and score ratio of comments.
    ratio_comment_sentiment_score = []
    classCommentScore = []
    for id in post_id:
        ratio_comment_sentiment_score.append(postHashtable[id])
        classCommentScore.append(commentScoreTable[id])

    return ratio_comment_sentiment_score, classCommentScore

def last3monthActiveUsers(df):
    """
    This function returns a list of authors that were active in the last 3 months (10,11,12)
    :param df: data frame obtained from preprocessed dataset
    :return last3monthUsers: list consisting of authors active in last 3 months
    """
    last3monthUsers = []
    for ind in df.index:
        if (12 >= df['month'][ind] >= 10) and (df['author'][ind] not in last3monthUsers):
            last3monthUsers.append(df['author'][ind])
    return last3monthUsers


def first9monthUsers(last3monthUsers, df):
    """
    This function creates the training dataset by taking into account only the authors that were active
    in the last 3 months. This is done so that the accuracy of testing data can be improved. It then creates
    two csv files from the two data frames created
    :param last3monthUsers:
    :param df: data frame obtained from preprocessed dataset
    :return: NIL
    """
    df2 = df[df['author'].isin(last3monthUsers) & (df['month'] >= 1) & (df['month'] <= 9)]
    df3 = df[df['author'].isin(last3monthUsers) & (df['month'] >= 10) & (df['month'] <= 12)]
    df2.to_csv("trainingData.csv")
    df3.to_csv("testingData.csv")


def main():
    df = pd.read_csv('../Preprocessing/preprocessed_posts.csv')
    first9monthUsers(last3monthActiveUsers(df), df)
    trainingDf = pd.read_csv("trainingData.csv")
    testingDf = pd.read_csv("testingData.csv")
    createnewdf(trainingDf,"train")
    createnewdf(testingDf, "test")

if __name__ == '__main__':
    main()
