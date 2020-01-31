"""
Author: Team 3B
Monash IDs: 29201128, 29201128, 29019834, 27467058
Last updated by: Yoveena Vencatasamy (29019834)
Last updated date: 30.01.2020
-------------------------------------------------------------
This python program further divides active users into potential clients and influencers by
performing sentiment analysis on the posts description and the comments. In order for an active user to
be a potential client, his posts should consists of mostly positive sentiments. If the user had already been
declared as potential influencer when determining active users, he has to pass 3 more criteria in order to become an
influencer. The additional criteria are as follows:
1) The sentiment score of the post has to be greater than the 75th percentile of all posts' sentiment scores combined.
2) The sentiment score of comments has to be greater than he 75th percentile of all comments' sentiment scores combined
3) The number of comments has to be greater than he 75th percentile of comments of each user combined
After filtering, 3 csv files are created 'potentialClients.csv', 'PotentialClientDetails.csv' and 'Influencers.csv'. Those
files are used as input to the database for the user interface.

Note:
- Compound Score is the combined score positive and negative sentiments. It has a value between -1 (most negative) to 1 (most positive)
  Therefore, neutral is zero.
"""

# nltk library
import nltk
import pandas as pd
import numpy as np

def findingQuantiles(hashtable):
    """
    This function creates a data frame for the quantiles (0.25, 0.50, 0.75) for the compound scores of comments distribution
    :param hashtable: hashtable contains author as key and a list of comments' sentiment scores as item
    :return quantileOfSentimentScore: data frame for the quantiles of compound scores of comments distribution
    """
    compound_score_list = []

    for author in hashtable:
        for item in hashtable[author]:
            compound_score_list.append(item)

    # finding quantiles for compound score range
    df2 = pd.DataFrame()
    df2['Compound_Score'] = compound_score_list
    quantileOfSentimentScore = df2[['Compound_Score']].quantile([0.25, 0.50, 0.75])

    return quantileOfSentimentScore

def obtain_sentiment(current_analyzer, current_comment):
    """
    This function returns the sentiments' score for a text
    :param current_analyzer: sentiment intensity analyser
    :param current_comment:
    :return current_sentiment: hashtable with 'neg', 'neu', 'pos' and 'compound' as key
    """
    current_sentiment = current_analyzer.polarity_scores(current_comment)
    return current_sentiment


def sentimentscoresComments(df, df_comments, hashtableClients):
    """
    This function returns a hashtable that for each key (author) has a list of comments' sentiment score
    :param df: data frame obtained from 'ActiveUserDetails.csv'
    :param df_comments: data frame obtained from the preprocessed comments' dataset
    :param hashtableClients: hashtable consists of only potential clients authors as key
    :return hashtableClients: hashtable that for each key (author) has a list of comments' sentiment score
    """
    #using vader for sentiment analysis
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    current_analyzer = SentimentIntensityAnalyzer()

    #get the post ids of potential clients only and update each hashtable item to an empty list
    post_id = []
    for ind in df.index:
        if df['author'][ind] in hashtableClients:
            post_id.append(df['id'][ind])
            hashtableClients[df['author'][ind]] = []

    #create a data frame for comments of post ids available
    comment_id_list = []
    for ind in df_comments.index:
        comment_id_list.append((df_comments['parent_id'][ind])[3:])
    df_comments['parent_id'] = comment_id_list
    df2 = df_comments[(df_comments['parent_id']).isin(post_id)]

    #update the hashtable by appending the sentiment score of each comment of an author
    for ind in df.index:
        for ind1 in df2.index:
            parent_id = df2['parent_id'][ind1]
            if parent_id == df['id'][ind]:
                score1 = obtain_sentiment(current_analyzer, df2['body'][ind1])
                scorecomment = score1['compound']
                hashtableClients[df['author'][ind]].append(scorecomment)
            else:
                pass
    #Each author has now his list of sentiment score of comments
    return hashtableClients


def sentimentscoresPost(df):
    """
    This function creates a hashtable with author as key and a list of sentiment scores of author's posts.
    The score is obtained by calling the 'obtain_sentiment' function on the post's description
    :param df: data frame obtained from 'ActiveUserDetails.csv'
    :return hashtable: hashtable with author as key and a list of sentiment scores of author's posts
    """
    # load the analyzer
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    current_analyzer = SentimentIntensityAnalyzer()

    hashtable = {}
    for ind in df.index:
        if df['author'][ind] not in hashtable:
            hashtable[df['author'][ind]] = []

    for ind in df.index:
        score = obtain_sentiment(current_analyzer, df['description'][ind])
        scorecompound = score['compound']
        hashtable[df['author'][ind]].append(scorecompound)

    return hashtable


def PositivityOfSentiments(alist, quantileDf, fieldName):
    """
    This function calculates the number of sentiment score that is less and greater than the benchmark. If number of
    sentiment scores that is greater > number of sentiment scores that is less, then the author has more positive
    comments
    :param alist: List of sentiment scores of comments of an author
    :param quantileDf: data frame giving the quantiles of all sentiment scores of comments combined
    :param fieldName: name given to the quantile data frame
    :return True or False: If True, it means that the author has mostly positive comments
    """
    upCount = 0
    downCount = 0
    for score in alist:
        if score >= quantileDf[fieldName][0.75]:
            upCount += 1
        else:
            downCount += 1

    if upCount >= downCount:
        return True
    else:
        return False


def influencer(hashtableClients, df_comments):
    """
    This function further filters through potential clients to get the influencers. It checks if the authors meet all
    the 5 criteria set to determine if they are influencers. It also creates a csv file 'Influencers.csv' that will be
    used as input for the database of the user interface.
    :param hashtableClients: hashtable consists of all users that are potential clients
    :param df_comments: data frame obtained from the preprocessed comments' dataset
    :return NIL:
    """
    df = pd.read_csv('RedditUserTypes/PotentialClientDetails.csv')

    postsScoreList = []
    postInfluencer = []
    nonPostInfluencer = []

    #update postsScoreList with sentiment score of all posts of all authors
    for author in hashtableClients:
        for score in hashtableClients[author]:
            postsScoreList.append(score)

    #get the 75th percentile of all posts' sentiment scores combined
    postsQuantile = np.percentile(postsScoreList, 75)

    #checking for first criterion mentioned above
    #Remove authors whose number of sentiment scores of post greater than 75th percentile < number sentiment scores of post less than 75th percentile
    for author in hashtableClients:
        if positivityOfText(hashtableClients[author], postsQuantile):
            postInfluencer.append(author)
        else:
            nonPostInfluencer.append(author)
    for author in nonPostInfluencer:
        del hashtableClients[author]

    #Updating hashtable with sentiment score of each comment of an author
    hashtable = sentimentscoresComments(df, df_comments, hashtableClients)
    quantileOfSentimentScore = findingQuantiles(hashtable)
    commentsQuantile = df[['num_comments']].quantile([0.25, 0.50, 0.75])
    influencers=[]

    #getting influencers by checking for the remaining 4 criteria
    for author in hashtable:
        activity = False
        NoOfCommentsList = []
        for ind in df.index:
            if df['author'][ind] == author:
                NoOfCommentsList.append(df['num_comments'][ind])
                activity = df['Influencer'][ind]
        compound_score_list = hashtable[author]
        positivity = PositivityOfSentiments(compound_score_list, quantileOfSentimentScore, 'Compound_Score')
        commentFlag = PositivityOfSentiments(NoOfCommentsList, commentsQuantile, 'num_comments')
        if (positivity and activity and commentFlag):
            influencers.append(author)

    #creating csv file for the user interface (Users who are Influencers)
    category = []
    brand = []
    userName = []
    for user in influencers:
        for ind in df.index:
            if user == df['author'][ind]:
                userName.append(user)
                brand.append(df['subreddit'][ind])
                category.append('Laptop')
    df1 = pd.DataFrame()
    df1['id'] = list(range(0, len(category)))
    df1['category_id'] = category
    df1['brand_id'] = brand
    df1['name'] = userName
    df1.to_csv('RedditUserTypes/Influencers.csv',index=False)


def positivityOfText(compound_score_list, value):
    """
    This function calculates the number of values in the list that is greater and less than the value given as parameter.
    It returns True or False based on which number is higher.
    :param compound_score_list: list of sentiment scores of posts for an author
    :param value:
    :return True or False: If True, it means that the author is a potential influencer.
    """
    upCount = 0
    downCount = 0
    for score in compound_score_list:
        if score > value:
            upCount += 1
        else:
            downCount += 1
    if upCount >= downCount:
        return True
    else:
        return False

def potentialClients(df):
    """
    This function filters the potential clients from active users. Potential clients are those whose posts have mostly
    compound score greater than 0.
    :param df: 'ActiveUserDetails.csv'
    :return potentialClientsList, hashtable: list consists of only potential clients, that is those that have positivity
    towards the subreddit product. Hashtable consists of only potential clients as key and list of sentiment sores of posts
    of the author
    """
    potentialClientsList = []
    nonPotentialClientsList = []
    hashtable = sentimentscoresPost(df)
    for author in hashtable:
        if positivityOfText(hashtable[author], 0):
            potentialClientsList.append(author)
        else:
            nonPotentialClientsList.append(author)

    for author in nonPotentialClientsList:
        del hashtable[author]
    return potentialClientsList,hashtable

def potentialClientsFile(df):
    """
    This function creates 2 csv files after obtaining potential clients, 'potentialClients.csv' and 'PotentialClientDetails.csv'.
    'PotentialClientDetails.csv' is a subset of 'ActiveUserDetails.csv' and is created so that it can be further filtered to
    obtain influencers.
    :param df: data frame obtained from 'ActiveUserDetails.csv'
    :return hashtable: Hashtable consists of only potential clients as key and list of sentiment sores of posts
    of the author
    """
    potentialClientsList, hashtable = potentialClients(df)
    category = []
    brand = []
    userName = []

    #creating csv file for the user interface (Users who are potential clients)
    for author in potentialClientsList:
        for ind in df.index:
            if author == df['author'][ind]:
                userName.append(author)
                brand.append(df['subreddit'][ind])
                category.append('Laptop')
    df1 = pd.DataFrame()
    df1['id'] = list(range(0, len(category)))
    df1['category_id'] = category
    df1['brand_id'] = brand
    df1['name'] = userName
    df1.to_csv('RedditUserTypes/potentialClients.csv', index=False)

    # creating csv file to be used to find influencers
    df2 = df[df['author'].isin(hashtable)]
    df2 = df2.drop(df2.columns[[0,1]], axis=1)
    df2.to_csv('RedditUserTypes/PotentialClientDetails.csv')

    return hashtable


def main():
   df = pd.read_csv('RedditUserTypes/ActiveUserDetails.csv')
   df_comments = pd.read_csv('Preprocessing/preprocessed_comments.csv')
   hashtable = potentialClientsFile(df)
   influencer(hashtable, df_comments)

