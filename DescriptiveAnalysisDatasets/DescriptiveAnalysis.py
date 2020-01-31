"""
Author: Team 3B
Monash IDs: 29201128, 29201128, 29019834, 27467058
Last updated by: Yoveena Vencatasamy (29019834)
Last updated date: 30.01.2020
-------------------------------------------------------------
This python program does descriptive analysis on raw, preprocess datasets and analyses active
and inactive users' behvaiour. It describes the number of post, comments and number of unique
authors left after each phase (raw data -> Preprocessing -> Active and Inactive users -> Potential clients -> Influencers
"""
import pandas as pd

def groupByMonth(df):
    """
    This function processes the given dataframe such that data is arranged by month, for each month,
    the average number of posts per user and the average number of comments per post is calculated. A
    new dataframe is created
    :param df: the source file is converted to data frame and then passed as parameter
    :return df2: A new data frame with data grouped by month
    """
    hashtable = {}
    for i in range(1, 13):
        hashtable[i] = [0, 0, []]

    for ind in df.index:
        if df['author'][ind] not in hashtable[df['month'][ind]][2]:
            hashtable[df['month'][ind]][2].append(df['author'][ind])
        hashtable[df['month'][ind]][0] += 1
        hashtable[df['month'][ind]][1] += df['num_comments'][ind]

    month=[]
    NoOfPost =[]
    NoofComments =[]
    for item in hashtable:
        if hashtable[item][0]!= 0:
            month.append(item)
            NoOfPost.append(hashtable[item][0]/len(hashtable[item][2]))
            NoofComments.append(hashtable[item][1]/hashtable[item][0])
    df2 = pd.DataFrame()
    df2['Month'] = month
    df2['No_of_Posts'] = NoOfPost
    df2['No_of_Comments'] = NoofComments
    return df2

def getstatistics(df,df2,df3,df4,df5,df6):
    """
    This function get the statistical values such as number of posts, number of comment, number
    of unque authors in a dataset
    :param df: data frame created from the raw dataset (unprocessed)
    :param df2: data frame created from preprocessed dell posts
    :param df3: data frame created from the active user details csv file
    :param df4: data frame created from the active user details csv file
    :param df5: data frame created from the potential clients csv file
    :param df6: data frame created from the influencers csv file
    :return: NIL
    """
    print("----------------------Before preprocessing----------------------")
    print("No of posts: " + str(df['created_utc'].count()))
    print("No of unique authors: " + str(df['author'].nunique()))
    print("Total number of comments: " + str(df['num_comments'].sum()))
    print("----------------------After preprocessing----------------------")
    print("No of posts: " + str(df2['created_utc'].count()))
    print("No of unique authors: " + str(df2['author'].nunique()))
    print("Total number of comments: " + str(df2['num_comments'].sum()))
    print("----------------------Active and Inactive Users----------------------")
    print("No of unique active authors: " + str(df3['author'].nunique()))
    print("No of unique inactive authors: " + str(df4['author'].nunique()))
    print("----------------------Potential Clients----------------------")
    print("No of unique authors: " + str(df5['author'].nunique()))
    print("----------------------Influencers----------------------")
    print("No of unique authors: " + str(df6['name'].nunique()))



def main():
    df = pd.read_csv('../RedditUserTypes/ActiveUserDetails.csv')
    df2 = pd.read_csv('../RedditUserTypes/InactiveUserDetails.csv')
    df_1 = groupByMonth(df)
    df_1.to_csv("ActiveUserMonthDetails.csv")
    df_2= groupByMonth(df2)
    df_2.to_csv("InactiveUserMonthDetails.csv")
    dfs1 = pd.read_csv("../SourceFiles/Dell.csv")
    dfs2= pd.read_csv("../Preprocessing/preprocessed_posts.csv")
    dfs3 = pd.read_csv ("../RedditUserTypes/ActiveUserDetails.csv")
    dfs4 = pd.read_csv("../RedditUserTypes/InactiveUserDetails.csv")
    dfs5 = pd.read_csv("../RedditUserTypes/PotentialClientDetails.csv")
    dfs6 = pd.read_csv("../RedditUserTypes/Influencers.csv")
    getstatistics(dfs1,dfs2,dfs3,dfs4,dfs5,dfs6)

if __name__ == '__main__':
    main()
