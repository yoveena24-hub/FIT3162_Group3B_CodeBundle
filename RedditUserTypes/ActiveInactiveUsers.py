"""
Author: Team 3B
Monash IDs: 29201128, 29201128, 29019834, 27467058
Last updated by: Yoveena Vencatasamy (29019834)
Last updated date: 30.01.2020
-------------------------------------------------------------
This python program divides the preprocessed dataset for posts into active and inactive users. It creates two csv files where
'ActiveUserDetails.csv' will contain posts of active users and 'InactiveUserDetails.csv' will contain posts of inactive users.
Active users are defined based on two criteria; the first criteria is that the number of gaps, which is directly proportional to
the number of post exceeds the 25th percentile of the number of gaps for each user combined, the second criteria is that the gap
should be less than the 75th percentile of all users' gaps' value combined. The second criteria is set because if the gap is less for a user, it
means that he/she posts more frequently. If a user meets those 2 criteria, he is considered as active user, else he is inactive.

Note:
- A gap is the interval of time between two consecutive post of a user (one gap == two posts, time interval between the 2 posts)
"""
import pandas as pd
from datetime import date

def activityBasedOnGaps(gapList, df):
    """
    This function calculates the number of gaps that is less and greater than the benchmark. The ratio is then given by
    (number of gaps that is less/number of gaps that is greater). If the ratio is greater than one, it means that the author
    passed the first criteria
    :param gapList: List of gaps for each user, intervals between each post
    :param df: data frame giving the quantiles of the combined intervals obtained from each user
    :return True or False: if true, first criteria for determining active user is passed
    """
    upCount = 0
    downCount = 0
    for gap in gapList:
        if gap < df['Gap'][0.75]:
            upCount+=1
        else:
            downCount+=1
    if upCount >= downCount:
        return True
    else:
        return False

def activityBasedOnNoOfGaps(number, df):
    """
    This function determines if the number of posts/gaps of a user is greater than the 25th percentile of
    all posts for each user combined
    :param number: This parameter is the number of gaps for a user
    :param df: data frame giving the quantiles of the combined number of gaps obtained from each user
    :return True or False: If True, it means that the user meets the 2nd criterion for being active
    """
    if number > df['No_Of_Gaps'][0.25]:
        return True
    else:
        return False


def getInfluencers(hashtable,df,df2):
    """
    This function is to determine whether an active user is a potential influencer. This criteria are set higher.
    For a user to be a potential influencer, the number of post must exceed the 75th percentile of the number of gaps for each user combined and
    the second criteria being the gap should be less than the 25th percentile of all users' gaps' value combined.
    :param hashtable: The hashtable consists of author as key and gap list of author as item
    :param df:  data frame giving the quantiles of the combined intervals obtained from each user
    :param df2: data frame giving the quantiles of the combined number of gaps obtained from each user
    :return hashtable: hashtable that consists of author as key and True or False as item. True means that author is a
    potential influencer
    """
    RatioOfGaps = False
    NoOfGaps = False
    for author in hashtable:
        upCount = 0
        downCount = 0
        for gap in hashtable[author]:
            if gap < df['Gap'][0.25]:
                upCount += 1
            else:
                downCount += 1
        if upCount >= downCount:
            RatioOfGaps = True

        if len(hashtable[author]) > df2['No_Of_Gaps'][0.75]:
            NoOfGaps = True

        hashtable[author] = RatioOfGaps and NoOfGaps
    return hashtable


def findingGaps(df):
    """
    This function creates a hashtable with each unique author as key and list of gaps as item.
    :param df: data frame obtained from the preprocessed posts' dataset
    :return hashtable: hashtable with each unique author as key and list of gaps as item
    """
    #create hashtable with author as key and list of dates of posts as item
    hashtable = {}
    for ind in df.index:
        if df['author'][ind] not in hashtable:
            hashtable[df['author'][ind]] = []
    for ind in df.index:
        hashtable[df['author'][ind]].append([df['year'][ind], df['month'][ind], df['day'][ind]])

    for i in hashtable:
        for j in range(len(hashtable[i])):
            if j+1 < len(hashtable[i]):
                user = hashtable[i]
                f_date = date(user[j][0], user[j][1], user[j][2])
                l_date = date(user[j+1][0], user[j+1][1], user[j+1][2])
                delta = l_date - f_date    #calculated interval for two dates
                user[j] = abs(delta.days)
            else:
                del hashtable[i][-1]
    return hashtable

def findingquantiles(df):
    """
    This function creates 2 data frames. The first data frame contains quantiles of the combined intervals obtained
    from each user. The second data frame consists of quantiles of the combined number of gaps obtained from each user.
    It also updates the hashtable by removing the users that have no gaps, that is o posts; thus creating a list
    of inactive users.
    :param df: data frame obtained from the preprocessed posts' dataset
    :return hashtable,inactive_list,quantileOfGaps,quantileOfNoOfGaps: updated hastable with authors that have posted at least
    once, list of users that have never posted, uantiles of the combined intervals obtained from each user, quantiles of the
    combined number of gaps obtained from each user
    """
    hashtable = findingGaps(df)
    user_list = []
    gap_list = []
    inactive_list=[]
    unique_author_list = []
    NoOfGaps = []

    #filtering authors by active and inactive if gap is 0 (posted only once)
    for author in hashtable:
        if len(hashtable[author]) != 0:
            unique_author_list.append(author)
            NoOfGaps.append(len(hashtable[author]))
            for item in hashtable[author]:
                user_list.append(author)
                gap_list.append(item)
        else:
           inactive_list.append(author)

    #finding quantiles for gaps range
    df2 = pd.DataFrame()
    df2['User'] = user_list
    df2['Gap'] = gap_list
    quantileOfGaps = df2[['Gap']].quantile([0.25, 0.50, 0.75])

    #finding quantiles for range of number of gaps
    df3 = pd.DataFrame()
    df3['User'] = unique_author_list
    df3['No_Of_Gaps'] = NoOfGaps
    quantileOfNoOfGaps = df3[['No_Of_Gaps']].quantile([0.25, 0.50, 0.75])

    return hashtable,inactive_list,quantileOfGaps,quantileOfNoOfGaps

def filteringActive(df):
    """
    This function checks for each author if he meets the 2 criteria set for determining an active user. If the user is not active,
    it is appended to the inactive users' list and deleted from the hashtable later on. Thus the hashtable will consist of
    only active authors/users.
    :param df: data frame obtained from the preprocessed posts' dataset
    :return hashtable1: The hastable consists of only active users as key
    """
    hashtable, inactive_list, quantileOfGaps, quantileOfNoOfGaps = findingquantiles(df)

    #filtering active and inactive users based on quantiles
    for author in hashtable:
        length = len(hashtable[author])
        gapList = hashtable[author]
        if not (activityBasedOnGaps(gapList, quantileOfGaps) and activityBasedOnNoOfGaps(length, quantileOfNoOfGaps)):
            if author not in inactive_list:
                inactive_list.append(author)
    for user in inactive_list:
        del hashtable[user]
    hashtable1 = getInfluencers(hashtable,quantileOfGaps,quantileOfNoOfGaps)
    return hashtable1

def activeUsersFile(hashtable, df):
    """
    This function creates 2 csv files. 'ActiveUserDetails.csv' contains posts of active users and 'InactiveUserDetails.csv'
    contains posts of inactive users. They are both subsets of 'preprocessed_posts.csv'. The active users are obtained
    from the remaining authors in the hashtable after removing the authors that do not meet the 2 criteria set.
    :param hashtable: hashtable has only active users as key
    :param df: data frame obtained from the preprocessed posts' dataset
    :return: NIL
    """
    df2 = df[df['author'].isin(hashtable)]
    df3 = df[~df['author'].isin(hashtable)] #create a data frame for users that are not in the hastable
    df2 = df2.drop(df2.columns[[0]], axis=1)
    influencerList = []
    for ind in df2.index:
        influencerList.append(hashtable[df2['author'][ind]])
    df2['Influencer'] = influencerList
    df2.to_csv('ActiveUserDetails.csv')
    df3.to_csv('InactiveUserDetails.csv')


def main():
    df = pd.read_csv('../Preprocessing/preprocessed_posts.csv')
    hashtable = filteringActive(df)
    activeUsersFile(hashtable, df)

if __name__ == '__main__':
    main()