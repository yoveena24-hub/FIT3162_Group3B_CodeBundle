"""
Author: Team 3B
Monash IDs: 29201128, 29201128, 29019834, 27467058
Last updated by: Yoveena Vencatasamy (29019834)
Last updated date: 30.01.2020
-------------------------------------------------------------
This program cleans the attributes of the dataset in the order as follows:
Delete empty posts -> Delete duplicate posts -> Delete posts in other languages -> Convert utc time to string format ->
Convert all posts' title and description to lower case -> Tokenize -> Filter stop wors -> Stemming -> Lemmatizing
"""
import pandas as pd
import string

#Use of nltk library
from nltk.tokenize import word_tokenize
import nltk

def preprocess(filename, type):
    """
    This function preprocesses the data in the file indicated by the file name. It then writes the preprocessed data into
    another file
    Preprocessed posts - 'preprocessed_posts.csv'
    Preprocessed comments - 'preprocessed_comments.csv'
    :param filename: The name of the file from which data must be preprocessed
    :param type: posts/comments based on the file that needs to be preprocessed. Since the attributes for each of the types change.
    :return: NIL
    """
    df = convertToLower(convertTime(deleteOtherLangPosts(deleteDuplicatedPosts(deleteEmptyPosts(filename)), type)),
                        type)
    if type == "posts":
        df.to_csv('preprocessed_posts.csv')
    else:
        df.to_csv('preprocessed_comments.csv')


def deleteEmptyPosts(filename):
    """
    This function deletes all empty posts in the dataset.
    Empty posts can be denoted as [deleted] or empty character in the post
    :param filename: name of the file from which the empty posts might be deleted
    :return: a dataframe holding the dataset after removing empty posts
    """
    # Making a list of missing value types
    missing_values = ["[deleted]", " "]

    # convert missing values into NA
    df = pd.read_csv(filename, na_values=missing_values)

    # delete rows with NA
    df = df.dropna()

    return df


def deleteDuplicatedPosts(df):
    """
    Duplication of posts are eliminated by checking the id of the post or comment.
    Since the same id cannot exist twice, such posts are deleted
    :param df: dataframe from which duplicated posts must be deleted
    :return: dataframe which now contains unique posts or comments only
    """
    # deleting duplicated posts
    df = df.drop_duplicates(subset=['id'], keep=False)
    return df


def deleteOtherLangPosts(df, type):
    """
    Our system is only going to analyse posts in the english language, therefore other language posts are eliminated
    Other language posts are identified using the ascii values of the characters
    :param df: the dataframe in which other language posts must be eliminated
    :param type: type of the file (posts/comments)
    :return: dataframe that contains posts only in english lanuage
    """

    # if the type of the dataframe is posts, then check for ascii characters in the description and title fields
    if type == "posts":
        df['description'] = df['description'].apply(
            lambda x: ''.join(["" if ord(i) < 32 or ord(i) > 126 else i for i in x]))
        df['title'] = df['title'].apply(lambda x: ''.join(["" if ord(i) < 32 or ord(i) > 126 else i for i in x]))

    # else if the type of the dataframe is comments, then check for ascii characters in the body column
    else:
        df['body'] = df['body'].apply(lambda x: ''.join(["" if ord(i) < 32 or ord(i) > 126 else i for i in x]))

    return df


def convertTime(df):
    """
    This function converts utc time to normal time
    The time format now becomes year-month-date hour-min-secs
    :param df: data frame in which 'created_utc' column has to be converted
    :return df: data frame in which UTC time has been converted to string format
    """
    from datetime import datetime
    normalTime = []
    for ind in df.index:
        ts = int(df['created_utc'][ind])
        normalTime.append(datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'))
    df['created_utc'] = normalTime
    df['year'] = pd.DatetimeIndex(df['created_utc']).year
    df['month'] = pd.DatetimeIndex(df['created_utc']).month
    df['day'] = pd.DatetimeIndex(df['created_utc']).day
    return df


def convertToLower(df, type):
    """
    This functions converts all characters related to posts to lower case
    :param df: data frame that require the posts' details to be converted
    :param type: posts/comments based on the file that needs to be preprocessed. Since the attributes for each of the types change.
    :return df: returns the data frame with edited rows of title and description or body (comments data frame)
    """
    # convert to lower case
    if type == "posts":
        df['title'] = df['title'].str.lower()
        df['description'] = df['description'].str.lower()
    else:
        df['body'] = df['body'].str.lower()
    return df


def tokenization(df, type):
    """
    This function performs tokenization on posts' details. It removes extra spaces,special characters
    and punctuation marks from the rows of columns 'title' and 'description' or 'body' (comments data frame).
    It splits all words into a list for each row.
    :param df: data frame that require the posts' details to be tokenized
    :param type: posts/comments based on the file that needs to be preprocessed. Since the attributes for each of the types change.
    :return df: returns the data frame with edited rows of title and description or body (comments data frame)
    """
    # tokenization
    # removing the punctuations
    if type == "posts":
        df['title'] = df['title'].str.replace('[{}]'.format(string.punctuation), '')
        df['description'] = df['description'].str.replace('[{}]'.format(string.punctuation), '')

        # splitting into words in a list
        df['title'] = df['title'].apply(word_tokenize)
        df['description'] = df['description'].apply(word_tokenize)
    else:
        df['body'] = df['body'].str.replace('[{}]'.format(string.punctuation), '')
        df['body'] = df['body'].apply(word_tokenize)
    return df


def filtering(df, type):
    """
    This function removes common word, stop words and specific jargon words. Those words do not contribute to
    defining a user. eg of words are articles, prepositions, conjunctions and auxiliary words
    :param df: data frame that require the posts' details to be filtered
    :param type: posts/comments based on the file that needs to be preprocessed. Since the attributes for each of the types change.
    :return df: returns the data frame with edited rows of title and description or body (comments data frame). Each row is a list of words
    """
    # filtering
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    if type == "posts":
        df['title'] = df['title'].apply(lambda x: [item for item in x if item not in stop_words])
        df['description'] = df['description'].apply(lambda x: [item for item in x if item not in stop_words])
    else:
        df['body'] = df['body'].apply(lambda x: [item for item in x if item not in stop_words])
    return df


def stemming(df, type):
    """
    This function reduces each word in a list to the word stem or root
    :param df: data frame that require the posts' details to be stemmed
    :param type: posts/comments based on the file that needs to be preprocessed. Since the attributes for each of the types change.
    :return df: returns the data frame with edited rows of title and description or body (comments data frame). Each row is a list of words
    """
    # stemming - converting words back to their stems (roots)
    from nltk.stem.snowball import SnowballStemmer
    stemmer = SnowballStemmer("english")
    if type == "posts":
        df['title'] = df['title'].apply(lambda x: [stemmer.stem(y) for y in x])
        df['description'] = df['description'].apply(lambda x: [stemmer.stem(y) for y in x])
    else:
        df['body'] = df['body'].apply(lambda x: [stemmer.stem(y) for y in x])
    return df


def lemmatize_text(text):
    """
    This function is a more advanced stemming, before reducing words, it takes into account meaning in a sentence
    :param text:
    :return lemmatised text:
    """
    lemmatizer = nltk.stem.WordNetLemmatizer()
    return [lemmatizer.lemmatize(w) for w in text]


def lemmatize(df, type):
    """
    This function performs lemmatization on all rows of 'title' and 'description column or 'body' column depending on the type
    of the data frame (posts/comments). It calls 'lemmatize_text' function
    :param df: data frame that require the posts' text to be stemmed
    :param type: posts/comments based on the file that needs to be preprocessed. Since the attributes for each of the types change.
    :return df: returns the data frame with edited rows of title and description or body (comments data frame). Each row is a list of words
    """
    # lemmatizing - more stemming
    if type == "posts":
        df['title'] = df['title'].apply(lemmatize_text)
        df['description'] = df['description'].apply(lemmatize_text)
    else:
        df['body'] = df['body'].apply(lemmatize_text)
    return df


def joinList(df, type):
    """
    This function joins the words in a list to a single string. This step is required for sentiment analysis of the posts' details.
    All the words in the columns of the dataframe are currently in a list and the sentiment analyser function requires a
    string as parameter.
    :param df: data frame that require the posts' details lists to be joined in a single string
    :param type: posts/comments based on the file that needs to be preprocessed. Since the attributes for each of the types change.
    :return df: returns the data frame with edited rows of title and description or body (comments data frame). Each row is string
    """
    if type == "posts":
        titleJoin = []
        descJoin = []
        for ind in df.index:
            titleJoin.append(" ".join(df['title'][ind]))
            descJoin.append(" ".join(df['description'][ind]))
        df['title'] = titleJoin
        df['description'] = descJoin
    else:
        bodyJoin = []
        for ind in df.index:
            bodyJoin.append(" ".join(df['body'][ind]))
        df['body'] = bodyJoin

    return df

def main():
    # preprocess("../SourceFiles/Dell.csv", 'posts')
    # preprocess("../SourceFiles/DellComments.csv", 'comments')
    preprocess("../SourceFiles/laptops.csv", 'posts')
    preprocess("../SourceFiles/laptopComments.csv", 'comments')

if __name__ == '__main__':
    main()