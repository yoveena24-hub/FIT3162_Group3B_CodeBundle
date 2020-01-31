import pytest
import preprocess
import pandas as pd


def test_deleteEmptypost():
    column = ["author","abc", "29019", "1"," ", "309738", " ", "[deleted]", "309738", "1", "1"]
    description = ["Hello am there!!", "नमस्कार", "Say", "a", "bc","def", "fdhj", "gfdchk", "5167", "hfugwdl", "ghcw"]
    title = ["Any ongoing good deals on XPS 15?", "स्वागत", "सुप्रभात", "a", "bc","def", "fdhj", "gfdchk", "5167", "sflyfsl","ghhhh"]
    utc = ['1514784048','1514788947','1514793905','1514798227','1514801477', '1514818023','1514820682','1514818023','1514820682','1514821651','1514828916']
    df = pd.DataFrame(columns=['created_utc','id', 'title','description'])
    df['id'] = column
    df['description'] = description
    df['title'] = title
    df['created_utc'] = utc

    df.to_csv('testPreprocessfile.csv')
    df1 = preprocess.deleteEmptyPosts("testPreprocessfile.csv")
    df1.to_csv("testPreprocessfile.csv")
    assert df1['id'].count() == 8, "Delete empty post test failed"


def test_deleteDuplicates():
    df = pd.read_csv("testPreprocessfile.csv")
    df1 = preprocess.deleteDuplicatedPosts(df)
    df1.to_csv("testPreprocessfile.csv")
    assert df1['id'].count() == 3, "Delete duplicates test failed"

def test_deleteOtherLangPosts():
    df = pd.read_csv("testPreprocessfile.csv")
    df1 = preprocess.deleteOtherLangPosts(df,"posts")
    df1.to_csv("testPreprocessfile.csv")
    df2 = preprocess.deleteEmptyPosts("testPreprocessfile.csv")
    df2.to_csv("testPreprocessfile.csv")
    assert df2['id'].count() == 1, "Delete other language posts test failed"

def testConvertTime():
    df = pd.read_csv("testPreprocessfile.csv")
    df1 = preprocess.convertTime(df)
    df1.to_csv("testPreprocessfile.csv")
    assert df1['created_utc'][0] == '2018-01-01 05:20:48', "Convert utc time failed"

def testtokenize():
    df = pd.read_csv("testPreprocessfile.csv")
    df1 = preprocess.tokenization(df,"posts")
    df1.to_csv("testPreprocessfile.csv")
    assert df1['title'][0] == ['Any', 'ongoing', 'good' ,'deals', 'on', 'XPS', '15']
    assert df1['description'][0] == ['Hello','am','there']

def teststopwords():
    df = pd.read_csv("testPreprocessfile.csv")
    df1 = preprocess.filtering(df,"posts")
    df1.to_csv("testPreprocessfile.csv")
    assert " ".join(df1['description'][0]) == "[ ' H e l l ' ,   ' ' ,   ' h e r e ' ]"

def teststemming():
    df = pd.read_csv("testPreprocessfile.csv")
    df1 = preprocess.stemming(df,"posts")
    df2 = preprocess.joinList(df1,"posts")
    df2.to_csv("testPreprocessfile.csv")
    # assert df2['description'][0] == "[ ' [ ' ,   "" ' "" ,   ' h ' ,   ' e ' ,   ' l ' ,   ' l ' ,   "" ' "" ,   ' , ' ,   '   ' ,   "" ' "" ,   "" ' "" ,   ' , ' ,   '   ' ,   "" ' "" ,   ' h ' ,   ' e ' ,   ' r ' ,   ' e ' ,   "" ' "" ,   ' ] ' ]"


