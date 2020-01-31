import pandas as pd
import pytest
import DescriptiveAnalysis

def testgroupbymonth():
    data = {'author': ['ivan','Loic','Loic','Jack','Gariel'],'month':[1,4,4,6,6],'num_comments': [15,1,8,10,8]}
    df = pd.DataFrame(data)
    df2 = DescriptiveAnalysis.groupByMonth(df)
    assert df2['No_of_Posts'][1] == (2/1), "Group by month test failed"
    assert df2['No_of_Posts'][2] == (2/2), "Group by month test failed"
    assert df2['No_of_Comments'][2] == (18/2), "Group by month test failed"

