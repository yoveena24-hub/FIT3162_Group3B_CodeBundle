from Preprocessing import preprocess
from RedditUserTypes import ActiveInactiveUsers
from RedditUserTypes import SentimentAnalysisOfActiveUsers
from Evaluation.TrainingAndTesting import TrainingAndTestingDataAttributes
from DescriptiveAnalysisDatasets import DescriptiveAnalysis

if __name__ == '__main__':
    preprocess.main()
    ActiveInactiveUsers.main()
    SentimentAnalysisOfActiveUsers.main()
    TrainingAndTestingDataAttributes.main()
    DescriptiveAnalysis.main()