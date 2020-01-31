from Preprocessing import preprocess
from RedditUserTypes import ActiveInactiveUsers
from RedditUserTypes import SentimentAnalysisOfActiveUsers
from DescriptiveAnalysisDatasets import DescriptiveAnalysis

import sys
sys.path.insert(0, 'Evaluation/TrainingAndTesting')
import TrainingAndTestingDataAttributes


if __name__ == '__main__':
    preprocess.main()
    ActiveInactiveUsers.main()
    SentimentAnalysisOfActiveUsers.main()
    TrainingAndTestingDataAttributes.main()
    DescriptiveAnalysis.main()

