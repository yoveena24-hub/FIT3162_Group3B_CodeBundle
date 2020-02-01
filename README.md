FIT3162_Group3B_CodeBundle

# USER PROFILING AND RECOMMENDATION SYSTEMS USING USER GENERATED CONTENT

This project aims at reducing marketing cost by using sentiment analysis techniques. User generated content for 1 year has been obtained from the online social platform, Reddit. The information is processed and a list of reddit users that are influencers and potential clients to a particular product are extracted. Those are used as inputs to a user recommender system web application. This application will be made available to vendors so that they can directly contact and make advertising to users that are interested in their brand or product. The second phase of the project is to perform the score prediction of a post; for Reddit social platform, 'score' is defined as the number of upvotes against the number of downvotes. Using various posts' attributes, different machine learning algorithms have been used for score prediction. The output allows us determine whether there is a relationship between an influencer and score of his post.

**Getting Started**

The following instructions will guide you through the packages needed to install and will get you a copy of the project up and running on your local machine for development and testing purposes.

**Prerequisites**

1. Pycharm - Python IDE for developers
2. Github Account
3. Latest python version
4. R 
5. RStudio - Integrated development environment for R
6. pip launched when set up pyCharm - pip can download and install other packages

**Installing**

In pyCharm , go to terminal and install the following packages:

nltk : 'sudo pip3 install -U nltk', for usage of nltk corpora, visit https://www.educba.com/install-nltk/

pandas: 'pip install pandas'

numpy: 'pip install numpy'

pytest: 'pip install pytest'

If you need to upgrade your current pip version: 'python -m pip install --upgrade pip' command.

In RStudio, go to console and install the following packages: tree, e1071, ROCR, randomForest, adabag, rpart using 'install.packages("package")'. A guideline has been provided in the R file.

Install R Markdown in Rstudio by visiting the link: https://rmarkdown.rstudio.com/authoring_quick_tour.html

**Running the python program**

1. Download all the folders in the repository
2. If you have new reddit data sets, save them in the SourceFiles folder.
3. Run main.py file 
4. Each folder allocated will have the related csv files

**Running the R program**

1. In Evaluation -> MachineLearning folder, use the R file and the datasets.  
2. In RStudio, go to File -> Open File.. Browse and click on the downloaded R file.
3. Click on knit.

**Built With**

- TeamViewer
- Github - Version control

###### Authors

Yoveena Vencatasamy (29019834)
