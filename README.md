# Can the Text of GitHub READMEs be Used to Predict the Primary Coding Language for the Repository?

##### *A CodeUp Data Science Project by Caroline Miller and Corey Baughman*
##### May 17th, 2023
***

### Abstract
**Project Description:** Can Natural Language Processing (NLP) techniques be used to analyze the text of the README file for a GitHub repository (repo) and predict the primary coding language used in that repo? That is to say, are there clues that can be discovered in the word selection, syntax, length of document, etc., that are indicative of repos written in a certain coding language and can they be used to make classification predictions of the coding languages of unseen repos?

**Goal:** To use NLP techniques on the text of README files to build a model which will predict the primary language that the repo is coded with.

**Plan of Attack:**
To achieve this goal we will work through the steps of the data science pipeline (which form the Table of Contents below). 
- We will utilize webscraping techniques to gather a list of trending repositories from GitHub. 
- We will then clean and lemmatize the text of each README and combine all text into a corpus to explore univariate distributions of words.
- We will then split the dataset into training, validation, and test datasets and complete Exploratory Data Analysis on the training data
- Then we will try use various classification algorithms to build models that will predict coding languages of validation set.
- Finally, we will use our best performing model on the test dataset and draw conclusions about success versus baseline models.

### Table of Contents
 
[1. Plan and Investigate Background Information](#Intro:-Background-Information)

[2. Formulate Initial Questions/Hypotheses](#Initial-Questions)

[3. Acquire and Prepare Data](#Data-Acquisition-and-Preparation)

[4. Exploratory Data Analysis](#Exploratory-Data-Analysis)

[5. Preprocessing](#Pre-processing)

[6. Modeling](#Modeling)

[7. Model Evaluation](#Model-Evaluation)

[8. Conclusions](#Conclusions)

[9. Next Steps](#Next-Steps)

[Appendix](#Appendix:-Instructions-to-Reproduce-Work)

***

### Intro: Background Information

Webscrapin

[back to top](#Table-of-Contents)
***

#### Data Dictionary 

| Attribute | Description of Attribute |
| :---------| :------------------------ |
| basic_clean | Original README text converted to all lowercase and characters normalized to 'NFKD' then encoded to ascii and decoded back to utf-8. Regular expression were then applied to replace all occurrences of newline characters carriage returns and match and remove any character that is not a letter (uppercase or lowercase), digit, apostrophe ('), or whitespace. |
| clean | A column with the text from basic_clean that is tokenized and has stopwords removed with nltk.tokenize.toktok.ToktokTokenizer() and nltk.corpus.stopwords(). |
| compound_ratio | The compound ratio is a single numerical value between -1 and +1 that indicates the overall sentiment polarity of a text. |
| language | The target of modeling. This is the primary coding language of repo. It is either Python, Javascript, or Rust. |
| lemmatized | A column with the lemmatized versions of the original README texts. Lemmatized with nltk.WordNetLemmatizer(). |
| num_unique | The number of unique words used in each README. |
| percent_unique | The percentage of unique words v. total words in each README. |
| repo | the url extension for the repo. It is formatted for GitHub REST API. Remove leading period and insert https://github.com/ to navigate to repo. |
| stemmed | A column with the cleaned READMEs where the words have been stemmed using nltk.porter.PorterStemmer(). |
| stopword_count | A column with a count of the total stopwords in each original README text. |
| stopword_ratio | A column with the ratio of stopwords to other words in each original README text. |
| word_count | the total number of words in each README. |

[back to top](#Table-of-Contents)
***

### Initial Questions

1. Does alcohol effect wine quality?
1. Does density effect wine quality?
1. Do chlorides effect wine quality?
1. Is there a difference in quality for red or white wine?
1. Is there a correlation between volatile acidity and quality?
1. Is there a linear correlation between residual free sulphur and quality?

[back to top](#Table-of-Contents)
***

### Data Acquisition and Preparation

- goals
- tools
- methods
- findings

[back to top](#Table-of-Contents)
***

### Exploratory Data Analysis

1. negative correlations:

    a. fixed acidity
    
    b. volatile acidity
    
    c. chlorides
    
    d. total sulfur dioxide
    
    e. density
    
2. positive correlations:
    
    a. citric acid
    
    b. free sulfur dioxide
    
    c. alcohol
    
    d. is red
    
3. no correlation:
    
    a. residual sugar
    
    b. pH
    
    c. sulphates
    
4. Clusters:

    We spent some effort examining different combinations of variables for useful clusters to aid in regression or classification, but we did not find any that beat our non-cluster models.

[back to top](#Table-of-Contents)
***

### Pre-processing

1. split target from features for all datasets

1. scaled X_train, X_validate, X_test

[back to top](#Table-of-Contents)
***

### Modeling

1. We chose to treat this as a regression problem

1. Regression algorithms used:

    a. OLS regressor
    
    b. Tweedie Regressor
    
    c. Polynomial Features
    
    d. LassoLars 

1. Findings: Our best model was a Polynomial Regression on all features that decreased errors by 23.35% over baseline

[back to top](#Table-of-Contents)
***

### Model Evaluation

1. established a baseline by testing mean and median quality values as predictions. We found that mean had a slightly lower RMSE and settled on that as a baseline model.

1. We used RMSE as our evaluation metric supplemented by plotting overlaid histograms of predicted and actual values to understand where the model performed well and where it didn't.

[back to top](#Table-of-Contents)
***

### Conclusions

1. We achieved our goals of finding drivers of wine quality as well as creating a model that outperforms baseline predictions of quality.

1. However, our model did not perform well at predicting high quality wines.

1. We found that almost all features were needed to get the best model.

[back to top](#Table-of-Contents)
***

### Next Steps

1. add class balancing to model
2. experiment with outlier detection modeling
3. explore using a classification model

[back to top](#Table-of-Contents)
***

### Appendix A: Instructions to Reproduce Work

1. Download project repo here:
https://github.com/DCWineCO/clustering_wine

2. Open and Run 'cluster_final_report.ipynb'

3. Necessary modules are included in the repo and should not need additional work to run as long as the whole repo is stored in the same directory.

[back to top](#Table-of-Contents)
***

### Appendix B: Links to Original Research

    - summary: https://data.world/food/wine-quality/workspace/project-summary?agentid=food&datasetid=wine-quality
    - research abstract: https://www.sciencedirect.com/science/article/abs/pii/S0167923609001377?via%3Dihub

[back to top](#Table-of-Contents)
***

### Appendix C: Sources

This dataset is public available for research. The details are described in [Cortez et al., 2009].
Please include this citation if you plan to use this database:

P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
Modeling wine preferences by data mining from physicochemical properties.
In Decision Support Systems, Elsevier, 47(4):547-553. ISSN: 0167-9236.

Available at: [@Elsevier] http://dx.doi.org/10.1016/j.dss.2009.05.016
[Pre-press (pdf)] http://www3.dsi.uminho.pt/pcortez/winequality09.pdf
[bib] http://www3.dsi.uminho.pt/pcortez/dss09.bib

Title: Wine Quality
Sources
Created by: Paulo Cortez (Univ. Minho), Antonio Cerdeira, Fernando Almeida, Telmo Matos and Jose Reis (CVRVV) @ 2009

[back to top](#Table-of-Contents)


