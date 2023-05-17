### Can the Text of GitHub READMEs be Used to Predict the Primary Coding Language for the Repository?

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

We were tasked with acquiring data on at least 100 GitHub repositories in order to use Natural Language Processing(NLP) tools to predict the primary coding language of the repo from the text of the README file (with all mentions of the target languages removed). We explored the GitHub Rest API as well as Beautiful Soup to acquire the data. We used tools from the Natural Language Tool Kit (NLTK) to process README texts into cleaned and lemmatized forms that we then could explore and model.




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

1. What are the most frequently used words across the three programming languages and are any of them shared between them?
2. Are the means of the sentiment values for each language different?
3. Are the means of stopword ratios different between Python, JavaScript, and Rust READMEs?
4. Are the means of the unique word percentages between Python, JavaScript, and Rust READMEs different?

[back to top](#Table-of-Contents)
***

### Data Acquisition and Preparation

- **goal:** to acquire trending repository README files and the primary coding language of the repos.
- **tools:** GitHub REST API, Beautiful Soup, NLTK
- **methods** 
    - Acquisition: Used a python script to get URLs of trending repos. GitHub only makes the top 25 available but language and time window parameters can be adjusted, so the script was able to retrieve the top 25 for each of three languages (Python, Rust, and Java were selected) and also check daily, weekly, and monthly lists. After removing duplicates (there was some overlap between the time windows) We were able to retrieve 164 total READMEs: Python 58, JavaScript 56, and Rust 50, on May 14th, 2023.
    - Preparation:
- **findings:** GitHub REST API required the URLs to be input as ./repo_name 

The wrangle module uses the following steps to clean the README contents:

    lower()
    unicodedata.normalize to remove any inconsistencies in unicode character encoding
    .encode to convert the resulting string to the ASCII character set
    .decode to turn the resulting bytes object back into a string
    regular expressions cpture groups to isolate words and numbers
    tokenize to break words and any punctuation left over into discrete units
    lemmatize to keep only the root words
    remove stopwords as listed in the english dictionary (according to nltk.corpus)

The following features are engineered:

    count of stopwords
    ratio of stopwords to all words
    calcualtion of sentiment
    number of unique words
    ratio of unique words to all words



[back to top](#Table-of-Contents)
***

### Exploratory Data Analysis

 - Most frequent of words not shared proportionally
 - JavaScript contains technical words
 - Shared words: use, run, install, data
 - Means differ across features
 - Sentiment lower in Python
 - Rust features are uniform

[back to top](#Table-of-Contents)
***

### Pre-processing

1. split target from features for all datasets

1. scaled X_train, X_validate, X_test for models with additional features included (versus models with just vectorized text).


[back to top](#Table-of-Contents)
***

### Modeling

1. We chose to treat this as a classification problem

1. Algorithms used:

    a. Random Forest Classifier
    
    b. Decision Tree Classifier
    
    c. KNN Classifier
    
    d. Multinomial Naive Bayes Classifier

1. Findings: Best Model Performance on Test Dataset was knn model with k=16 on the out-of-sample data: improved classification accuracy by 60.06% over baseline.

[back to top](#Table-of-Contents)
***

### Model Evaluation

1. established a baseline by using the most common class ('Python') as the prediction.

1. We used accuracy as our evaluation metric as the classes were well balanced and there was not implicit benefit in detecting false positives v false negatives

[back to top](#Table-of-Contents)
***

### Conclusions

1. Engineered features are useful for exploratory data analysis, but not for modeling

2. Top words of each language are not shared  

3. KNN best for modeling

[back to top](#Table-of-Contents)
***

### Next Steps

1. Acquire more READMEs

2. Use Clusters as features

3. Refine regular expressions

4. Use CountVectorization

5. Latent Dirichlet Allocation (LDA)

[back to top](#Table-of-Contents)
***

### Appendix A: Instructions to Reproduce Work

1. Download project repo here:
https://github.com/EasyNLPeasy/nlp_team_project

2. Open and Run 'github_repos_nlp_report.ipynb'

3. Necessary modules are included in the repo and should not need additional work to run as long as the whole repo is stored in the same directory.

[back to top](#Table-of-Contents)
***

### Appendix B: Links

  - GitHub provides documentation of the GitHub REST API here: https://docs.github.com/en/rest?apiVersion=2022-11-28
  - NLTK resoureces can be found here: https://www.nltk.org
[back to top](#Table-of-Contents)
***


