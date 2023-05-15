import unicodedata
import re
import json

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords

import pandas as pd

import os
import acquire as a
from sklearn.model_selection import train_test_split
import nltk.sentiment

def acquire_github_data():
    '''
    The function returns a dataframe by either eading the CSV file titled 
    'repos.csv' or if the file does not exist, the function will run the web 
    scraping function from the acquire modeule to produce a dataframe which
    then is saved as 'repos.csv' and read to a dataframe.
    '''
    if os.path.isfile('repos.csv'):
        return pd.read_csv('repos.csv', index_col=0)
    else:
        df = pd.DataFrame(a.scrape_github_data())
        df.to_csv('repos.csv')
        return df
    
def lower(some_string):
    '''
    The function takes in a string and converts the characters to lowercase.
    Returns the string.
    '''
    some_string = some_string.lower()
    return some_string

def normalize(some_string):
    '''
    The function takes in a string and normalizes, encodes, and decodes it.
    The string then has a regualr expression applied to it which looks for 
    characters that are alphabetical, a number or an apostrophy.
    Returns the string.
    '''
    some_string = unicodedata.normalize('NFKD', some_string).encode('ascii', 'ignore').decode('utf-8')
    some_string = re.sub(r'[^a-zA-Z0-9\'\s]', '', some_string)
    return some_string

def basic_clean(some_string):
    '''
    The function takes a string and applies the lower and the normalize functions
    to it. Returns the string.
    '''
    some_string = lower(some_string)
    some_string = normalize(some_string)
    return some_string

def tokenize(some_string):
    '''
    The function takes in a string and applies the tokenizer to it. Returns the string.
    '''
    tokenizer = nltk.tokenize.ToktokTokenizer()
    return tokenizer.tokenize(some_string, return_str=True)

def lemmatize(some_string):
    '''
    The function takes in a string and applies the lemmatizer to it. returns the string.
    '''
    lemmatizer = nltk.WordNetLemmatizer()
    some_string = ' '.join(
        [lemmatizer.lemmatize(word) for word in some_string.split()])
    some_string = ' '.join(
        [lemmatizer.lemmatize(word, 'v') for word in some_string.split()])
    some_string = ' '.join(
        [lemmatizer.lemmatize(word, 'a') for word in some_string.split()])
    some_string = ' '.join(
        [lemmatizer.lemmatize(word, 'r') for word in some_string.split()])
    return some_string

def stem(some_string):
    '''
    The function splits each string and then applies the stemmer to each 
    element. It then joins the string, and returns it.
    '''
    stemmer = nltk.porter.PorterStemmer()
    return ' '.join(
        [stemmer.stem(word) for word in some_string.split()])

def remove_stopwords(some_string, extra_words=['python', 'Python', 'Rust', 'rust', 'JavaScript', 'javascript'], keep_words=[]):
    '''
    The function takes in a string as well as two optional lists.
    The parameter 'extra_words' is a list of string swhich should be 
    included in the stopwords, and the parameter 'keep_words' which 
    will be excluded from the standard list of English stopwords.
    '''
    stopwords_custom = set(stopwords.words('english')) - \
    set(keep_words)
    stopwords_custom = list(stopwords_custom.union(extra_words))
    return ' '.join([word for word in some_string.split()
                     if word not in stopwords_custom])

def transform_data(df):
    '''
    The function take in a dataframe and creates a new column to contain the
    changes from the functions called. Returns a dataframe.
    '''
    df = df.rename(columns={'readme_contents': 'original'})
    df['clean'] = df['original'].apply(basic_clean).apply(
        tokenize).apply(remove_stopwords)
    df['stemmed'] = df['clean'].apply(stem)
    df['lemmatized'] = df['stemmed'].apply(lemmatize)
    return df

def prepare_github_df():
    '''
    The function acquires the dataframe, and then applies the transfrom
    function on it. Returns the transformed dataframe.
    '''
    df = acquire_github_data()
    df_cleaned = transform_data(df)
    return df_cleaned

def get_word_count(train, validate, test):
    train['word_count'] = train['clean'].apply(lambda x: len(x.split()))
    validate['word_count'] = validate['clean'].apply(lambda x: len(x.split()))
    test['word_count'] = test['clean'].apply(lambda x: len(x.split()))
    return train, validate, test

# Use the following function for Exploring the data
def split_data():
    '''
    The function splits the dataframe into train, test, and validate 
    while stratifying on the target 'language'.
    '''
    df = prepare_github_df()
    
    train_validate, test = train_test_split(df, random_state = 1349, train_size=.8, stratify=df.language)

    train, validate = train_test_split(train_validate, random_state = 1349, train_size=.7, stratify=train_validate.language)
    
    return train, validate, test

def add_sentiment(train, validate, test):
    sia = nltk.sentiment.SentimentIntensityAnalyzer()
    train['compound_sentiment'] = train['clean'].apply(lambda x: sia.polarity_scores(x)['compound'])
    validate['compound_sentiment'] = validate['clean'].apply(lambda x: sia.polarity_scores(x)['compound'])
    test['compound_sentiment'] = test['clean'].apply(lambda x: sia.polarity_scores(x)['compound'])
    return train, validate, test
    
# Use the following function for modeling
def modeling_prep():
    '''
    The function splits the data into train, validate, and test. The function
    then isolates each of these elements into the X and y dataframes, where y contains
    the target of 'langauge', and X contains the README content.
    '''
    train, validate, test = split_data()
    X_train = train['lemmatized']
    X_validate = validate['lemmatized']
    X_test = test['lemmatized']
    y_train = train['language']
    y_validate = validate['language']
    y_test = test['language']
    return X_train, X_validate, X_test, y_train, y_validate, y_test

