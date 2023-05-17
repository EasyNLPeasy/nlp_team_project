# NLP Project: Modeling

# general imports
import numpy as np
import pandas as pd
import itertools

# scaling
from sklearn.preprocessing import MinMaxScaler

# classification tools
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB, MultinomialNB

# NLP
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# local modules
import wrangle as w

def rf_classification(df, df_val, target, least_min_samples_leaf=1, 
                      most_min_samples_leaf=10, min_max_depth=1, max_max_depth=10):
    '''
    Perform random forest classification on the given data.

    Args:
        df (DataFrame): The training DataFrame containing the feature columns and the 
        target column.
        df_val (DataFrame): The validation (or test) DataFrame containing the feature 
        columns and the target column.
        target (str): The name of the target column.
        least_min_samples_leaf (int): The low range of the minimum samples per leaf.
        most_min_samples_leaf (int): The high range of the minimum  samples per leaf.
        min_max_depth (int): The low range of the minimum depth of the tree.
        max_max_depth (int): The high range of the minimum depth of the tree.

  
    Returns:
        DataFrame: A DataFrame containing the combinations of hyperparameters and their 
        corresponding accuracy scores on the training and validation data. An 'algorithm' 
        column is added to denote random_forest.
    '''
    # define and split features and target
    X_train = df['lemmatized']
    y_train = df[target]
    X_validate = df_val['lemmatized']
    y_validate = df_val[target]
    
    # TFIDF vectorize the lemmatized text corpus
    tfidf = TfidfVectorizer()
    X_train = tfidf.fit_transform(X_train)
    X_validate = tfidf.transform(X_validate)

    rf_models = {}
    bootstrap = (True, False)
    sizes = range(least_min_samples_leaf, most_min_samples_leaf)
    depths = range(min_max_depth, max_max_depth)
    hyper_list = list(itertools.product(bootstrap, sizes, depths))
    
    for hyperparams in hyper_list:
        rf = RandomForestClassifier(bootstrap=hyperparams[0], 
                            class_weight=None, 
                            criterion='gini',
                            min_samples_leaf=hyperparams[1],
                            n_estimators=100,
                            max_depth=hyperparams[2], 
                            random_state=9751)
        rf.fit(X_train, y_train)
        preds = rf.predict(X_train)
        accuracy_train = rf.score(X_train, y_train)
        accuracy_val = rf.score(X_validate, y_validate)
        rf_models[hyperparams] = accuracy_train, accuracy_val
   
    df = pd.DataFrame([{'hyperparams': k, 'accuracy_train': v[0],
                          'accuracy_val': v[1]} for k, v in rf_models.items()])
    df['algorithm'] = 'random_forest'
    
    return df


def rf_classification_plus(df, df_val, target, least_min_samples_leaf=1, 
                      most_min_samples_leaf=10, min_max_depth=1, max_max_depth=10):
    '''
    Perform random forest classification on the given data.

    Args:
        df (DataFrame): The training DataFrame containing the feature columns and the 
        target column.
        df_val (DataFrame): The validation (or test) DataFrame containing the feature 
        columns and the target column.
        target (str): The name of the target column.
        least_min_samples_leaf (int): The low range of the minimum samples per leaf.
        most_min_samples_leaf (int): The high range of the minimum  samples per leaf.
        min_max_depth (int): The low range of the minimum depth of the tree.
        max_max_depth (int): The high range of the minimum depth of the tree.

  
    Returns:
        DataFrame: A DataFrame containing the combinations of hyperparameters and their 
        corresponding accuracy scores on the training and validation data. An 'algorithm' 
        column is added to denote random_forest.
    '''
    # reset indices after split
    df = df.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    
    # define and split features and target
    X_train = df['lemmatized']
    X_validate = df_val['lemmatized']
    y_train = df[target]
    y_validate = df_val[target]
    
    # TFIDF vectorize the lemmatized text corpus
    tfidf = TfidfVectorizer()
    X_train = tfidf.fit_transform(X_train)
    X_validate = tfidf.transform(X_validate)
    
    # make vectorized matrix dense and concat other features
    X_cols_train = df[['word_count', 'stopword_ratio', 'compound_sentiment', 'percent_unique']]
    X_cols_validate = df_val[['word_count', 'stopword_ratio', 'compound_sentiment', 'percent_unique']]
    X_train = pd.concat(
        [X_cols_train, pd.DataFrame(
            X_train.todense(), columns=tfidf.get_feature_names())], axis=1)
    X_validate = pd.concat(
        [X_cols_validate, pd.DataFrame(
            X_validate.todense(), columns=tfidf.get_feature_names())], axis=1)
    
    rf_models = {}
    bootstrap = (True, False)
    sizes = range(least_min_samples_leaf, most_min_samples_leaf)
    depths = range(min_max_depth, max_max_depth)
    hyper_list = list(itertools.product(bootstrap, sizes, depths))
    
    for hyperparams in hyper_list:
        rf = RandomForestClassifier(bootstrap=hyperparams[0], 
                            class_weight=None, 
                            criterion='gini',
                            min_samples_leaf=hyperparams[1],
                            n_estimators=100,
                            max_depth=hyperparams[2], 
                            random_state=9751)
        rf.fit(X_train, y_train)
        preds = rf.predict(X_train)
        accuracy_train = rf.score(X_train, y_train)
        accuracy_val = rf.score(X_validate, y_validate)
        rf_models[hyperparams] = accuracy_train, accuracy_val
   
    df = pd.DataFrame([{'hyperparams': k, 'accuracy_train': v[0],
                          'accuracy_val': v[1]} for k, v in rf_models.items()])
    df['algorithm'] = 'random_forest'
    
    return df


def dt_classification(df, df_val, target, least_min_samples_leaf=1, 
                      most_min_samples_leaf=10, min_max_depth=1, max_max_depth=10):
    '''
    Perform decision tree classification on the given data.

    Args:
        df (DataFrame): The training DataFrame containing the feature columns and the 
        target column.
        df_val (DataFrame): The validation (or test) DataFrame containing the feature 
        columns and the target column.
        target (str): The name of the target column.
        least_min_samples_leaf (int): The low range of the minimum samples per leaf.
        most_min_samples_leaf (int): The high range of the minimum  samples per leaf.
        min_max_depth (int): The low range of the minimum depth of the tree.
        max_max_depth (int): The high range of the minimum depth of the tree.

    Returns:
        DataFrame: A DataFrame containing the combinations of hyperparameters and their 
        corresponding accuracy scores on the training and validation data. An 'algorithm' 
        column is added to denote decision_tree.
    '''
    # define and split features and target
    X_train = df['lemmatized']
    y_train = df[target]
    X_validate = df_val['lemmatized']
    y_validate = df_val[target]
    
    # TFIDF vectorize the lemmatized text corpus
    tfidf = TfidfVectorizer()
    X_train = tfidf.fit_transform(X_train)
    X_validate = tfidf.transform(X_validate)
    
    dt_models = {}
    sizes = range(least_min_samples_leaf, (most_min_samples_leaf +1))
    depths = range(min_max_depth, (max_max_depth + 1))
    hyper_list = list(itertools.product(sizes, depths))
    
    for hyperparams in hyper_list:
        dt = DecisionTreeClassifier( 
                            criterion='gini',
                            min_samples_leaf=hyperparams[0],
                            max_depth=hyperparams[1], 
                            random_state=9751)
        dt.fit(X_train, y_train)
        preds = dt.predict(X_train)
        accuracy_train = dt.score(X_train, y_train)
        accuracy_val = dt.score(X_validate, y_validate)
        dt_models[hyperparams] = accuracy_train, accuracy_val
   
    df = pd.DataFrame([{'hyperparams': k, 'accuracy_train': v[0],
                          'accuracy_val': v[1]} for k, v in dt_models.items()])
    df['algorithm'] = 'decision_tree'
    
    return df


def dt_classification_plus(df, df_val, target, least_min_samples_leaf=1, 
                      most_min_samples_leaf=10, min_max_depth=1, max_max_depth=10):
    '''
    Perform decision tree classification on the given data.

    Args:
        df (DataFrame): The training DataFrame containing the feature columns and the 
        target column.
        df_val (DataFrame): The validation (or test) DataFrame containing the feature 
        columns and the target column.
        target (str): The name of the target column.
        least_min_samples_leaf (int): The low range of the minimum samples per leaf.
        most_min_samples_leaf (int): The high range of the minimum  samples per leaf.
        min_max_depth (int): The low range of the minimum depth of the tree.
        max_max_depth (int): The high range of the minimum depth of the tree.

    Returns:
        DataFrame: A DataFrame containing the combinations of hyperparameters and their 
        corresponding accuracy scores on the training and validation data. An 'algorithm' 
        column is added to denote decision_tree.
    '''
     # reset indices after split
    df = df.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    
    # define and split features and target
    X_train = df['lemmatized']
    X_validate = df_val['lemmatized']
    y_train = df[target]
    y_validate = df_val[target]
    
    # TFIDF vectorize the lemmatized text corpus
    tfidf = TfidfVectorizer()
    X_train = tfidf.fit_transform(X_train)
    X_validate = tfidf.transform(X_validate)
    
    # make vectorized matrix dense and concat other features
    X_cols_train = df[['word_count', 'stopword_ratio', 'compound_sentiment', 'percent_unique']]
    X_cols_validate = df_val[['word_count', 'stopword_ratio', 'compound_sentiment', 'percent_unique']]
    X_train = pd.concat(
        [X_cols_train, pd.DataFrame(
            X_train.todense(), columns=tfidf.get_feature_names())], axis=1)
    X_validate = pd.concat(
        [X_cols_validate, pd.DataFrame(
            X_validate.todense(), columns=tfidf.get_feature_names())], axis=1)
    
    dt_models = {}
    sizes = range(least_min_samples_leaf, (most_min_samples_leaf +1))
    depths = range(min_max_depth, (max_max_depth + 1))
    hyper_list = list(itertools.product(sizes, depths))
    
    for hyperparams in hyper_list:
        dt = DecisionTreeClassifier( 
                            criterion='gini',
                            min_samples_leaf=hyperparams[0],
                            max_depth=hyperparams[1], 
                            random_state=23)
        dt.fit(X_train, y_train)
        preds = dt.predict(X_train)
        accuracy_train = dt.score(X_train, y_train)
        accuracy_val = dt.score(X_validate, y_validate)
        dt_models[hyperparams] = accuracy_train, accuracy_val
   
    df = pd.DataFrame([{'hyperparams': k, 'accuracy_train': v[0],
                          'accuracy_val': v[1]} for k, v in dt_models.items()])
    df['algorithm'] = 'decision_tree'
    
    return df



def knn_classification(df, df_val, target, min_n_neighbors=2, max_n_neighbors=20):
    '''
    Perform KNN classification on the given data.

    Args:
        df (DataFrame): The training DataFrame containing the feature columns and the 
        target column.
        df_val (DataFrame): The validation (or test) DataFrame containing the feature 
        columns and the target column.
        target (str): The name of the target column.
        min_n_neighbors: The minimum number of neighbors to test
        max_n_neighbors: The maximum number of neighbors to test
    Returns:
        DataFrame: A DataFrame containing the combinations of hyperparameters and their 
        corresponding accuracy scores on the training and validation data. An 'algorithm' 
        column is added to denote knn.
    '''
    # define and split features and target
    X_train = df['lemmatized']
    y_train = df[target]
    X_validate = df_val['lemmatized']
    y_validate = df_val[target]
    
    # TFIDF vectorize the lemmatized text corpus
    tfidf = TfidfVectorizer()
    X_train = tfidf.fit_transform(X_train)
    X_validate = tfidf.transform(X_validate)
    
    knn_models = {}
    hyper_list = range(min_n_neighbors, (max_n_neighbors +1), 2)
    
    for hyperparams in hyper_list:
        knn = KNeighborsClassifier(n_neighbors=hyperparams)
        knn.fit(X_train, y_train)
        preds = knn.predict(X_train)
        accuracy_train = knn.score(X_train, y_train)
        accuracy_val = knn.score(X_validate, y_validate)
        knn_models[hyperparams] = accuracy_train, accuracy_val
   
    df = pd.DataFrame([{'hyperparams': k, 'accuracy_train': v[0],
                          'accuracy_val': v[1]} for k, v in knn_models.items()])
    df['algorithm'] = 'knn'
    
    return df


def knn_classification_plus(df, df_val, target, min_n_neighbors=2, max_n_neighbors=20):
    '''
    Perform KNN classification on the given data.

    Args:
        df (DataFrame): The training DataFrame containing the feature columns and the 
        target column.
        df_val (DataFrame): The validation (or test) DataFrame containing the feature 
        columns and the target column.
        target (str): The name of the target column.
        min_n_neighbors: The minimum number of neighbors to test
        max_n_neighbors: The maximum number of neighbors to test
    Returns:
        DataFrame: A DataFrame containing the combinations of hyperparameters and their 
        corresponding accuracy scores on the training and validation data. An 'algorithm' 
        column is added to denote knn.
    '''
     # reset indices after split
    df = df.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    
    # define and split features and target
    X_train = df['lemmatized']
    X_validate = df_val['lemmatized']
    y_train = df[target]
    y_validate = df_val[target]
    
    # TFIDF vectorize the lemmatized text corpus
    tfidf = TfidfVectorizer()
    X_train = tfidf.fit_transform(X_train)
    X_validate = tfidf.transform(X_validate)
    
    # make vectorized matrix dense and concat other features
    X_cols_train = df[['word_count', 'stopword_ratio', 'compound_sentiment', 'percent_unique']]
    X_cols_validate = df_val[['word_count', 'stopword_ratio', 'compound_sentiment', 'percent_unique']]
    X_train = pd.concat(
        [X_cols_train, pd.DataFrame(
            X_train.todense(), columns=tfidf.get_feature_names())], axis=1)
    X_validate = pd.concat(
        [X_cols_validate, pd.DataFrame(
            X_validate.todense(), columns=tfidf.get_feature_names())], axis=1)
    # scale all features
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_validate = scaler.transform(X_validate)
    
    # make an empty dictionary for model hyerparams and metrics
    knn_models = {}
    hyper_list = range(min_n_neighbors, (max_n_neighbors +1), 2)
    
    for hyperparams in hyper_list:
        knn = KNeighborsClassifier(n_neighbors=hyperparams)
        knn.fit(X_train, y_train)
        preds = knn.predict(X_train)
        accuracy_train = knn.score(X_train, y_train)
        accuracy_val = knn.score(X_validate, y_validate)
        knn_models[hyperparams] = accuracy_train, accuracy_val
   
    df = pd.DataFrame([{'hyperparams': k, 'accuracy_train': v[0],
                          'accuracy_val': v[1]} for k, v in knn_models.items()])
    df['algorithm'] = 'knn'
    
    return df



def nb_classification(df, df_val, target, min_alpha=0.1, max_alpha=5.0):
    '''
    Perform Naive Bayes classification on the given data.

    Args:
        df (DataFrame): The training DataFrame containing the feature columns and the 
        target column.
        df_val (DataFrame): The validation (or test) DataFrame containing the feature 
        columns and the target column.
        target (str): The name of the target column.
        min_alpha: The minimum alpha to test
        max_alpha: The maximum alpha to test
    Returns:
        DataFrame: A DataFrame containing the combinations of hyperparameters and their 
        corresponding accuracy scores on the training and validation data. An 'algorithm' 
        column is added to denote naive_bayes.
    '''
    # define and split features and target
    X_train = df['lemmatized']
    y_train = df[target]
    X_validate = df_val['lemmatized']
    y_validate = df_val[target]
    
    # TFIDF vectorize the lemmatized text corpus
    tfidf = TfidfVectorizer()
    X_train = tfidf.fit_transform(X_train)
    X_validate = tfidf.transform(X_validate)
    
    nb_models = {}
    hyperparams = 'default'
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    preds = nb.predict(X_train)
    accuracy_train = nb.score(X_train, y_train)
    accuracy_val = nb.score(X_validate, y_validate)
    nb_models[hyperparams] = accuracy_train, accuracy_val
   
    df = pd.DataFrame([{'hyperparams': k, 'accuracy_train': v[0],
                          'accuracy_val': v[1]} for k, v in nb_models.items()])
    df['algorithm'] = 'multinomial_naive_bayes'
    
    return df


def nb_classification_plus(df, df_val, target, min_alpha=0.1, max_alpha=5.0):
    '''
    Perform Naive Bayes classification on the given data.

    Args:
        df (DataFrame): The training DataFrame containing the feature columns and the 
        target column.
        df_val (DataFrame): The validation (or test) DataFrame containing the feature 
        columns and the target column.
        target (str): The name of the target column.
        min_alpha: The minimum alpha to test
        max_alpha: The maximum alpha to test
    Returns:
        DataFrame: A DataFrame containing the combinations of hyperparameters and their 
        corresponding accuracy scores on the training and validation data. An 'algorithm' 
        column is added to denote naive_bayes.
    '''
    # reset indices after split
    df = df.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df.compound_sentiment = df.compound_sentiment + 1
    df_val.compound_sentiment = df_val.compound_sentiment + 1
    
    # define and split features and target
    X_train = df['lemmatized']
    X_validate = df_val['lemmatized']
    y_train = df[target]
    y_validate = df_val[target]
    
    # TFIDF vectorize the lemmatized text corpus
    tfidf = TfidfVectorizer()
    X_train = tfidf.fit_transform(X_train)
    X_validate = tfidf.transform(X_validate)
    
    # make vectorized matrix dense and concat other features
    X_cols_train = df[['word_count', 'stopword_ratio', 'compound_sentiment', 'percent_unique']]
    X_cols_validate = df_val[['word_count', 'stopword_ratio', 'compound_sentiment', 'percent_unique']]
    X_train = pd.concat(
        [X_cols_train, pd.DataFrame(
            X_train.todense(), columns=tfidf.get_feature_names())], axis=1)
    X_validate = pd.concat(
        [X_cols_validate, pd.DataFrame(
            X_validate.todense(), columns=tfidf.get_feature_names())], axis=1)
    
    nb_models = {}
    hyperparams = 'default'
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    preds = nb.predict(X_train)
    accuracy_train = nb.score(X_train, y_train)
    accuracy_val = nb.score(X_validate, y_validate)
    nb_models[hyperparams] = accuracy_train, accuracy_val
   
    df = pd.DataFrame([{'hyperparams': k, 'accuracy_train': v[0],
                          'accuracy_val': v[1]} for k, v in nb_models.items()])
    df['algorithm'] = 'multinomial_naive_bayes'
    
    return df



def model_data(df, df_val, target):
    '''
    This function takes in a train and test dataset and models the data
    with multiple random forest, decision tree, and KNN models of a range
    of hyperparameters. It also uses applies a Multinomial Naive Bayes model.
    The function takes the best performing model of each type based on
    accuracy of the validation predictions and returns a dataframe of the 
    results.
    
    Arguments:
        df: the training dataframe
        df_val: the validation (or test) dataframe.
        target(str): the string literal of the target column.
        
    Returns:
        A dataframe of the best result from each type of model with it's
        hyperparameters and accuracy score.
    '''
    top_rf = (rf_classification(df=df, df_val=df_val, target=target)).sort_values(
        'accuracy_val', ascending=False).head(1)
    top_dt = dt_classification(df=df, df_val=df_val, target=target).sort_values(
        'accuracy_val', ascending=False).head(1)
    top_knn = knn_classification(df=df, df_val=df_val, target=target).sort_values(
        'accuracy_val', ascending=False).head(1)
    top_nb = nb_classification(df=df, df_val=df_val, target=target)
    df = pd.concat([top_rf, top_dt, top_knn, top_nb], axis=0, ignore_index=True)

    return df


def model_data_plus(df, df_val, target):
    '''
    This function takes in a train and test dataset and models the data
    with multiple random forest, decision tree, and KNN models of a range
    of hyperparameters. It also uses applies a Multinomial Naive Bayes model.
    The function takes the best performing model of each type based on
    accuracy of the validation predictions and returns a dataframe of the 
    results.
    
    Arguments:
        df: the training dataframe
        df_val: the validation (or test) dataframe.
        target(str): the string literal of the target column.
        
    Returns:
        A dataframe of the best result from each type of model with it's
        hyperparameters and accuracy score.
    '''
    top_rf = (rf_classification_plus(df=df, df_val=df_val, target=target)).sort_values(
        'accuracy_val', ascending=False).head(1)
    top_dt = dt_classification_plus(df=df, df_val=df_val, target=target).sort_values(
        'accuracy_val', ascending=False).head(1)
    top_knn = knn_classification_plus(df=df, df_val=df_val, target=target).sort_values(
        'accuracy_val', ascending=False).head(1)
    top_nb = nb_classification_plus(df=df, df_val=df_val, target=target)
    df = pd.concat([top_rf, top_dt, top_knn, top_nb], axis=0, ignore_index=True)

    return df
    
    
def best_model_on_test(df, df_validate, df_test, target, n_neighbors=[16]):
    '''
    Perform KNN classification on the test data with the best hyperparameters from 
    the validate set .

    Args:
        df (DataFrame): The training DataFrame containing the feature columns 
        and the target column.
        df_val (DataFrame): The test DataFrame containing the feature 
        columns and the target column.
        target (str): The name of the target column.
        n_neighbors: the k-number of neighbors to test
    Returns:
        DataFrame: A DataFrame containing the hyperparameters and their 
        corresponding accuracy scores on the training and test data. An 
        'algorithm' column is added to denote knn.
    '''
    # define and split features and target
    X_train = df['lemmatized']
    y_train = df[target]
    X_validate = df_validate['lemmatized']
    y_validate = df_validate[target]
    X_test = df_test['lemmatized']
    y_test = df_test[target]
    
    # TFIDF vectorize the lemmatized text corpus
    tfidf = TfidfVectorizer()
    X_train = tfidf.fit_transform(X_train)
    X_validate = tfidf.transform(X_validate)
    X_test = tfidf.transform(X_test)
    
    knn_models = {}
    hyper_list = n_neighbors
    
    for hyperparams in hyper_list:
        knn = KNeighborsClassifier(n_neighbors=hyperparams)
        knn.fit(X_train, y_train)
        preds = knn.predict(X_train)
        accuracy_train = knn.score(X_train, y_train)
        accuracy_validate = knn.score(X_validate, y_validate)
        accuracy_test = knn.score(X_test, y_test)
        knn_models[hyperparams] = accuracy_train, accuracy_validate, accuracy_test
   
    df = pd.DataFrame([{'hyperparams': k, 'accuracy_train': v[0],
                          'accuracy_validate': v[1], 'accuracy_test': v[2]} for k, v in knn_models.items()])
    df['algorithm'] = 'knn'
    
    return df