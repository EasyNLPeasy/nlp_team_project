import wrangle as w

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import nltk
import unicodedata
import re
from re import match
from wordcloud import WordCloud
import nltk.sentiment

train, validate, test = w.split_data()

def train_by_language(): 

    python = train[train.language == 'Python']
    java = train[train.language == 'JavaScript']
    rust = train[train.language == 'Rust']

    return python, java, rust

def clean(text):
    'A simple function to cleanup text data'
    wnl = nltk.stem.WordNetLemmatizer()
    stopwords = nltk.corpus.stopwords.words('english')
    text = (unicodedata.normalize('NFKD', text)
             .encode('ascii', 'ignore')
             .decode('utf-8', 'ignore')
             .lower())
    words = re.sub(r'[^\w\s]', '', text).split()
    return [wnl.lemmatize(word) for word in words if word not in stopwords]

def words_by_language():

    python, java, rust = train_by_language()
    python_words = clean(' '.join(python['lemmatized']))
    java_words = clean(' '.join(java['lemmatized']))
    rust_words = clean(' '.join(rust['lemmatized']))
    all_words = clean(' '.join(train['lemmatized']))
    return python_words, java_words, rust_words, all_words

def get_word_count_frequency_df():
    python_words, java_words, rust_words, all_words = words_by_language()
    python_freq = pd.Series(python_words).value_counts()
    java_freq = pd.Series(java_words).value_counts()
    rust_freq = pd.Series(rust_words).value_counts()
    all_freq = pd.Series(all_words).value_counts()
    
    all_freqs = pd.concat([python_freq, java_freq, rust_freq, all_freq], 
                      axis=1).fillna(0).astype(int)
    all_freqs.columns =['python', 'java', 'rust', 'all']
    return all_freqs

def viz_top_word_freqs():
    # Visualization of 15 Most Frequently Occurring Words
    all_freqs = get_word_count_frequency_df()
    colors = ['pink', 'plum', 'purple']
    all_freqs.sort_values('all', ascending=False)[['python', 'java', 'rust']].head(15).plot.bar(ec='black', color=colors).set(title='15 Most Frequent Words by Language')
    plt.ylabel('Count of Word Usage')
    plt.xticks(rotation=45)
    plt.show()

def viz_proportional_word_freq():

    # Visualization of Top 20 Frequently Used Words - Proportionally Calculated
    all_freqs = get_word_count_frequency_df()
    colors = ['pink', 'plum', 'purple']
    all_freqs.sort_values('all', ascending=False
                       ).head(20).apply(
        lambda row: row/row['all'], axis=1
    )[['python', 'java', 'rust']].plot.barh(
        stacked=True, legend=False, ec='black', 
        width=1, color=colors).set(title='Proportions of the Top 20 Words')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.xlabel('Proportion of most Used Words')
    plt.ylabel('Most used Words')
    plt.show()

def get_only_alphabetical_words():
    python_words, java_words, rust_words, all_words = words_by_language()
    python_words_only = pd.Series(filter(lambda x: match(r'[^0-9]', x), python_words))
    java_words_only = pd.Series(filter(lambda x: match(r'[^0-9]', x), java_words))
    rust_words_only = pd.Series(filter(lambda x: match(r'[^0-9]', x), rust_words))
    all_words_only = pd.Series(filter(lambda x: match(r'[^0-9]', x), all_words))
    language_words_only = pd.concat([python_words_only, java_words_only, rust_words_only, all_words_only], axis=1)
    language_words_only.columns = ['python_words', 'java_words', 'rust_words', 'all_words']
    return language_words_only

def viz_words_freqs_by_language():
    all_freqs = get_word_count_frequency_df()
    for col in all_freqs.columns:
        all_freqs[col].sort_values(ascending=False).head(20).plot.barh(ec='black', color='pink')
        plt.title(f'Top 20 Words Used in {col} READMEs')
        plt.xlabel('Number of Occurances')
        plt.ylabel('Top Words Used')
        plt.show()

# Bigrams
def viz_python_bigrams():
    language_words_only = get_only_alphabetical_words()
    p = language_words_only[language_words_only['python_words'].isna() == False]
    pd.Series(nltk.bigrams(p['python_words'])).value_counts().head(20).plot.barh(ec='black', color='pink')
    plt.title(f'Top 20 Bigrams Used in Python READMEs')
    plt.xlabel('Number of Occurances')
    plt.ylabel('Top Bigrams Used')
    plt.show()

def viz_java_bigrams():
    language_words_only = get_only_alphabetical_words()
    p = language_words_only[language_words_only['java_words'].isna() == False]
    pd.Series(nltk.bigrams(p['java_words'])).value_counts().head(20).plot.barh(ec='black', color='pink')
    plt.title(f'Top 20 Bigrams Used in JavaScript READMEs')
    plt.xlabel('Number of Occurances')
    plt.ylabel('Top Bigrams Used')
    plt.show()

def viz_rust_bigrams():
    language_words_only = get_only_alphabetical_words()
    p = language_words_only[language_words_only['rust_words'].isna() == False]
    pd.Series(nltk.bigrams(p['rust_words'])).value_counts().head(20).plot.barh(ec='black', color='pink')
    plt.title(f'Top 20 Bigrams Used in Rust READMEs')
    plt.xlabel('Number of Occurances')
    plt.ylabel('Top Bigrams Used')
    plt.show()

def viz_all_bigrams():
    language_words_only = get_only_alphabetical_words()
    p = language_words_only[language_words_only['all_words'].isna() == False]
    pd.Series(nltk.bigrams(p['all_words'])).value_counts().head(20).plot.barh(ec='black', color='pink')
    plt.title(f'Top 20 Bigrams Used in All READMEs')
    plt.xlabel('Number of Occurances')
    plt.ylabel('Top Bigrams Used')
    plt.show()

# Trigrams

def viz_python_trigrams():
    language_words_only = get_only_alphabetical_words()
    p = language_words_only[language_words_only['python_words'].isna() == False]
    pd.Series(nltk.ngrams(p['python_words'], 3)).value_counts().head(20).plot.barh(ec='black', color='pink')
    plt.title(f'Top 20 Trigrams Used in Python READMEs')
    plt.xlabel('Number of Occurances')
    plt.ylabel('Top Trigrams Used')
    plt.show()

def viz_java_trigrams():
    language_words_only = get_only_alphabetical_words()
    p = language_words_only[language_words_only['java_words'].isna() == False]
    pd.Series(nltk.ngrams(p['java_words'], 3)).value_counts().head(20).plot.barh(ec='black', color='pink')
    plt.title(f'Top 20 Trigrams Used in JavaScript READMEs')
    plt.xlabel('Number of Occurances')
    plt.ylabel('Top Trigrams Used')
    plt.show()

def viz_rust_trigrams():
    language_words_only = get_only_alphabetical_words()
    p = language_words_only[language_words_only['rust_words'].isna() == False]
    pd.Series(nltk.ngrams(p['rust_words'], 3)).value_counts().head(20).plot.barh(ec='black', color='pink')
    plt.title(f'Top 20 Trigrams Used in Rust READMEs')
    plt.xlabel('Number of Occurances')
    plt.ylabel('Top Trigrams Used')
    plt.show()

def viz_all_trigrams():
    language_words_only = get_only_alphabetical_words()
    p = language_words_only[language_words_only['all_words'].isna() == False]
    pd.Series(nltk.ngrams(p['all_words'], 3)).value_counts().head(20).plot.barh(ec='black', color='pink')
    plt.title(f'Top 20 Trigrams Used in All READMEs')
    plt.xlabel('Number of Occurances')
    plt.ylabel('Top Trigrams Used')
    plt.show()

def viz_python_word_cloud():
    language_words_only = get_only_alphabetical_words()
    p = language_words_only[language_words_only['python_words'].isna() == False]
    img = WordCloud(colormap='magma', background_color='white', ).generate(' '.join(p['python_words']))
    plt.imshow(img)
    plt.axis('off')
    plt.title('Most Commonly Used Words in Python READMEs')
    plt.show()

def viz_java_word_cloud():
    language_words_only = get_only_alphabetical_words()
    p = language_words_only[language_words_only['java_words'].isna() == False]
    img = WordCloud(colormap='magma', background_color='white', ).generate(' '.join(p['java_words']))
    plt.imshow(img)
    plt.axis('off')
    plt.title('Most Commonly Used Words in JavaScript READMEs')
    plt.show()

def viz_rust_word_cloud():
    language_words_only = get_only_alphabetical_words()
    p = language_words_only[language_words_only['rust_words'].isna() == False]
    img = WordCloud(colormap='magma', background_color='white', ).generate(' '.join(p['rust_words']))
    plt.imshow(img)
    plt.axis('off')
    plt.title('Most Commonly Used Words in Rust READMEs')
    plt.show()

def viz_all_word_cloud():
    language_words_only = get_only_alphabetical_words()
    p = language_words_only[language_words_only['all_words'].isna() == False]
    img = WordCloud(colormap='magma', background_color='white', ).generate(' '.join(p['all_words']))
    plt.imshow(img)
    plt.axis('off')
    plt.title('Most Commonly Used Words in All READMEs')
    plt.show()

def viz_word_counts(train):

    # plot to visualize actual vs predicted models
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(4, 1, figsize=(15, 40))
    
    python_count = train[train['language'] == 'Python']
    java_count = train[train['language'] == 'JavaScript']
    rust_count = train[train['language'] == 'Rust']
    
    ax0.hist(python_count.word_count, color='violet', alpha=.5, edgecolor='black')
    ax1.hist(java_count.word_count, color='indigo', alpha=.5, edgecolor='black')
    ax2.hist(rust_count.word_count, color='purple', alpha=.5, edgecolor='black')
    ax3.hist(train.word_count, color='plum', alpha=.5, edgecolor='black')

    ax0.set_xticklabels(ax0.get_xticks(), rotation = 45)
    ax1.set_xticklabels(ax1.get_xticks(), rotation = 45)
    ax2.set_xticklabels(ax2.get_xticks(), rotation = 45)
    ax3.set_xticklabels(ax3.get_xticks(), rotation = 45)

    ax0.set_title("Distribution of Word Counts in Python READMEs")
    ax1.set_title("Distribution of Word Counts in JavaScript READMEs")
    ax2.set_title("Distribution of Word Counts in Rust READMEs")
    ax3.set_title("Distribution of Word Counts in All READMEs")
    plt.show()
