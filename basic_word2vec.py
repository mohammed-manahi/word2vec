# this implementation is from tutorial available at: https://www.kaggle.com/pierremegret/gensim-word2vec-tutorial

import re # for regex text preprocessing 
import pandas as pd # for data handling lib
from time import time # for time operations
from collections import defaultdict # for word frequency 
import spacy # for preprocessing
import logging # for loggers
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)

from gensim.models.phrases import Phrases, Phraser # gensim phrases for bigram 

import multiprocessing # for concurrency 
from gensim.models import Word2Vec # word2vec model

import numpy as np # for math
import matplotlib.pyplot as plt # for plot representation
import seaborn as sns # for visualization
sns.set_style("darkgrid")
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

df = pd.read_csv('data/simpsons_dataset.csv', error_bad_lines=False) # read the csv file
print(df.shape) # print file dimensions
print(df.head()) # get data of first 5 rows 
print(df.isnull().sum()) # get sum of null data
df = df.dropna().reset_index(drop=True) # remove missing values 
print(df.isnull().sum())

nlp = spacy.load("en_core_web_sm", disable=['ner', 'parser']) # disable named entity recogition for speed

def cleaning(doc):
    # lemmatizes and removes stopwords
    txt = [token.lemma_ for token in doc if not token.is_stop]
    if len(txt) > 2:
        return ' '.join(txt)

brief_cleaning = (re.sub("[^A-Za-z']+", ' ', str(row)).lower() for row in df['spoken_words']) # remove non-alphabatic characters
t = time()
txt = [cleaning(doc) for doc in nlp.pipe(brief_cleaning, batch_size=5000, n_threads=-1)] # spacy pipeline sppeds up cleaning process
print('Time to clean up everything: {} mins'.format(round((time() - t) / 60, 2)))

# pandas dataframe is two dimensional table to remove unnecessary data. 
df_clean = pd.DataFrame({'clean': txt})
df_clean = df_clean.dropna().drop_duplicates()
print(df_clean.shape)

# apply bigram form n-grams model to catch words like mr_burns
sent = [row.split() for row in df_clean['clean']]
bigram = Phrases(sent, min_count=30, progress_per=10000)
sentences = bigram[sent]

# word frequencies after lemmatization, stop words removal and bigram
word_freq = defaultdict(int)
for sent in sentences:
    for i in sent:
        word_freq[i] += 1
print(len(word_freq))
print(sorted(word_freq, key=word_freq.get, reverse=True)[:10])

# count nimber of cpu cores
cores = multiprocessing.cpu_count()

# specifies word2vec params
w2v_model = Word2Vec(min_count=20, window=2, size=300, sample=6e-5, alpha=0.03, min_alpha=0.0007,  negative=20, workers=cores-1)

# building vocabulary 
t = time()
w2v_model.build_vocab(sentences, progress_per=10000)
print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

# training word2vec model
t = time()
w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1) # epochs is number of iterations over corpus 
print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

# init_sims make memory more effecient if called it indicates that there is no further training 
w2v_model.init_sims(replace=True)

# word2vec similarities
# most similar words
print(w2v_model.wv.most_similar(positive=["homer"]))
# most similar words in bigram model
print(w2v_model.wv.most_similar(positive=["homer_simpson"]))
# two words similarity 
print(w2v_model.wv.similarity('maggie', 'baby'))
# analogy difference
print(w2v_model.wv.most_similar(positive=["woman", "homer"], negative=["marge"], topn=3))

def tsne_plot(model):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()

tsne_plot(w2v_model)