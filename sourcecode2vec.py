import gensim
from gensim.models import Word2Vec
import pandas as pd
import spacy
# data visualization libraries
from sklearn.decomposition import PCA
from matplotlib import pyplot

df = pd.read_csv('data/source_code_dataset.csv', error_bad_lines=False) # read the csv file
nlp = spacy.load("en_core_web_sm", disable=['ner', 'parser']) # disable named entity recogition for speed

# train model
model = Word2Vec(df, min_count=1)
# summarize the loaded model
print(model)
# summarize vocabulary (distinct words in sentences)
words = list(model.wv.vocab)
print(words)
# access vector for one word 'sentence'
# print(model['sentence'])
# save model
model.save('model.bin')
# load model
new_model = Word2Vec.load('model.bin')
print(new_model)
# visualize  word representations using principal component analysis (pca)
X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
# create a scatter plot of the projection
pyplot.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
    pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()