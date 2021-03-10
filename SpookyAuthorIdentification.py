import csv
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem.snowball import SnowballStemmer
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer,  TfidfVectorizer
from sklearn.model_selection import train_test_split, KFold

import matplotlib
from matplotlib import pyplot as plt

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer


from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GridSearchCV


data = pd.read_csv("train.csv") 
data2=pd.DataFrame(data)


import string
stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
stemmer = SnowballStemmer("english")
def clean(doc):
    punc_free = ''.join(ch for ch in doc if ch not in exclude)
    # ADJ, ADJ_SAT, ADV, NOUN, VERB = 'a', 's', 'r', 'n', 'v'
    # POS_LIST = [NOUN, VERB, ADJ, ADV]
    # lemma.lemmatize("rocks") = rock
    # lemma.lemmatize("better", pos="a") = good
    normalized_1 = " ".join(lemma.lemmatize(word, 'a') for word in punc_free.split()) 
    normalized_2 = " ".join(lemma.lemmatize(word, 'n') for word in normalized_1.split()) #this deleted s from as
    normalized_3 = " ".join(stemmer.stem(word) for word in normalized_2.split()) #this is very effective
    stop_free = " ".join([i for i in normalized_3.lower().split() if i not in stop])
    return stop_free


# clean(data['text'][0])  
data['text_clean'] = data['text'].apply(clean) #clean(data['text'][0]) 

# # Re-tokenize: 
#TfidfVectorizer() requires data non-tokenized
# tokenizer = RegexpTokenizer(r'\w+')
# data['text_tokens'] = data['text_clean'].apply(tokenizer.tokenize)
# all_nose_words = [word for tokens in data['text_tokens'] for word in tokens]


#input matrix is a list of all words in All Corpus (col) with tfidf value in each document (row)
#tfidf is high with a high term frequency in the document (or 1 if present) and a low document frequency of the term in the whole corpus (dividing the total number of documents by the number of documents containing the term), boolean, frequentistic or normalized formula
def cv(data):
    """
    Count vectorizing function.  Returns embedded vector and vectorizer.
    """
    #count_vectorizer = CountVectorizer() # Bag-of-words vectorization
    count_vectorizer = TfidfVectorizer() # Term Frequency-Inverse Document Frequency vectorization
    #count_vectorizer = CountVectorizer(ngram_range =(1, 2)) # n-gram vectorization  
    emb = count_vectorizer.fit_transform(data)
    # print(count_vectorizer.get_feature_names())
    count_vectorizer.fit(data)
    dictionary=count_vectorizer.vocabulary_.items() 
    return emb, dictionary

def barplotTopWords(vocab_):
    # count= []
    # vocab= []
    # for key, value in dictionary:
    #     vocab.append(key)
    #     count.append(value)
    # vocab_ = pd.Series(count, index=vocab)
    # vocab_ =vocab_.sort_values(ascending=False)
    top_vocab = vocab_.head(20)
    top_vocab.plot(kind = 'barh', figsize=(5,10), xlim= (14845, 14866))
    # matplotlib.pyplot.barh(top_vocab.index, top_vocab)
    plt.grid()
    plt.show()
    return vocab_ 



from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, pairwise_distances

def gen_cos_sim(df):     
    # Vectorize nose notes
    emb, dictionary = cv(df.tolist())
    return cosine_similarity(emb)


#cosine similarity matrix is DocumentXDocument shape, tells how much a document is similar to another (rows in the input tfidf sparse matrix)
# cos_sim= gen_cos_sim(data['text_clean'])
# EAP_data = data[data['author'] == 'EAP']

emb, dictionary = cv(data['text_clean'].tolist())

#multinomial and bernoulli NB classifier
def gen_NaiveBayesClassifier(data, emb, classifier):     
    # freq= pd.Series((' '.join(data['text']).split())).valuecounts().sortvalues(ascending = False)
    # vocab_ = barplotTopWords(freq)
    array = emb.todense()
    emb2= pd.DataFrame(array)
    emb2['output'] = data['author']
    emb2['id'] = data['id']

    features = emb2.columns.tolist()
    output = 'output'
    # removing the output and the id from features
    features.remove(output)
    features.remove('id')

    #list of alpha parameters we are going to try
    alpha_list1 = np.linspace(0.006, 0.1, 2)
    alpha_list1 = np.around(alpha_list1, decimals=4)

    parameter_grid =[{"alpha": alpha_list1}]

    classifier1= classifier() #MultinomialNB, BernoulliNB
    # GridSearchCV allows to tune parameters of a model through k-fold cross validation using parameter grid
    # gridsearch object using 4 fold cross validation and neg_log_loss as scoring paramter
    gridsearch1 = GridSearchCV(classifier1,parameter_grid, scoring = 'neg_log_loss', cv = 2)
    my_model= gridsearch1.fit(emb2[features], emb2[output])
    return my_model

gridsearch1 = gen_NaiveBayesClassifier(data, emb, MultinomialNB)

instance= emb.todense()[0]
instance2= pd.DataFrame(instance)
instance2

def predict_NaiveBayesClassifier(instance, gridsearch1):
    
    results1 = pd.DataFrame()
    # collect alpha list
    results1['alpha'] = gridsearch1.cv_results_['param_alpha'].data
    # collect test scores
    results1['neglogloss'] = gridsearch1.cv_results_['mean_test_score'].data

    matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
    plt.plot(results1['alpha'], -results1['neglogloss'])

    "Best parameter: ", gridsearch1.best_params_
    "Best score: ",gridsearch1.best_score_ 
    gridsearch1.best_estimator_  
#     my_model.best_estimator_.feature_importances #MultinomialNB' object has no attribute 'feature_importances'

    prediction= gridsearch1.predict(instance)

    return prediction, gridsearch1.best_params_, gridsearch1.best_score_ , gridsearch1.best_estimator_  


prediction, best_params, best_score, best_estimator = predict_NaiveBayesClassifier(instance2, gridsearch1)


# best_params_= best_params
# best_score_=best_score 
# sorted(zip(best_params_, best_score_), reverse=True)


#NOW VERIFY THE MODEL ON TEST SET
