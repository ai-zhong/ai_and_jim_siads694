import re
from tqdm import tqdm
from gensim.models.word2vec import Word2Vec
from nltk.corpus import stopwords
import random
import pickle
import pandas as pd


def load_data(filename='WikiLarge_Train.csv', read_partial=False, s=100000):
    """
    if read_partial, then only part of the data is read
    s is the length of sample you want to take if read_partial==True
    """

    if read_partial:
        # n is the total length of the file
        n = sum(1 for line in open(filename)) - 1
        skip = sorted(random.sample(range(1, n + 1), n - s))
        df = pd.read_csv(filename, skiprows=skip)
        return df

    df = pd.read_csv(filename)

    return df


def tokenize(df, lemmatize=False):
    """
    some preprocessing before training word2vec
    note we do NOT lower case word here
    """
    stops = stopwords.words('english')

    re_tokenized = []
    if not lemmatize:
        for text in tqdm(df.original_text):
            re_tokenized.append([x for x in re.findall(r'(\w+)', text)
                                 if x.lower() not in stops and len(x) > 1])
        pickle.dump(re_tokenized, open('re_tokenized_nolemma.pkl', 'wb'))

    if lemmatize:
        import spacy
        nlp = spacy.load('en', disable=['parser', 'ner'])
        for text in tqdm(df.original_text):
            text = nlp(text)
            re_tokenized.append([token.lemma_ for token in text
                                 if str(token).lower() not in stops and len(str(token)) > 1
                                 and not str(token.lemma_).startswith('-')])
        pickle.dump(re_tokenized, open('re_tokenized_lemma.pkl', 'wb'))

    return re_tokenized


def train_wordvec(re_tokenized, vec_size=100, rand_seed=42):
    quick_model = Word2Vec(sentences=re_tokenized,
                           size=vec_size,
                           min_count=200,
                           workers=4,
                           seed=rand_seed)
    word_vecs = quick_model.wv

    return word_vecs


if __name__ == "__main__":
    df = load_data()
    # re_tokenized = tokenize(df)
    # word_vecs = train_wordvec(re_tokenized)
    # pickle.dump(word_vecs, open('word_vecs_train.pkl', 'wb'))

    re_tokenized = tokenize(df, lemmatize=True)
    word_vecs = train_wordvec(re_tokenized)
    pickle.dump(word_vecs, open('word_vecs_train_lemma.pkl', 'wb'))
