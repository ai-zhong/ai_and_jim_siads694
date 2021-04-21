from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from train_wordvecs import tokenize
import pandas as pd
import numpy as np
import pickle
from train_wordvecs import load_data
from tqdm import tqdm

stops = stopwords.words('english')


def load_external_resources():
    """
    load external files for later processing
    1. dale_chall.txt
    2. AoA_51715_words.csv
    3. Concreteness_ratings_Brysbaert_et_al_BRM.txt
    """
    # file 1
    with open('dale_chall.txt', 'r') as f:
        dale = f.readlines()
        f.close()

    dale = [x.strip() for x in dale]

    # file 2
    aoa = pd.read_csv('AoA_51715_words.csv', encoding='iso-8859-1')
    aoa = aoa[['Word', 'AoA_Kup_lem']]
    aoa_dict = aoa.set_index('Word').to_dict()['AoA_Kup_lem']

    # file 3
    with open('Concreteness_ratings_Brysbaert_et_al_BRM.txt', 'r') as f:
        concrete = f.readlines()
        f.close()

    concrete_df = pd.DataFrame([x.split('\t') for x in concrete])
    concrete_df.columns = concrete_df.iloc[0]
    concrete_df = concrete_df.iloc[1:]
    concrete_df = concrete_df.drop('Dom_Pos\n', axis=1)

    float_cols = ['Bigram', 'Conc.M', 'Conc.SD', 'Unknown', 'Total', 'Percent_known', 'SUBTLEX']
    concrete_df[float_cols] = concrete_df[float_cols].astype(float)

    return dale, aoa_dict, concrete_df


def syllable_count(word):
    """
    count number of syllables in a word
    """
    word = word.lower()
    count = 0
    vowels = "aeiouy"
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if count == 0:
        count += 1
    return count


def calc_dale_score(doc_tokens, doc_sent_cnt, dale_list):
    """
    doc_tokens is a list of tokens that have been preprocessed (lower case, removed stopwords).
    doc_sent_cnt: how many sentences in this document.
    dale_list is the criteria of 'simple words' that defined by official study.
    
    return a calculated score to imply the readability of this chunck of text
    
    (pdw: percentage of difficult words
    asl: average sentence length in words
    asw: average number of syllables per word
    ref: https://readabilityformulas.com/new-dale-chall-readability-formula.php)
    """
    simple_cnt = len([x for x in doc_tokens if x.lower() in dale_list])
    total_cnt = len(doc_tokens) * 1.0
    if total_cnt == 0:
        return 0, 0

    pdw = 1 - simple_cnt / total_cnt
    asl = total_cnt / doc_sent_cnt
    score = 0.1579 * pdw + 0.0496 * asl
    if pdw > 0.05:
        score = score + 3.6365

    text = ''.join(doc_tokens)
    sy_cnt = syllable_count(text)
    asw = sy_cnt / total_cnt
    # score2 = 206.835 - 1.015*asl-84.6*asw

    return (score, asw)


def get_aoa_score(list_of_tokens, aoa_dict):
    """
    AoA: early (simpler) words have lower score
    calculate the average score for a list of cleaned tokens
    """
    scores = []
    for token in list_of_tokens:
        if token.lower() in aoa_dict:
            scores.append(aoa_dict[token.lower()])

    if scores:
        return np.nanmean(scores)

    else:
        return np.nan


def calculate_scores(df, lemma=False):
    """
    generate scores for dale_chall, syllables count per word, and aoa.
    df has a column named 're_tokened'

    return df with 3 new columns of scores
    """
    dale, aoa_dict, _ = load_external_resources()

    df['sentence_cnt'] = df.original_text.apply(lambda x: len(sent_tokenize(x)))

    if 're_tokened' not in df.columns:
        try:
            if not lemma:
                re_tokened = pickle.load(open('re_tokenized_nolemma.pkl', 'rb'))
            if lemma:
                re_tokened = pickle.load(open('re_tokenized_lemma.pkl', 'rb'))
            df['re_tokened'] = re_tokened
        except:
            if not lemma:
                print('tokenizing original text without lemmatization')
                df['re_tokened'] = tokenize(df, lemmatize=False)
            if lemma:
                df['re_tokened'] = tokenize(df, lemmatize=True)

        scores1 = []
        scores2 = []
        aoa_scores = []
        for i in tqdm(range(len(df))):
            item = df.iloc[i]
            score1, score2 = calc_dale_score(item['re_tokened'], item['sentence_cnt'], dale)
            scores1.append(score1)
            scores2.append(score2)

            score = get_aoa_score(item['re_tokened'], aoa_dict)
            aoa_scores.append(score)

        df['dale_chall_score'] = scores1
        df['syllable_per_word'] = scores2
        df['aoa'] = aoa_scores

        return df


if __name__ == '__main__':
    df = load_data()
    # df = calculate_scores(df)
    # df[['dale_chall_score']].to_csv('WikiLarge_Train_dale_chall_score.csv')
    # df[['syllable_per_word']].to_csv('WikiLarge_Train_syllable_per_w.csv')
    # df[['aoa']].to_csv('WikiLarge_Train_aoa.csv')

    # df = calculate_scores(df, lemma=True)
    # df[['dale_chall_score']].to_csv('WikiLarge_Train_dale_chall_score2.csv')
    # df[['syllable_per_word']].to_csv('WikiLarge_Train_syllable_per_w2.csv')
    # df[['aoa']].to_csv('WikiLarge_Train_aoa2.csv')
