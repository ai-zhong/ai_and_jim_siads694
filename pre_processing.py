from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from nltk import pos_tag


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


def tokenize(df, lemmatize=True):
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
        # save for later import
        pickle.dump(re_tokenized, open('re_tokenized_lemma.pkl', 'wb'))

    return re_tokenized


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
    aoa_df = pd.read_csv('AoA_51715_words.csv', encoding='iso-8859-1')
    # aoa = aoa[['Word', 'AoA_Kup_lem']]
    # aoa_dict = aoa.set_index('Word').to_dict()['AoA_Kup_lem']

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

    return dale, aoa_df, concrete_df


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


def get_score_from_dict(list_of_tokens, score_dict):
    """
    Calculate the average score for a list of cleaned tokens based on different score dictionaries
    """
    scores = []
    for token in list_of_tokens:
        if token.lower() in score_dict:
            scores.append(score_dict[token.lower()])
    if scores:
        return np.nanmean(scores)
    else:
        return np.nan


def get_avg_word_length(list_of_tokens):
    """
    Calculate the average token length for a list of cleaned tokens in a sentence.
    """
    lengths = [len(token) for token in list_of_tokens]
    if len(lengths) > 0:
        return np.nanmean(lengths)
    else:
        return np.nan


def calculate_scores(df, lemma=False, token_file='re_tokenized_', train_or_test='Train'):
    """
    Calculate 11 numerical scores and save each as a csv file.
    df has a column named 're_tokened'
    """
    dale, aoa_df, concrete_df = load_external_resources()

    aoa_dict1 = aoa_df.set_index('Word').to_dict()['AoA_Kup_lem']
    
    concrete_dict1 = concrete_df.set_index('Word').to_dict()['Percent_known']
    concrete_dict2 = concrete_df.set_index('Word').to_dict()['Conc.M']
    concrete_dict3 = concrete_df.set_index('Word').to_dict()['SUBTLEX']
    concrete_dict4 = concrete_df.set_index('Word').to_dict()['Total']

    df['sentence_cnt'] = df.original_text.apply(lambda x: len(sent_tokenize(x)))

    if 're_tokened' not in df.columns:
        try:
            if not lemma:
                re_tokened = pickle.load(open(f'{token_file}nolemma.pkl', 'rb'))
            if lemma:
                re_tokened = pickle.load(open(f'{token_file}lemma.pkl', 'rb'))
            df['re_tokened'] = re_tokened
        except:
            if not lemma:
                print('tokenizing original text without lemmatization')
                df['re_tokened'] = tokenize(df, lemmatize=False, save_as=f'{token_file}nolemma.pkl')
            if lemma:
                df['re_tokened'] = tokenize(df, lemmatize=True, save_as=f'{token_file}lemma.pkl')

        # 9 scores
        dale_chall_scores = []
        asw_scores = []
        avg_word_lens = []
        cnt_verbs_all = []

        for i in tqdm(range(len(df))):
            item = df.iloc[i]

            score1, score2 = calc_dale_score(item['re_tokened'], item['sentence_cnt'], dale)
            dale_chall_scores.append(score1)
            asw_scores.append(score2)
            avg_word_lens.append(get_avg_word_length(item['re_tokened']))

            # count how many verbs appeared in this doc without removing stopwords
            tags = pos_tag(word_tokenize(df['original_text'].iloc[i]))
            cnt_verbs = sum([x[-1][0]=='V' for x in tags])
            cnt_verbs_all.append(cnt_verbs)


        df['dale_chall_score'] = dale_chall_scores
        df['syllable_per_word'] = asw_scores
        df['concrete_score'] = df['re_tokened'].apply(lambda x: get_score_from_dict(x, concrete_dict1))
        df['conc_mean'] = df['re_tokened'].apply(lambda x: get_score_from_dict(x, concrete_dict2))
        df['subtlex'] = df['re_tokened'].apply(lambda x: get_score_from_dict(x, concrete_dict3))
        df['conc_total'] = df['re_tokened'].apply(lambda x: get_score_from_dict(x, concrete_dict4))
        df['average_word_len'] = avg_word_lens
        df['aoa'] = df['re_tokened'].apply(lambda x: get_score_from_dict(x, aoa_dict1))
        df['verb2'] = cnt_verbs_all

        df['word_cnt'] = df['re_tokened'].apply(len)

        # ============================================
        # save results as csvs for later reloading
        df[['dale_chall_score']].to_csv(f'WikiLarge_{train_or_test}_dale_chall_score2.csv')
        df[['syllable_per_word']].to_csv(f'WikiLarge_{train_or_test}_syllable_per_w2.csv')
        df[['concrete_score']].to_csv(f'WikiLarge_{train_or_test}_concrete_score2.csv')
        df[['conc_mean']].to_csv(f'WikiLarge_{train_or_test}_conc_mean_score2.csv')
        df[['subtlex']].to_csv(f'WikiLarge_{train_or_test}_conc_subtlex_score2.csv')
        df[['conc_total']].to_csv(f'WikiLarge_{train_or_test}_conc_total_scores2.csv')

        df[['average_word_len']].to_csv(f'WikiLarge_{train_or_test}_avg_word_len2.csv')
        df[['aoa']].to_csv(f'WikiLarge_{train_or_test}_aoa2.csv')
        df[['verb2']].to_csv(f'WikiLarge_{train_or_test}_verb_cnts2.csv')

        df[['word_cnt']].to_csv(f'WikiLarge_{train_or_test}_word_count2.csv')



def load_df_and_features(path, train=True):
    """
    Load the calculated features along with the dataset
    :param path: file path to where the files are
    :param train: whether it's training dataset. If =False, load for testing dataset.
    :return: return df with loaded features
    """
    import os
    import glob
   
    if train:
        feature_files = glob.glob("WikiLarge_Train_*2.csv")
        features = []
        for f in feature_files:
            temp = pd.read_csv(f, index_col=0)
            features.append(temp)
        
        feature_df = pd.concat(features, axis=1)
        # fill na with mean
        feature_df = feature_df.fillna(feature_df.mean())

        df = load_data(filename='WikiLarge_Train.csv', read_partial=False)
        feature_df = pd.concat([df, feature_df], axis=1)
        
        chosen_features = ['aoa', 'concrete_score','verb2', 'conc_unknown', 'aoa_perc_known_lem',
        'conc_total', 'syllable_per_word', 'conc_mean_score','dale_chall_score', 'conc_subtlex_score']
        chosen_cols = ['original_text', 'label']+chosen_features

        feature_df = feature_df[chosen_cols]

        return feature_df


if __name__ == '__main__':
    test = load_df_and_features(path='')
    print(test.head())
    print(test.columns)
