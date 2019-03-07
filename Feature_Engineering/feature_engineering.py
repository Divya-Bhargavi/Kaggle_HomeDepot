from collections import defaultdict, Counter
from itertools import repeat
from Levenshtein import distance
import lzma
import math
from math import sqrt
from matplotlib import pyplot as plt
from multiprocessing import *
import numpy as np
import nltk
from nltk.stem.porter import *
import operator
import os
import pandas as pd
import re
from scipy import spatial
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.feature_extraction import stop_words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline, FeatureUnion
import string
import sys
import time
import xml.etree.cElementTree as ET
import zipfile

def tokenize(text):
    """
    Tokenize text and return a non-unique list of tokenized words
    found in the text. Normalize to lowercase, strip punctuation,
    remove stop words, drop words of length < 3, strip digits.

    :param text:            a string
    :returns:               the same string stripped of numbers,
                            tabs, newline characters, and punctuation
    """
    stops = list(stop_words.ENGLISH_STOP_WORDS)
    text = text.lower()
    regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')
    # delete stuff but leave at least a space to avoid clumping together
    nopunct = regex.sub(" ", text)
    words = nopunct.split(" ")
    # ignore a, an, to, at, be, ...
    words = [w for w in words if (len(w) > 2 and (w not in stops))]
    return words

def stemmed(words):
    """
    Stem a tokenized text and return a non-unique list of stemmed words
    found in the text. This is based on the output of function
    tokenize(text).

    :param text:            a list of tokenized words
    :returns:               a list of stemmed words
    """
    stemmer = PorterStemmer()
    return [stemmer.stem(w) for w in words]

def tokenizer(text):
        return stemmed(tokenize(text))

def make_dictionary(file):
    '''
    Initiate the glove model as a dictionary
    input: A String which is a file in the project directory
    returns: A dictionary with item = word : 300 d list

    :param file:            the filepath string of the dictionary
    :returns:               a dictionary with words as keys 
                            and 300d vectors as values
    '''
    vecs = defaultdict(lambda: np.zeros(shape=(300, 1)))
    with open(file) as f:
        lines = f.readlines()
        for word_and_vec in lines:
            elems = word_and_vec.strip().split(' ')
            word = elems[0]
            vec = np.array(elems[1:], dtype=float)
            vecs[word] = vec
    return vecs

def split_dictionary(glove_file):
    """
    firstly, I split the dictionary into a wordlist and a matrix.
    returns a list of words and 
    a 2d matrix of the normalized word vectors

    :returns:               the words and matrix associated with
                            the glove dictionary
    """
    wordlist = []
    matrix = []
    with open(glove_file) as f:
        lines = f.readlines()
        for word_and_vec in lines:
            wordvec = np.array([float(x) for x in word_and_vec.split()[1:]])
            matrix.append(wordvec / np.linalg.norm(wordvec))
            wordlist.append(word_and_vec.split()[0])
        matrix = np.array(matrix)
    return wordlist, matrix


def unique_words(cleaned_terms):
    """
    I then obtain the unique words that appear in the search_term.

    :param train_df:        the training set Pandas dataframe
    :returns:               a list of unique words from search terms
                            that have been stripped of numbers, symbols, etc.
    """
    cleaned = list(cleaned_terms)
    all_words = []
    for t in cleaned:
        all_words += t.split(' ')

    return list(set(all_words))[1:]


def find_nearest_neighbors(filename, cleaned_set, matrix, wordlist, dictionary):
    """
    here I count the cos_distance of each word that is in the cleaned_set.
    the output file looks like (each line): w0, w1, w2, w3, w4,
    i didn't print the distance, just the neighbour words
    this will take couple of minutes.

    :param filename:        a string representing the filename to write to
    :param clenaed_set:     a list of search terms that have 
                            been stripped of numbers, symbols, etc.
    :param matrix:          a 2d Numpy array of the word vectors in wordlist
    :param wordlist:        a list of words from the glove dictionary
    :param dictionary:      a dictionary with words as keys 
                            and 300d vectors as values
    """
    output_string = ''

    for word in cleaned_set:
        dots = matrix.dot(dictionary[word])
        close_index_vec = np.argsort(dots)
        for i in range(5):
            output_string += wordlist[int(close_index_vec[-1-i])] + ','
        output_string += '\n'

    f = open(filename, "w")
    f.write(output_string)
    f.close()


def get_all_terms_neighbors(dictionary, cleaned):
    """
    terms_neighbour is the list which stores the top 4 neighbours of each searching_terms. 
    for example, if the searching term is: cleaned[0]='w1_w2', 
    then the terms_neighbour[0]='n11_n12_n13_n14_n21_n22_n23_n24'.

    :param dictionary:      a dictionary
    :param cleaned:         a list of search terms that have
                            been stripped of numbers, symbols, etc.
    :returns:               a list of concatenated words that are neighbors
                            of the 'cleaned' terms
    """
    terms_neighbour = []
    for i in range(len(cleaned)):
        neighbours = ''
        if cleaned[i] != '':
            words = cleaned[i].split(' ')
            for w in words:
                neighbours = neighbours + dictionary[w] + ' '
        terms_neighbour.append(neighbours)
    return terms_neighbour


def build_dictionary(file):
    """
    based on the above output file, I then built a dictionary;
    this dictionary stores each word (as key) 
    with its top 4 neighbour words (as value) 

    :param file:            the file containing the list of strings of neighbors
    :returns:               a dictionary with words as keys 
                            and 4 neighbors of that word as values
    """
    k_dic = defaultdict(lambda: '')
    with open(file) as f:
        lines = f.readlines()
        for line in lines:
            words = line.strip().split(',')
            k_dic[words[0]] = words[1] + ' ' + \
                words[2] + ' ' + words[3] + ' ' + words[4]
    return k_dic


def clean_term_in_doc(terms, title):
    """
    This cleans the given terms in the specified document

    :param terms:           a list of unique search terms
    :param title:           a list of titles of products
    :return:                a list of the counts of the 
                            cleaned terms within a product's title
    """
    count = np.zeros(len(terms))
    terms = list(terms)
    title = list(title)
    for i in range(len(terms)):
        if not pd.isnull(terms[i]):
            title[i] = title[i].lower()
            for term in terms[i].split(' '):
                if term in title[i].split(' '):
                    count[i] += 1
    return pd.DataFrame(count)


def get_length(column):
    """
    This calculates and returns the number of words
    for each row in a specified column

    :param column:          the feature/attribute which
                            will have its words counted
    :returns:               a column with the count of 
                            words in each string
    """
    length = np.zeros(len(column))
    for index in range(len(column)):
        if not pd.isnull(column[index]):
            length[index] = len(column[index].split(' '))
    return length

def lemmatized(words):
    """
    lemmatize a tokenized text and return a non-unique list of stemmed words
    found in the text. This is based on the output of function
    tokenize(text).

    :param text:            a list of tokenized words
    :returns:               a list of lemmatized words
    """
    lemmatized_words = [nltk.stem.WordNetLemmatizer().lemmatize(w)
                        for w in words]
    return lemmatized_words

def attrib_stack(attributes, filepath_to_save_to):
    """
    Aggregate all the features of a product into a single description
    and return a dataframe with product id and description that is tokenized.
    """
    attributes['value'] = attributes['value'].apply(lambda x: str(x))
    attrib_per_product = attributes.groupby(
        'product_uid').agg(lambda x: x.tolist())
    attrib_per_product = attrib_per_product.reset_index()
    attrib_per_product['value'] = attrib_per_product['value'].apply(
        lambda x: ','.join(x))

    value_col = [(value,) for value in attrib_per_product['value']]
    pool = Pool()
    attrib_per_product['value'] = pool.starmap(attrib_stack_parallelized, value_col)
    pool.close()
    
    attrib_per_product.to_csv(filepath_to_save_to)
    attrib_per_product = pd.read_csv(filepath_to_save_to)
    attrib_per_product = attrib_per_product.drop('Unnamed: 0', axis=1)
    return attrib_per_product

def attrib_stack_parallelized(value_col):
    value_col = tokenizer(value_col)
    value_col = ','.join(value_col)
    return value_col


def join_attrib(train, attrib_per_product):
    """
    Join the aggregated attributes to the train dataframe
    """
    train = train.set_index('product_uid').join(
        attrib_per_product.set_index('product_uid'))
    train = train.reset_index()
    attrib_per_product = attrib_per_product.reset_index()
    return train, attrib_per_product


def search_term_in_attrib(train):
    """
    Convert the search term (stemmed) and attributes description to a set of words
    and find the number of common terms between both in the column search_term_in_attrib.
    """
    train['value'].fillna('', inplace=True)
    train['value'] = train['value'].apply(lambda x: set(x.split(',')))
    train['search_term_split'] = train['search_term'].apply(
        lambda x: set(tokenizer(x)))
    search_term_in_attrib = []
    for i in range(len(train)):
        p = len(train['search_term_split'][i].intersection(train['value'][i]))
        search_term_in_attrib.append(p)
    train['search_term_in_attrib'] = search_term_in_attrib
    return train


def color_df(attributes, train):
    """
    Find the attributes for color per product, join it with train data and 
    check for match in the search term
    """
    attrib_col = attributes[attributes['name'].apply(
        lambda x: 'color' in str(x).lower())]
    attrib_col = attrib_col.groupby('product_uid').agg(lambda x: x.tolist())
    attrib_col = attrib_col.drop('name', axis=1)
    attrib_col = attrib_col.reset_index()
    attrib_col = attrib_col.rename(columns={'value': 'color'})

    attrib_col['color'] = attrib_col['color'].apply(lambda x: ','.join(x))
    attrib_col['color'] = attrib_col['color'].apply(
        lambda x: ','.join(x.replace('/', '').replace(' ', ',').split(',')).replace(',,', ','))

    train = train.set_index('product_uid').join(
        attrib_col.set_index('product_uid'))
    train = train.reset_index()
    attrib_col = attrib_col.reset_index()
    train['color'].fillna('', inplace=True)
    train['search_term'].fillna('', inplace=True)
    train['color'] = train['color'].apply(lambda x: set(x.split(',')))
        
    pool = Pool()
    color_in_search_term = pool.starmap(find_color_in_search_term_parallelized, zip(list(train['color']),
                                                                                    list(train['search_term_split'])))
    pool.close()
    train['color_in_search_term'] = color_in_search_term

    return train

def find_color_in_search_term_parallelized(color_column, search_term_split_col):
    return len(color_column.intersection(search_term_split_col))

def search_title_lev_dist(train, filepath_to_save_to):
    """
    Calculate Levenshtein distance between search term and the product title
    """
    train.to_csv(filepath_to_save_to)
    train = pd.read_csv(filepath_to_save_to)
    train = train.drop(['Unnamed: 0'], axis=1)
    train['product_title_clean'] = train['product_title'].apply(
        lambda x: list(set(tokenize(x))))
    train['search_term'].fillna('', inplace=True)
    train['search_term_split'] = train['search_term'].apply(
        lambda x: x.split(' '))

    p = []
    for i in range(len(train)):
        q = []
        if len(train['search_term_split'][i][0]) > 0:
            for j in range(len(train['search_term_split'][i])):
                for k in range(len(train['product_title_clean'][i])):
                    if train['search_term_split'][i][j] in train['product_title_clean'][i][k]:
                        q.append((train['product_title_clean'][i]
                                  [k], train['product_title_clean'][i][k]))
                        continue
                    elif train['search_term_split'][i][j][0] == train['product_title_clean'][i][k][0]:
                        q.append((train['search_term_split'][i][j],
                                  train['product_title_clean'][i][k]))
        p.append(q)

    l = []
    for i in range(len(p)):
        q = []
        for j in range(len(p[i])):
            q.append(distance(p[i][j][0], p[i][j][1]))
        l.append(q)

    m = []
    for q in l:
        if q == []:
            m.append(1000)
        else:
            m.append(min(q))

    train['min_levenstein_dist_title'] = m

    return train


def search_brand_lev_dist(train, attributes):
    """
    Filter out the brand from attributes, join it with train data.
    Calculate Levenshtein distance between search term and the brand
    """
    attr_brand = attributes[(attributes['name'].str.lower().str.contains(
        'brand') == True) & attributes['value'].notnull()]
    attr_brand = attr_brand.drop('name', axis=1)
    attr_brand = attr_brand.rename(columns={'value': 'brand'})
    attr_brand['product_uid'] = attr_brand['product_uid'].apply(
        lambda x: int(x))

    d = defaultdict(list)
    p = list(attr_brand['product_uid'])
    b = list(attr_brand['brand'])
    for i in range(len(p)):
        if p[i] not in d:
            d[p[i]] = tokenize(b[i])
        else:
            continue
    train['brand'] = train['product_uid'].apply(lambda x: d[x])
    train['brand'].fillna('', inplace=True)
    train['search_term'].fillna('', inplace=True)
    train['search_term_split'] = train['search_term'].apply(
        lambda x: x.split(' '))

    p = []
    for i in range(len(train)):
        q = []
        if len(train['search_term_split'][i][0]) > 0:
            for j in range(len(train['search_term_split'][i])):
                for k in range(len(train['brand'][i])):
                    if train['search_term_split'][i][j] in train['brand'][i][k]:
                        q.append((train['brand'][i][k], train['brand'][i][k]))
                        continue
                    elif train['search_term_split'][i][j][0] == train['brand'][i][k][0]:
                        q.append((train['search_term_split']
                                  [i][j], train['brand'][i][k]))
        p.append(q)

    l = []
    for i in range(len(p)):
        q = []
        for j in range(len(p[i])):
            q.append(distance(p[i][j][0], p[i][j][1]))
        l.append(q)

    m = []
    for q in l:
        if q == []:
            m.append(1000)
        else:
            m.append(min(q))

    train['min_levenstein_dist_brand'] = m

    return train

def letter_prob(phrases):
    """
    :param phrases:         a list of strings of text
    :returns:               a list of dictionaries of probabilities for characters in the text 
    """
    letter_counters = []
    for phrase in phrases:
        letter_count = defaultdict(lambda: 0)
        for char in phrase:
            if char.isalpha():
                if char in letter_count:
                    letter_count[char] += 1
                else:
                    letter_count[char] = 1
        letter_counters.append(letter_count)

        total_count = float(sum(list(letter_count.values())))

        for key in letter_count.keys():
            letter_count[key] = letter_count[key] / total_count

    return letter_counters


def calculate_entropy(probs_list):
    """
    :param probs_list:      a list of dictionaries in which the values are probabilities
    :returns:               a list of entropies calculated for the given probs_list
    """
    entropies = []
    for distribution in probs_list:
        entropy = 0
        for key in distribution.keys():
            entropy += distribution[key] * math.log2(distribution[key])
        entropy *= -1
        entropies.append(entropy)
    return entropies


def longest_common_subsequence(X, Y):
    """
    :param X:               a list of strings of text
    :param Y:               a list of strings of text
    :returns:               a list of the integer length of the longest common subsequence 
                            between the strings
    """
    lcs = []

    for idx, x in enumerate(X):
        m = len(x)
        n = len(Y[idx])

        L = [[None]*(n+1) for i in range(m+1)]

        for i in range(m+1):
            for j in range(n+1):
                if i == 0 or j == 0:
                    L[i][j] = 0
                elif x[i-1] == Y[idx][j-1]:
                    L[i][j] = L[i-1][j-1]+1
                else:
                    L[i][j] = max(L[i-1][j], L[i][j-1])
        lcs.append(L[m][n])

    return lcs

def longest_common_subsequence_parallelized(X, Y):
    m = len(X)
    n = len(Y)

    L = [[None]*(n+1) for i in range(m+1)]

    for i in range(m+1):
        for j in range(n+1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i-1] == Y[j-1]:
                L[i][j] = L[i-1][j-1]+1
            else:
                L[i][j] = max(L[i-1][j], L[i][j-1])
    return L[m][n]
    
def calculate_jaccard_index(text_1, text_2):
    """
    :param text_1:         a list of strings of text
    :param text_2:         a second list of strings of text
    :returns:              a list of jaccard indices (intersection of words / union of words)
                           between the strings of text provided
    """
    jaccard_indices = []
    text_1 = list(text_1)
    text_2 = list(text_2)
    for text in zip(text_1, text_2):
        tokens_1 = set(tokenize(text[0]))
        tokens_2 = set(tokenize(text[1]))
        intersection_ = tokens_1.intersection(tokens_2)
        union_ = tokens_1.union(tokens_2)
        jaccard_indices.append(
            len(list(intersection_)) / float(len(list(union_))))
    return pd.DataFrame(jaccard_indices)

def jaro(s, t):
    """
    Jaro distance is to describe the editing distance from one string to another. But all Jaro distance is bigger than 0.
    So we need some threshold to select the closest in meanning. 0.8 as a threshold should be sufficient after testing.

    reference: https://rosettacode.org/wiki/Jaro_distance#Python
    param: s,t are two strings to compute the Jaro distiance. For instance, s='soup', t='sour'.
    return: The Jaro distance.
    """
    s_len = len(s)
    t_len = len(t)

    if s_len == 0 and t_len == 0:
        return 1

    match_distance = (max(s_len, t_len) // 2) - 1

    s_matches = [False] * s_len
    t_matches = [False] * t_len

    matches = 0
    transpositions = 0

    for i in range(s_len):
        start = max(0, i-match_distance)
        end = min(i+match_distance+1, t_len)

        for j in range(start, end):
            if t_matches[j]:
                continue
            if s[i] != t[j]:
                continue
            s_matches[i] = True
            t_matches[j] = True
            matches += 1
            break
    if matches == 0:
        return 0

    k = 0
    for i in range(s_len):
        if not s_matches[i]:
            continue
        while not t_matches[k]:
            k += 1
        if s[i] != t[k]:
            transpositions += 1
        k += 1

    return ((matches / s_len) +
            (matches / t_len) +
            ((matches - transpositions/2) / matches)) / 3


def getJaroScoreOnDocs(query, long_text):
    """
    Use pairwise between two list of words to compute the Jaro distance and filter on only sufficiently large
    distance. A large Jaro distance ie. >0.9 indicate the two words have same semantics. The sum of all jaro score
    indicates the number of overlapping terms.

    param: query, the search term passed in to compute Jaro.
    param: long_text is the description or title to compute Jaro.
    return: A sum of Jaro score.
    """
    # transform query and long_text to list of words.
    query_ls = query.split()
    long_text_ls = long_text.split()

    total_J_score = 0
    for i in query_ls:
        j_score_in_i = sum([jaro(i, j)
                            for j in long_text_ls if jaro(i, j) > 0.83])
        total_J_score += j_score_in_i

    return total_J_score


def createJaroCol(df, query_col_name, text_col_name, new_col_name):
    """
    This function builds on getJaroScoreOnDocs, which means to compute the whole two columns' jaro distance.
    Could combine title and description as a unit to compute Jaro score. It will be higher but as one score, easy to compute.

    param: df is the data we want to append. Given query column name and long text col name we can get the data.
    return: Append the jaro score as a column at the end of data frame.
    """
    # Could combine title and description as a unit to compute Jaro score.
    # It will be higher but as one score, easy to compute.
    # compute all jscore in a list
    j_score_ls = []
    for i in range(len(df)):
        query = df[query_col_name].iloc[i]
        long_text = df[text_col_name].iloc[i]
        j_score = getJaroScoreOnDocs(query, long_text)
        j_score_ls.append(j_score)
    df[new_col_name] = j_score_ls

    return None


def smith_waterman(a: str, b: str, alignment_score: float = 1, gap_cost: float = 1) -> float:
    """
    Compute Smith_Waterman score on two strings.\n",
    The SW algorithm will yield a matrix represents all possible alignment's score. \n",
    We can use two string to get the optimal alignment sequence and compute the alignment score.\n",
    """
    # H holds the alignment score at each point, computed incrementally
    H = np.zeros((len(a) + 1, len(b) + 1))
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            # The score for substituting the letter a[i-1] for b[j-1]. Generally low
            # for mismatch, high for match.
            match = H[i-1, j-1] + (alignment_score if a[i-1] == b[j-1] else 0)

            # The scores for for introducing extra letters in one of the strings (or
            # by symmetry, deleting them from the other).
            delete = H[1:i, j].max() - gap_cost if i > 1 else 0
            insert = H[i, 1:j].max() - gap_cost if j > 1 else 0
            H[i, j] = max(match, delete, insert, 0)
    # The highest score is the best local alignment.
    # For our purposes, we don't actually care _what_ the alignment was, just how
    # aligned the two strings were.
    return H.max()


def getSWscore(query, long_text):
    """
    param: query is the search query as a string.
    param: text is the long text to compute the similarity.
    return the number of significant alignment strings in both text. ie. the number of similar terms in query and long_text.
    """
    query_ls = query.split()
    long_text_ls = long_text.split()

    sw_score = []
    for i in query_ls:
        score = sum([smith_waterman(i, j)
                     for j in long_text_ls if smith_waterman(i, j) >= 4.0])
        sw_score.append(score)
    return round(sum(sw_score)/5)


def createSWscoreCol(df, query_col_name, long_text_col_name, new_col_name):
    """
    Just to compute pairwise SW score on two columns.
    param: df is the data we want to append. Given query column name and long text col name we can get the data.
    return: Append the SW score as a column at the end of data frame.
    """
    first_col = df[query_col_name]
    second_col = df[long_text_col_name]

    score_ls = []
    for i in range(len(first_col)):
        score_ls.append(getSWscore(first_col.iloc[i], second_col.iloc[i]))
    df[new_col_name] = score_ls
    return df

def create_SW_score_col_parallelized(query_col_name, long_text_col_name):
    return getSWscore(query_col_name, long_text_col_name)
    

def computeNCD(string1, string2):
    """
    params: string1 is the query term
    params: string2 is the word in long-text, like title, description.
    """
    # Get concated strings and transform to bytes-like object for lzma.compress.
    concat_str = string1+string2
    string1 = bytes(string1, 'utf-8')
    string2 = bytes(string2, 'utf-8')
    concat_str = bytes(concat_str, 'utf-8')

    # Get the compressed file for each string.
    str1_comp = lzma.compress(string1)  # compress file 1
    str2_comp = lzma.compress(string2)  # compress file 2
    concat_str_comp = lzma.compress(concat_str)  # compress file concatenated

    # magic happens here
    ncd = (len(concat_str_comp) - min(len(str1_comp), len(str2_comp))) / \
        max(len(str1_comp), len(str2_comp))

    return ncd


def createNCDCol(df, search_name, long_text_name, new_col_name):
    """
    Just to compute pairwise NCD similarity score on two columns.
    param: df is the data we want to append. Given query column name and long text col name we can get the data.
    return: Append the NCD score as a column at the end of data frame.
    """
    NCD_score_ls = []
    for i in range(len(df)):
        str1 = df[search_name].iloc[i]
        str2 = df[long_text_name].iloc[i]
        NCD_score_ls.append(np.mean([computeNCD(a, b)
                                     for a in str1.split() for b in str2.split()]))
    df[new_col_name] = NCD_score_ls
    return df

def compute_ncd_parallelized(str1, str2):
    return np.mean([computeNCD(a, b)
                    for a in str1.split() for b in str2.split()])

def gettext(xmltext):
    """
    Parse xmltext and return the text from <title> and <text> tags
    """

    # ensure there are no weird char
    xmltext = xmltext.encode('ascii', 'ignore')
    root = ET.fromstring(xmltext)
    text = []
    for elem in root.iterfind('title'):
        text.append(elem.text)
    for elem in root.iterfind('.//text/*'):
        text.append(elem.text)
    text = ' '.join(text)

    return text


def compute_tfidf(corpus):
    """
    Create and return a TfidfVectorizer object after training it on
    the list of articles pulled from the corpus dictionary. The
    corpus argument is a dictionary mapping file name to xml text.
    """
    tfidf = TfidfVectorizer(input='content',
                            analyzer='word',
                            preprocessor=gettext,
                            tokenizer=tokenizer,
                            stop_words='english',
                            decode_error='ignore')
    tfidf.fit(list(corpus.values()))

    return tfidf


def add_prod_description_column(train):
    """
    Add the product description from product df to train df.
    Concatenate Title and description to form total_description column.
    """
    train['total_description'] = train['product_title'] + \
        train['product_description']
    return train

def get_words(x):
    """
    Remove the tfidf scores and return only the top tfidf words
    """
    q = []
    for i in range(len(x)):
        if x[i][0] != []:
            q.append(x[i][0])
    return q


def add_tfidf_col(train):
    train['tfidf'] = train['tfidf'].apply(lambda x: get_words(x))
    return train


def num_stop_words(x):
    stops = list(stop_words.ENGLISH_STOP_WORDS)
    return len([w for w in x if w in stops])


def find_tfidf_words_in_search(train):
    train['search_term_split'] = train['search_term'].apply(
        lambda x: tokenizer(x))
    p = train['search_term_split']
    q = train['tfidf']
    l = []
    for i in range(len(p)):
        l.append(len(set(p[i]).intersection(set(q[i]))))
    train['tfidf_search_common'] = l

    return train


def num_attrib_per_product(attributes):
    """
    Find the number of attributes per product
    """
    attributes['value'] = attributes['value'].apply(
        lambda x: tokenizer(str(x)))
    attributes['value'] = attributes['value'].apply(lambda x: ','.join(x))
    attrib_per_product = attributes.groupby(
        'product_uid').agg(lambda x: x.tolist())
    attrib_per_product = attrib_per_product.reset_index()
    attrib_per_product['value'] = attrib_per_product['value'].apply(
        lambda x: ','.join(x).replace(',', ' '))
    attrib_per_product['num_attrib'] = attrib_per_product['name'].apply(
        lambda x: len(x))
    attrib_per_product['value'].fillna('', inplace=True)
    attrib_per_product.rename(columns={'value': 'attribs'})
    attrib_per_product['product_uid'] = attrib_per_product['product_uid'].apply(
        lambda x: int(x))

    return attrib_per_product

def find_n_tfidf_highest_scores(train_set, n):
    tfidf = TfidfVectorizer(input='content',
                            analyzer='word',
                            tokenizer=tokenizer,
                            stop_words='english',
                            decode_error='ignore')
    tfidf.fit(train_set['total_description'])

    p = []
    total_description = list(train_set['total_description'])
    
    pool = Pool()
    p = pool.starmap(find_n_tfidf_highest_scores_parallelized, zip(total_description, repeat(tfidf), repeat(5)))
    pool.close()

    train_set['tfidf'] = p
    return train_set

def find_n_tfidf_highest_scores_parallelized(total_description, tfidf, n):
    response = tfidf.transform([total_description])
    feature_names = tfidf.get_feature_names()
    col = response.nonzero()[1]
    t = [(feature_names[col], response[0, col])
         for col in response.nonzero()[1] if response[0, col] >= 0.09]
    t.sort(key=lambda x: x[1], reverse=True)
    return t[0:n]

def add_word_count_features(train_df):
    train_df['num_words_in_description'] = train_df['total_description'].apply(
        lambda x: len(tokenize(x)))
    train_df['num_stop_words'] = train_df['search_term'].apply(
        lambda x: num_stop_words(x.split(' ')))
    train_df['num_search_words'] = train_df['search_term'].apply(
        lambda x: len(x.split(' ')))
    return train_df

def add_num_attrib_per_prod_column(train_df, attributes_df):
    attrib_per_product = num_attrib_per_product(attributes_df)
    train_df = train_df.set_index('product_uid').join(
        attrib_per_product.set_index('product_uid'),
        lsuffix='', rsuffix='_r')
    train_df = train_df.reset_index()
    attrib_per_product = attrib_per_product.reset_index()

    train_df = train_df.drop('name_r', 1)
    train_df = train_df.drop('value_r', 1)
    train_df['num_attrib'] = train_df['num_attrib'].fillna(0)

    return train_df

def get_all_numerical_cols(all_features):
    """
    param: all_features is a data frame containning all features.
    output: column names of all numerical features.
    """
    col_names = all_features.columns.tolist()
    all_num_ind = [10, 12]+list(range(22, len(col_names)))
    all_num_col = [col_names[i] for i in all_num_ind]

    return all_num_col


def get_similarity_cols(all_num_features):
    """
    param: all_features is a data frame containning all numerical features.
    output: column names of all similarity features.
    """
    all_similarity_features = [all_num_features.columns.tolist(
    )[i] for i in [0, 1, 11, 12, 15, 16, 17, 18, 19, 20, 21, 22, 23]]
    return all_similarity_features


def get_count_and_other_cols(all_similarity_features, all_num_features):
    """
    return the column names of all count features and len_Entropy columns.
    """
    all_other_num_cols = set(all_num_features.columns.tolist()).difference(
        set(all_similarity_features.columns.tolist()))
    col_has_in = [i for i in all_other_num_cols if "in" in i]
    len_H_features = list(set(all_other_num_cols).difference(set(col_has_in)))

    return col_has_in, len_H_features

def create_cleaned_terms_col(train_df):
    search_terms = train_df['search_term']
    cleaned_terms = [' '.join(tokenize(search_term))
                     for search_term in search_terms]
    train_df['cleaned_terms'] = cleaned_terms
    return train_df

def create_stemmed_cols(train_df):
    # stem the search terms, title, and descriptions
    search_terms = train_df['search_term']
    stemmed_terms = [' '.join(stemmed(tokenize(search_term)))
                     for search_term in search_terms]
    stemmed_title = [' '.join(stemmed(tokenize(t)))
                     for t in train_df['product_title']]
    stemmed_desc = [' '.join(stemmed(tokenize(d)))
                    for d in train_df['product_description']]

    train_df['stemmed_terms'] = stemmed_terms
    train_df['stemmed_title'] = stemmed_title
    train_df['stemmed_desc'] = stemmed_desc
    return train_df

def create_lemmatized_cols(train_df):
    search_terms = train_df['search_term']
    # lemmatize the search terms, title, and descriptions
    lemmatized_terms = [' '.join(lemmatized(tokenize(search_term)))
                        for search_term in search_terms]
    lemmatized_title = [' '.join(lemmatized(tokenize(t)))
                        for t in train_df['product_title']]
    lemmatized_desc = [' '.join(lemmatized(tokenize(d)))
                       for d in train_df['product_description']]

    train_df['lemmatized_terms'] = lemmatized_terms
    train_df['lemmatized_title'] = lemmatized_title
    train_df['lemmatized_desc'] = lemmatized_desc
    return train_df

def create_length_cols(train_df):
    train_df['clean_length'] = get_length(list(train_df['cleaned_terms']))
    train_df['title_length'] = get_length(list(train_df['product_title']))
    train_df['desc_length'] = get_length(list(train_df['product_description']))
    return train_df

def find_cleaned_terms_in_corpus_cols(train_df):
    train_df['clean_terms_in_title'] = clean_term_in_doc(list(train_df['cleaned_terms']), 
                                                         list(train_df['product_title']))
    train_df['clean_terms_in_desc'] = clean_term_in_doc(list(train_df['cleaned_terms']), 
                                                        list(train_df['product_description']))
    return train_df

def find_stemmed_terms_in_corpus_cols(train_df):
    train_df['stemmed_terms_in_title'] = clean_term_in_doc(
        list(train_df['stemmed_terms']), list(train_df['stemmed_title']))
    train_df['stemmed_terms_in_desc'] = clean_term_in_doc(
        list(train_df['stemmed_terms']), list(train_df['stemmed_desc']))
    return train_df

def find_lemmatized_terms_in_corpus_cols(train_df):
    train_df['lemmatized_terms_in_title'] = clean_term_in_doc(
        list(train_df['lemmatized_terms']), list(train_df['lemmatized_title']))
    train_df['lemmatized_terms_in_desc'] = clean_term_in_doc(
        list(train_df['lemmatized_terms']), list(train_df['lemmatized_desc']))
    return train_df

def create_jaccard_index_cols(train_df):
    train_df['jaccard_index_title'] = calculate_jaccard_index(list(train_df['product_title']), 
                                                              list(train_df['cleaned_terms']))
    train_df['jaccard_index_desc'] = calculate_jaccard_index(list(train_df['product_description']), 
                                                             list(train_df['cleaned_terms']))
    return train_df

def create_entropy_cols(train_df):
    train_df['search_terms_entropy'] = calculate_entropy(letter_prob(list(train_df['cleaned_terms'])))
    train_df['title_entropy'] = calculate_entropy(letter_prob(list(train_df['product_title'])))
    return train_df

def create_LCS_cols(train_df):
    train_df['lcs_title'] = longest_common_subsequence(list(train_df['cleaned_terms']), 
                                                       list(train_df['product_title']))
    train_df['lcs_desc'] = longest_common_subsequence(list(train_df['cleaned_terms']), 
                                                      list(train_df['product_description']))
    return train_df

def find_neighbors_in_corpus(train_df, dictionary, glove_file):
    wordlist, matrix = split_dictionary(glove_file)
    cleaned_set = unique_words(train_df)
    find_nearest_neighbors('Data/glove_neighbour_no_w.txt',
                           cleaned_set, matrix, wordlist, dictionary)
    k_dict = build_dictionary('Data/glove_neighbour_no_w.txt')
    terms_neighbour = get_all_terms_neighbors(k_dict, list(train_df['cleaned_terms']))
    train_df['terms_neighbour'] = terms_neighbour
    train_df['neighbours_in_title'] = clean_term_in_doc(terms_neighbour, list(train_df['product_title']))
    train_df['neighbours_in_desc'] = clean_term_in_doc(terms_neighbour, list(train_df['product_description']))
    return train_df 

def grid_search_models_rmse(models, model_params, train_data, train_target):
    """
    :param models:          a list of pipelines of models
    :param model_params:    a list of dictionaries with the model parameters
    """
    best_models = []
    for model in zip(models, model_params):
        gs = GridSearchCV(estimator=model[0],
                          param_grid=model[1],
                          scoring='neg_mean_squared_error',
                          cv=5)
        if type(train_target) != np.ndarray:
            y = train_target.values.ravel()
            train_target = np.array(y).astype(float)
        gs.fit(train_data, train_target)
        best_models.append(
            (sqrt(-1 * gs.best_score_), gs.best_params_, model[0]))
    return best_models

def fit_best_model(best_model, train_data, train_target, test_data):
    best_model = best_model[2].steps[-1][1].__class__(**best_model[1])
    best_model.fit(train_data, train_target)
    return best_model.predict(test_data)

def change_best_params_keys(best_models, old_names, new_names):
    best_model = sorted(best_models, key=lambda model: model[0])[0]
    for idx in range(len(old_names)):
        best_model[1][new_names[idx]] = best_model[1].pop(old_names[idx])
    return (best_model[0], best_model[1], best_model[2])