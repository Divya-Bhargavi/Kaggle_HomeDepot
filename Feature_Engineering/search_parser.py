
#  ¯\_(ツ)_/¯ ¯\_(ツ)_/¯ ¯\_(ツ)_/¯ ¯\_(ツ)_/¯ ¯\_(ツ)_/¯ ¯\_(ツ)_/¯
import operator
import math
import numpy as np
from scipy import spatial

glove_file = 'glove.6B.300d.txt'
#  ¯\_(ツ)_/¯ ¯\_(ツ)_/¯ ¯\_(ツ)_/¯ ¯\_(ツ)_/¯ ¯\_(ツ)_/¯ ¯\_(ツ)_/¯


def try_num(potential_num):
    try:
        float(potential_num)
        return True
    except ValueError:
        return False


def make_dictionary(file):
    '''
    Initiate the glove model as a dictionary

    input: A String which is a file in the project directory
    returns: A dictionary with item = word : 300 d list
    '''
    vecs = dict()
    with open(file) as f:
        lines = f.readlines()
        for word_and_vec in lines:
            elems = word_and_vec.strip().split(' ')
            word = elems[0]
            vec = np.array(elems[1:], dtype=float)
            vecs[word] = vec
    return vecs

def find_count(searched_word,document):
    lines = document.split('\n')
    count = 0
    for line in lines:
        words = line.split()
        for word in words:
            if word == searched_word:
                count += 1
    return count

def find_related(current_word,dictionary,num_wanted):
    '''
    Find num_wanted words which are the minimal distance
    from current_word as defined by the cos_dict function

    :param current_word: Word which is a search term
    :param dictionary: model to search for words in
    :param num_wanted: number of nearest neighbors to search for

    :return: a list of current words nearest neighbors
    '''

    #distance used in evaluation of words
    def cos_dist(vec1,vec2):
        return spatial.distance.cosine(vec1,vec2)
    distDict = dict()
    if current_word in dictionary:
        for word in dictionary:
            distDict[word] = cos_dist(dictionary[current_word],dictionary[word])
        sorted_words = sorted(distDict.items(), key=operator.itemgetter(1))

        return sorted_words[:int(num_wanted)+1]
def decide_num_wanted(count_in, desired_length):
    # helper function to find number of words per search term
    return int(math.ceil((desired_length-count_in)/count_in))

def find_ten(search_list,dictionary):
    '''
    A method to get ten heavily related words from an unspecified number of terms

    :param search_list: The search query passed as a list of words
    :param dictionary: {word : vector}
    :return: a dictionary of word to the closest de
    '''
    DESIRED_LENGTH = 10
    related_words={}
    #decide how many words are in dictionary
    count_in = 0
    for word in search_list:
        if word in dictionary:
            count_in+=1
    '''
    Add words as equally as possible for each search term to get DESIRED_LENGTH
    number of terms.
    '''
    for word in search_list:
        if word in dictionary:
            if len(related_words)<=DESIRED_LENGTH:
                related_words[word] = find_related(word, dictionary, decide_num_wanted(count_in, DESIRED_LENGTH))
    # find 10 closest if any overflow
    return(related_words)

def multiply(wordList,numneighbors):
    wordlist = []
    matrix = []
    with open(glove_file) as f:
        lines = f.readlines()
        for word_and_vec in lines:
            wordvec = np.array([float(x) for x in word_and_vec.split()[1:]])
            matrix.append(wordvec / np.linalg.norm(wordvec))
            wordlist.append(word_and_vec.split()[0])
        matrix = np.array(matrix)
    dictionary = make_dictionary(glove_file)
    closest = {}
    for word in wordList:
        if word in dictionary:
            wordVec = dictionary[word]
            dots = matrix.dot(wordVec.T)
            knn = []
            close_index_vec = np.argsort(dots)

            for x in range(numneighbors):
                close_index = close_index_vec[-2-x]
                knn.append((wordlist[close_index],matrix[close_index].dot(wordVec)/np.linalg.norm(wordVec)))
            closest[word] = knn
    return(closest)
    
