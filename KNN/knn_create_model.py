import sys
import json
import os
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
import unicodedata
import string
import math
import time
import csv
import collections
from collections import defaultdict
nltk.download('punkt')
LABELS_DIC = ["business" ,"entertainment" ,"politics" ,"sport" ,"tech"]

'''
remove punctuaionos for a string
tokenization and normalization for a string
keep apart of punctuations with in string
return a string of all text in the document
'''
def preprocess(text):
    # remove punctuations
    text = word_tokenize(text.lower())
    handel_digit = []
    for atoken in text:
        for char in atoken:
            if char.isalpha() or char.isdigit():
                handel_digit.append(atoken.strip(string.punctuation))
                break
    # stemmed = [nltk.PorterStemmer().stem(token) for token in handel_digit]
    return handel_digit # stemmed


'''
    convert all tokens into integer 
    load the input json file which follows the format a list of {"category": astring, "text": astring}
    token2id: a dictionary have tokens as key and its corresponding integer as value
    and id2token: a dictionary have integer as key and its corresponding token as value
'''
def build_vocab(path1):
    # corpus: a list of {"category": astring, "text": astring}
    f1 = open(path1)
    corpus = json.load(f1)
    _token2id = {}
    _id_count = 0
    for each_doc in corpus:
        text = each_doc["text"]
        for token in preprocess(text):
            if token not in _token2id:
                _token2id[token] = _id_count
                _id_count += 1

    _id2token = {v: k for k, v in _token2id.items()}
    return _token2id, _id2token
'''
represent all training documents as vector of tf*idf based on ltn.
remove 0 for each vector and write the output to a tsv file.
'''
def train(path1, token2id, id2token, path2):
    if not os.path.exists(path1):
        return 0
    f1 = open(path1)
    # corpus: a list of {"category": astring, "text": astring}
    corpus = json.load(f1)
    train_corpus = corpus
    training_data = []
    training_label = []
    N_documents = len(train_corpus)
    num_terms = len(token2id)
    # X_tf_dataset =[[0 for _ in range(num_terms)] for _ in range(N_documents)]
    X_tf_dataset = defaultdict(lambda: defaultdict(lambda: 0))

    df_dict = defaultdict(set)
    # load all training documents and save them as training_data and training_label respectively
    # compute tf for each term in each document and save tf in X_tf_dataset
    # compute df for each term and save df in  df_dict[term]
    # compute idx for each term
    idf_dic = defaultdict(float)
    for doc_idx, doc in enumerate(train_corpus):
        label = doc["category"]
        long_string = doc["text"]
        training_data.append(long_string)
        training_label.append(label)
        for token in preprocess(long_string):
            i = token2id[token]
            X_tf_dataset[doc_idx][i] += 1
            df_dict[token].add(doc_idx)

    for token in token2id.keys():
        idf_dic[token] = math.log10(N_documents / len(df_dict[token]))

    # use 1+logtf to compute tf
    # compute weights for each term in each doc by ltn
    # represent each documents by vectors of weights
    # X_train_dataset = [[0 for _ in range(num_terms)] for _ in range(N_documents)]
    X_train_dataset = defaultdict(lambda: defaultdict(lambda: 0))

    for i in range(len(train_corpus)):
        for j in X_tf_dataset[i].keys():
            X_train_dataset[i][j] = (1 + math.log10(X_tf_dataset[i][j]))*idf_dic[id2token[j]]

    # write idf of each term and document vectors in to tsv file
    with open(path2, "wt") as knn_file:
        tsv_writer = csv.writer(knn_file, delimiter="\t")
        tsv_writer.writerow(['idf/vector', "term", "idf value/vector"])
        for token in idf_dic:
            line = ["idf", token, idf_dic[token]]
            tsv_writer.writerow(line)

        for doc_id in X_train_dataset.keys():
            vector = [id2token[token_id] + "," + str(X_train_dataset[doc_id][token_id]) for token_id in X_train_dataset[doc_id].keys()]
            string_vector = " ".join(vector)
            line = ["vector", training_label[doc_id], string_vector]
            tsv_writer.writerow(line)


if __name__ == "__main__":
    if len(sys.argv) == 3:

        path1 = sys.argv[1]
        path2 = sys.argv[2]

        if not os.path.exists(path1):
            sys.exit("No such file\n")
        elif os.path.exists(path2):
            confirmation = input("Write to existed file, enter y to overwrite: \n")
            if confirmation != 'y':
                sys.exit("Unable to write to existed file")
        time1 = time.time()

        token2id, id2token = build_vocab(path1)
        time0 = time.time()
        train(path1, token2id, id2token, path2)
        time2 = time.time()
        print("build index successfully")
        print("time using: ", time2 - time1)
    else:
        sys.exit(
            "incorrect number of arguments:python3 ./knn/knn_create_model.py ./data/train.json ./bbc_model.tsv")
