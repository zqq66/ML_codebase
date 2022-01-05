import sys
import json
import os
from nltk.tokenize import word_tokenize
from collections import defaultdict
import string
import time
import nltk
import csv
import math
import heapq
from knn.knn_inference import evaluation
nltk.download('punkt')


'''
class for calculating mutual information between terms and classes in the training data
and write a new training file in which the features are only those in the top-k for each class
'''
class MI:
    def __init__(self, train_json_path):
        f1 = open(train_json_path)
        self.corpus = json.load(f1)
        # used to store words eliminated for each class
        self.eliminateword = defaultdict(list)
        # used to store class for each documents in the same order of corpus
        self.label_list = []
        # used to store text for each documents in the same order of corpus
        self.unique_text_list = []
        self.text_list = []
        # used to store all labels appeared in corpus
        self.all_label = []
        # number of documents
        self.N = 0
        self.token2id = {}
        self.id2token = {}
        # number of unique token
        self.numtoken = 0
        # number of class
        self.numlabel = 0

    '''
    convert all tokens into integer 
    use self.corpus to update 
    self.all_labels with names of all classes appeared in the training set
    self.label_list with names of classes in the ordered they appeared in the training set
    self.text_list with documents in the ordered they appeared in the training set
    self.numtoken with number of unique tokens in training set
    self.numlabel with number of unique classes in training set
    self.token2id and self.id2token for storing the vocabulary of the training set
    '''
    def build_vocab(self):
        # corpus: a list of {"category": astring, "text": astring}
        _id_count = 0
        self.N = len(self.corpus)
        for each_doc in self.corpus:
            text = each_doc["text"]
            label = each_doc["category"]
            self.label_list.append(label)
            if label not in self.all_label:
                self.all_label.append(label)
            temp = []
            unique_temp = []
            for token in preprocess(text):
                if token not in self.token2id:
                    self.token2id[token] = _id_count
                    _id_count += 1
                temp.append(self.token2id[token])
                if self.token2id[token] not in unique_temp:
                    unique_temp.append(self.token2id[token])
            self.text_list.append(temp)
            self.unique_text_list.append(unique_temp)

        self.numtoken = _id_count
        self.numlabel = len(self.all_label)
        self.id2token = {v: k for k, v in self.token2id.items()}

    '''
    update the self.notation_contain and self.notation_not_contain
    where both of them have shape of [self.numlabel, self.numtoken]
    self.notation_contain is used to record the number of documents that one class contain one word
    self.notation_not_contain is used to record the number of documents that one class not contain one word
    '''
    def build_matrix(self):
        # shape: self.numlabel, self.numtoken
        self.notation_contain = [[0 for i in range(self.numtoken)] for i in range(self.numlabel)]
        # np.zeros((self.numlabel, self.numtoken))
        self.notation_not_contain = [[0 for i in range(self.numtoken)] for i in range(self.numlabel)]
        # np.zeros((self.numlabel, self.numtoken))
        all_words = set(self.id2token.keys())
        for i in range(self.N):
            label = self.label_list[i]
            class_idx = self.all_label.index(label)
            word_contain = set()
            text = self.unique_text_list[i]
            # accumulate one to self.notation_contain[class_idx][word_idx]
            # for those word_idx appeared in current document
            for word_idx in text:
                word_contain.add(word_idx)
                self.notation_contain[class_idx][word_idx] += 1
            # accumulate one to self.notation_not_contain[class_idx][word_idx]
            # for those word_idx not appeared in current document
            not_contain = all_words - word_contain
            for word in not_contain:
                self.notation_not_contain[class_idx][word] += 1

    '''
    use the self.notation_contain and self.notation_not_contain to find 
    N11: the document belongs to the class and contain the specific token
    N10: the document not belongs to the class but contain the specific token
    N01: the document belongs to the class but does not contain the specific token
    N00: the document not belongs to the class and does not contain the specific token
    '''
    def mi(self, k):
        contain_sum = [sum(x) for x in zip(*self.notation_contain)]
        not_contain_sum = [sum(x) for x in zip(*self.notation_not_contain)]
        # print(self.notation_contain, contain_sum)
        for label_id in range(self.numlabel):
            # initialize a heapq to store the top-k MI of tokens and current class
            maxQ = []
            for word in self.id2token.keys():
                N11 = self.notation_contain[label_id][word]
                # document not in class, but contain word
                N10 = contain_sum[word] - N11
                # document in the class, but word not found
                N01 = self.notation_not_contain[label_id][word]
                N00 = not_contain_sum[word] - N01
                # print(self.id2token[word], N11, N10, N01, N00)
                I = (N11/self.N)*math.log2((self.N * N11)/((N11+N10)*(N11+N01))) if N11 != 0 else 0
                I += (N10/self.N)*math.log2((self.N * N10)/((N11+N10)*(N00+N10))) if N10 != 0 else 0
                I += (N01/self.N)*math.log2((self.N * N01)/((N00+N01)*(N11+N01))) if N01 != 0 else 0
                I += (N00/self.N)*math.log2((self.N * N00)/((N00+N01)*(N00+N10))) if N00 != 0 else 0
                # print(self.id2token[word], I)
                # when the length of heapq is smaller than k, push the new value to the minheap
                if len(maxQ) < k:
                    heapq.heappush(maxQ, (I, word))
                # otherwise, compare the value of current MI and the smallest value of the minheap
                # if current MI has a greater value, push current MI to the minheap
                # otherwise push the smallest value to the minheap
                # in this way, minheap will always obtain the greatest k MI in the dataset.
                else:
                    a, word2 = heapq.heappop(maxQ)
                    if a >= I:
                        heapq.heappush(maxQ, (a, word2))
                    else:
                        heapq.heappush(maxQ, (I, word))

            self.eliminateword[self.all_label[label_id]] = [self.id2token[word] for p, word in maxQ]

    def gather_data(self):
        data = []
        for idx, text in enumerate(self.text_list):
            label = self.label_list[idx]
            cur_eliminate = self.eliminateword[label]
            # only keep those appear in the heap
            eliminated = [self.id2token[ele] for ele in text if self.id2token[ele] in cur_eliminate]
            data.append({"category": label,
                         "text": " ".join(eliminated)})
        return data

'''
class for implementing Naive Bayes Classifier 
parameter:
---------
LABELS_DIC: all classes appeared in the training set
'''
class MultinomialNaiveBayes:

    def __init__(self, LABELS_DIC):
        # count is a dictionary which stores several dictionaries corresponding to each news category
        # each value in the subdictionary represents the freq of the key corresponding to that news category
        self.count = {}
        # classes represents the different news categories
        self.classes = LABELS_DIC
        self.prior = defaultdict(float)
        self.likelihood = defaultdict(dict)

    '''
    count the frequency of each word of each class
    compute parameter based on the frequency
    write probabilities of Naive Bayes model to tsv file 
    parameter
    ---------
    X_train: documents encoded by integer based on token2id
    Y_train: True corresponding class of the documents
    path2: path of output tsv file
    id2token: a dictionary have tokens as key and its corresponding integer as value
    token2id: a dictionary have integer as key and its corresponding token as value
    '''
    def fit(self, X_train, Y_train, token2id=None):
        for class_ in self.classes:
            self.count[class_] = {}
            for i in range(len(X_train[0])):
                self.count[class_][i] = 0
            self.count[class_]['total'] = 0
            self.count[class_]['total_points'] = 0
        self.count['total_points'] = len(X_train)

        for i in range(len(X_train)):
            for j in range(len(X_train[0])):
                # count the frequency of word j in ith document
                self.count[Y_train[i]][j] += X_train[i][j]
                # count the total number of word appeared in ith document
                self.count[Y_train[i]]['total'] += X_train[i][j]
            # count the total number of documents of the class of ith document
            self.count[Y_train[i]]['total_points'] += 1
        N = len(X_train)
        for class_ in self.classes:
            NC = self.count[class_]['total_points']
            prior = NC / N
            self.prior[class_] = float(prior)
        # find likelihood p(document|class)
        for class_ in self.classes:
            for token in token2id.keys():
                # add-one smoothing
                # numerator + 1
                # denominator + size of the vocabulary
                T = self.count[class_][token2id[token]] + 1
                num = self.count[class_]['total'] + len(token2id.keys())
                # print(T, num, token, class_)
                prob = T / num
                self.likelihood[class_][token] = prob
        return self.prior, self.likelihood

    '''
        compute the log probability for specific class and documents
        parameter:
        ----------
        test_doc: document 
        class_: class
        returns:
        --------
        log probability of input class given input document
        '''

    def __probability(self, test_doc, class_):

        log_prob = math.log2(self.prior[class_])
        tokens = preprocess(test_doc)
        # traverse all tokens to compute the log-prob of the documents
        for token in tokens:
            if token in self.likelihood[class_].keys():
                log_prob += math.log2(float(self.likelihood[class_][token]))

        return log_prob

    '''
    predict the topic(class) of the given document
    parameter:
    ----------
    test_doc: document

    returns:
    --------
    class with highest log probability of the input document
    '''

    def __predictSinglePoint(self, test_doc):
        best_class = None
        best_prob = None
        first_run = True
        # traverse all possible classes to find the one with highest log probability
        for class_ in self.classes:
            log_probability_current_class = self.__probability(test_doc, class_)
            if (first_run) or (log_probability_current_class > best_prob):
                best_class = class_
                best_prob = log_probability_current_class
                first_run = False

        return best_class

    '''
    predict the topic(class) of all documents in the given dataset
    parameter:
    ----------
    test_doc: dataset with different documents

    returns:
    --------
    the most fit class of all documents in the input dataset
    '''

    def predict(self, X_test):
        # This can take some time to complete

        Y_pred = []
        for i in range(len(X_test)):
            # print(i) # Uncomment to see progress
            Y_pred.append(self.__predictSinglePoint(X_test[i]))

        return Y_pred

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
    return handel_digit


'''
    convert all tokens into integer 
    load the input json file which follows the format a list of {"category": astring, "text": astring}
    token2id: a dictionary have tokens as key and its corresponding integer as value
    and id2token: a dictionary have integer as key and its corresponding token as value
'''
def build_vocab(corpus):
    # corpus: a list of {"category": astring, "text": astring}
    _token2id = {}
    _id_count = 0
    for each_doc in corpus:
        text = each_doc["text"]
        for token in preprocess(text):#text.split():
            if token not in _token2id:
                _token2id[token] = _id_count
                _id_count += 1

    _id2token = {v: k for k, v in _token2id.items()}
    return _token2id, _id2token

'''
process the corpus and encode token in documents to its corresponding integer based on token2id
call the fit function in MultinomialNaiveBayes to complete the classification process
'''
def build_training_set(train_path, val_path, test_path):
    HYPER_PARA = [10, 20, 40, 80, 100, 160, 240]
    best_F1 = float('-inf')
    best_k = None
    test = open(test_path)
    test_corpus = json.load(test)
    best_clf = None
    for k in HYPER_PARA:
        data = MI(train_path)
        data.build_vocab()
        data.build_matrix()
        data.mi(k)
        filtered_json = data.gather_data()
        val = open(val_path)
        LABELS_DIC = []
        # corpus: a list of {"category": astring, "text": astring}
        val_corpus = json.load(val)
        training_data = []
        training_label = []
        token2id, id2token = build_vocab(filtered_json)
        # X_train_dataset = np.zeros((len(train_corpus),len(token2id.keys())))
        X_train_dataset = [[0 for _ in range(len(token2id.keys()))] for _ in range(len(filtered_json))]
        for idx, doc in enumerate(filtered_json):
            label = doc["category"]
            long_string = doc["text"]
            training_data.append(long_string)
            training_label.append(label)
            if label not in LABELS_DIC:
                LABELS_DIC.append(label)
            for token in preprocess(long_string):# long_string.split():
                i = token2id[token]
                X_train_dataset[idx][i]+=1

        clf = MultinomialNaiveBayes(LABELS_DIC)
        clf.fit(X_train_dataset, training_label, token2id)
        val_data = []
        val_label = []
        for idx, doc in enumerate(val_corpus):
            label = doc["category"]
            long_string = doc["text"]
            val_data.append(long_string)
            val_label.append(label)
        y_val_pred = clf.predict(val_data)
        print('-----------------Validation Set with respect to k = %d-----------------' % k)
        F1_val = evaluation(y_val_pred, val_label)
        if best_F1 < F1_val:
            best_F1 = F1_val
            best_k = k
            best_clf = clf
    print('Best Macro_F1 found on Validation Set is %f, the corresponding k is %d ' % (best_F1, best_k))
    test_data = []
    test_label = []
    for idx, doc in enumerate(test_corpus):
        label = doc["category"]
        long_string = doc["text"]
        test_data.append(long_string)
        test_label.append(label)
    y_test_pred = best_clf.predict(test_data)
    print('-----------------Test Set with respect to k = %d-----------------' % best_k)
    evaluation(y_test_pred, test_label)


if __name__ == "__main__":
    if len(sys.argv) == 4:
        train_path = sys.argv[1]
        val_path = sys.argv[2]
        test_path = sys.argv[3]
        if not os.path.exists(train_path):
            sys.exit("No Training set\n")
        if not os.path.exists(val_path):
            sys.exit("No Validation set\n")
        if not os.path.exists(test_path):
            sys.exit("No Testing set\n")

        build_training_set(train_path, val_path, test_path)

    else:
        sys.exit("incorrect number of arguments:"
                 " python3 nbc_train.py ./data/train_data.json ./data/val_data.json ./data/test_data.json\n")
