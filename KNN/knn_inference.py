import sys
import json
import os
from nltk.tokenize import word_tokenize
import string
import heapq
import math
import time
import nltk
from collections import defaultdict

nltk.download('punkt')
LABELS_DIC = ["neg", "pos"]

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
    return handel_digit  # stemmed

'''
   read from the input tsv file and load the model into X_weights_dataset, X_labels, idf_dic
   where X_weights_dataset for weights of tokens in each document,
   X_labels for all labels of documents in order,
   idf_dic for idf of all documents
'''
def load_model(model_path):
    idf_dic = defaultdict(float)
    token2id = {}
    id = 0
    id2token = {}
    num_doc = 0
    with open(model_path, "rt") as index_file:
        for line_id, line in enumerate(index_file):
            if line_id == 0:
                continue
            line = line.split()
            if line[0] == "idf":
                term = line[1]
                idf_value = float(line[2])
                idf_dic[term] = idf_value
                token2id[term] = id
                id2token[id] = term
                id += 1

            else:
                num_doc += 1

    num_terms = len(token2id.keys())
    # X_weights_dataset = [[0 for _ in range(num_terms)] for _ in range(num_doc)]
    X_weights_dataset = defaultdict(lambda: defaultdict(lambda: 0))
    X_labels = {}
    with open(model_path, "rt") as index_file:
        for line_id, line in enumerate(index_file):
            line = line.split()
            if line[0] == "vector":
                doc_id = line_id - num_terms - 1
                class_ = line[1]
                X_labels[doc_id] = class_
                term_weights_list = line[2:]
                for each_term_weights in term_weights_list:
                    each_term_weights_list = each_term_weights.split(",")
                    term = each_term_weights_list[0]
                    weights = float(each_term_weights_list[1])
                    term_id = token2id[term]
                    # print(term_id)
                    X_weights_dataset[doc_id][term_id] = weights
    # print(X_weights_dataset)
    return token2id, id2token, X_weights_dataset, X_labels, idf_dic


'''
compute Euclidean distance of two vectors 
'''


def compute_distance(v1, v2):
    a = 0

    set_of_term_ids = set(v1.keys()).union(set(v2.keys()))

    for term_id in set_of_term_ids:
        a+=(v1.get(term_id, 0.0) - v2.get(term_id,0.0))**2

    return math.sqrt(a)


'''
return of set of k documents that are most closes to a test document
we use heap to minimize the time usage to find top-k documents
'''


def find_top_k(X_weights_dataset, test_doc, k, id2token):
    heap_doc_ids = []
    num_doc = len(X_weights_dataset)
    for doc_id in range(num_doc):
        cos_score = (-1) * compute_distance(X_weights_dataset[doc_id], test_doc)
        # print(doc_id, cos_score)
        if len(heap_doc_ids) < k:
            heapq.heappush(heap_doc_ids, (cos_score, doc_id))
        else:
            doc_score_id = heapq.heappushpop(heap_doc_ids, (cos_score, doc_id))

    doc_ids = [i[1] for i in heap_doc_ids]
    return doc_ids


'''
for a set of documents that have top-k smallest distance
this function find the majority classes for this k documents
'''


def find_majority_class(doc_ids_set, X_labels):
    output_dic = defaultdict(int)
    for each_class in LABELS_DIC:
        output_dic[each_class] = 0
    for each_doc_id in doc_ids_set:
        output_dic[X_labels[each_doc_id]] += 1

    max_class = LABELS_DIC[0]
    for each_class in LABELS_DIC:
        if output_dic[each_class] > output_dic[max_class]:
            max_class = each_class

    return max_class


'''
Find the true positive, false positive, false negative and true negative counts
between ground truth and prediction
Based on the [TP, FP, FN, TN],
calculate the corresponding precision and recall, F1-score for each class
and the micro and macro averaging F1 for all classes
parameters
----------
Y_pred: prediction of the Naive Bayes Classifier
Y_true: Ground truth of the test set
'''


def evaluation(Y_pred, Y_true):
    # [TP, FP, FN, TN]
    # print(Y_pred, Y_true)
    # dictionary for storing [TP, FP, FN, TN] for all classes
    accuracy_dict = defaultdict(lambda: [0, 0, 0, 0])
    all_labels = set(Y_true)
    for i in range(len(Y_pred)):
        predict_label = Y_pred[i]
        true_label = Y_true[i]
        cur_set = set()
        # when prediction is the same as ground truth
        # the true positive of current class + 1
        # the false negative of all other classes + 1
        if true_label == predict_label:
            accuracy_dict[predict_label][0] += 1
            cur_set.add(predict_label)
        # otherwise false positive of predicted class + 1
        # false negative of true class + 1
        # false negative of all classes other than predicted class and true class + 1
        else:
            accuracy_dict[predict_label][1] += 1
            accuracy_dict[true_label][2] += 1
            cur_set.add(predict_label)
            cur_set.add(true_label)
        other_labels = all_labels - cur_set
        # print(other_labels)
        for label in other_labels:
            accuracy_dict[label][-1] += 1
    # [TP, FP, FN, TN]
    aggregate = [0 for _ in range(4)]
    macro_F1 = 0
    # apply formula to TP, FP, FN, TN for calculating precision, recall and F1 score
    for label in accuracy_dict.keys():
        TP, FP, FN, TN = accuracy_dict[label]
        precision = TP / (TP + FP) if TP != 0 else 0
        recall = TP / (TP + FN) if TP != 0 else 0
        F1 = 2 * precision * recall / (precision + recall) if precision * recall != 0 else 0
        print("Class: %s, True Positive: %d, False Positive: %d, False Negative: %d and True Negative: %d,"
              " Precision: %f, Recall: %f, F1 score: %f"
              % (label, TP, FP, FN, TN, precision, recall, F1))
        macro_F1 += F1
        aggregate = [aggregate[i] + accuracy_dict[label][i] for i in range(4)]
    aTP, aFP, aFN, aTN = aggregate
    aprecision = aTP / (aTP + aFP) if aTP != 0 else 0
    arecall = aTP / (aTP + aFN) if aTP != 0 else 0
    micro_F1 = 2 * aprecision * arecall / (aprecision + arecall) if aprecision * arecall != 0 else 0
    print("micro-averaged F1: %f" % micro_F1)
    macro_F1 /= len(all_labels)
    print("macro-averaged F1: %f" % macro_F1)
    return macro_F1


def construct_val_set(test_json_path, token2id, idf_dic, id2token):
    # load model from test dataset path
    test_file = open(test_json_path)
    test_corpus = json.load(test_file)  # [0:1]
    test_data = []
    test_label = []
    for idx, doc in enumerate(test_corpus):
        label = doc["category"]
        long_string = doc["text"]
        test_data.append(long_string)
        test_label.append(label)

    # construct a dictionary save all tf values for each term of each documents in test set
    X_tf_dataset = defaultdict(lambda: defaultdict(lambda: 0))
    for text_id, each_text in enumerate(test_data):
        token_list = preprocess(each_text)
        for each_token in token_list:
            if each_token in token2id.keys():
                X_tf_dataset[text_id][token2id[each_token]] += 1

    # use l rule to compute tf which is 1+log(tf)
    for i in range(len(test_corpus)):
        for term_id in X_tf_dataset[i]:
            if X_tf_dataset[i][term_id] == 0:
                pass
            else:
                X_tf_dataset[i][term_id] = 1 + math.log10(X_tf_dataset[i][term_id])

    # construct a dictionary to save all weights vector of each test document, omitting 0
    X_test_dataset = defaultdict(lambda: defaultdict(lambda: 0))
    for text_id in X_tf_dataset.keys():
        for term_id in X_tf_dataset[text_id].keys():
            X_test_dataset[text_id][term_id] = X_tf_dataset[text_id][term_id] * idf_dic[id2token[term_id]]
    return test_label, X_tf_dataset, X_test_dataset

'''
main function to inference for a given test dataset path, an int k, training model path
print out time evaluation result
'''


def inference(model_path, val_json_path, test_json_path):
    HYPERPARAs = [1, 5, 10, 15, 20]
    token2id, id2token, X_weights_dataset, X_labels, idf_dic = load_model(model_path)

    # use validation set to find the best k among HYPERPARAs
    val_label, X_tf_dataset_val, X_val_dataset = construct_val_set(val_json_path, token2id, idf_dic, id2token)
    best_F1 = float('-inf')
    best_k = None
    for k in HYPERPARAs:
        Y_predict_val = []
        for test_doc_id in X_tf_dataset_val.keys():
            top_k_doc_ids = find_top_k(X_weights_dataset, X_val_dataset[test_doc_id], k, id2token)
            output_class = find_majority_class(top_k_doc_ids, X_labels)
            Y_predict_val.append(output_class)
        print('-----------------Validation Set with respect to k = %d-----------------' % k)
        macro_F1 = evaluation(Y_predict_val, val_label)

        if macro_F1 > best_F1:
            best_F1 = macro_F1
            best_k = k
    print('Best Macro_F1 found on Validation Set is %f, the corresponding k is %d ' % (best_F1, best_k))
    # find top k closest documents in the training set
    # determin the majority class of top k documents
    # add inference result to prediction list
    test_label, X_tf_dataset, X_test_dataset = construct_val_set(test_json_path, token2id, idf_dic, id2token)
    Y_predict = []
    for test_doc_id in X_tf_dataset.keys():  # range(len(test_corpus)):
        top_k_doc_ids = find_top_k(X_weights_dataset, X_test_dataset[test_doc_id], best_k, id2token)
        output_class = find_majority_class(top_k_doc_ids, X_labels)
        Y_predict.append(output_class)
    print('-----------------Test Set with respect to k = %d-----------------' % best_k)
    evaluation(Y_predict, test_label)


if __name__ == "__main__":
    if len(sys.argv) == 4:
        time1 = time.time()
        model_path = sys.argv[1]
        test_json_path = sys.argv[2]
        val_json_path = sys.argv[3]
        if not os.path.exists(model_path):
            sys.exit("No model path exist\n")
        if not os.path.exists(test_json_path):
            sys.exit("No test json path exist\n")
        if not os.path.exists(val_json_path):
            sys.exit("No test json path exist\n")
        else:
            inference(model_path, val_json_path, test_json_path)
            time2 = time.time()
            print("time using: ", time2 - time1)
    else:
        sys.exit("incorrect number of arguments: python3 ./knn/knn_inference.py ./model/knn_model.tsv ./data/test_data.json ./data/val_data.json \n")
