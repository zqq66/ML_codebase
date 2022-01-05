from hmmlearn.hmm import GaussianHMM, GMMHMM, MultinomialHMM
import pickle
import numpy as np
import random


def findnumcuts(tag_data):
    with open(tag_data, 'rb') as f:
        data = pickle.load(f)
        numcuts = 0
        for i in data:
            numcuts += i.count("B") - 1
    print(len(data), len(data[0]))
    return numcuts, data

def build_vocab(data):
    word2vec = dict()
    vec2word = dict()
    count = 0
    max_iter = 0
    for line in data:
        if len(line) > max_iter:
            max_iter = len(line)
        for token in line:
            if token.lower() not in word2vec.keys():
                word2vec[token.lower()] = count
                count += 1

    return word2vec

def writetotxt(data, tag):
    word2vec = build_vocab(data)
    # print(word2vec)
    x = [word2vec[e.lower()] for e in data[0]]
    X = np.array(x)
    lengths = [len(data[0])]
    for i in range(1, len(data)):
        # print(i, x)
        line = data[i]
        x = [word2vec[e.lower()] for e in line]
        lengths.append(len(x))
        x = np.array(x)
        X = np.concatenate((X, x))
        print('X', X)

    print(X)
    X = X.reshape(-1, 1)
    lengths = np.array(lengths)
    # print(list(lengths))
    print(X)
    X = X.astype('int32')
    # print(max_iter)
    #
    model = MultinomialHMM(n_components=2)
    model.fit(X, lengths)

    # try viterbi
    result_path = '/Users/qianqiu/Documents/CMPUT499/Unsupervised_Parsing/trees_from_transformers/chunking_output'
    # with open(f'{result_path}/output_review_hmm.pkl', "rb") as f:
    #     model = pickle.load(f)
    with open(f'{result_path}/output_review_hmm.pkl', "wb") as f:
        pickle.dump(model, f)

    cuts = model.predict(X, lengths)
    print(list(cuts))
    choice = [0, 1]
    with open(f'{result_path}/output_review_hmm.txt', "w") as f:
        cur_len = 0
        for i in range(len(data)):
            f.write("x y B B")
            f.write("\n")
            for j in range(1, len(data[i])):
                f.write("x y ")
                if cuts[cur_len + j] == 0:
                    f.write("I")
                else:
                    f.write("B")
                f.write(" ")
                f.write(tag[i][j].upper())
                f.write("\n")
            f.write("\n")
            cur_len += lengths[i]


if __name__ == '__main__':
    file_token = "/Users/qianqiu/Documents/CMPUT499/UC/data/review/review_test_token.pkl"
    file_tag = "/Users/qianqiu/Documents/CMPUT499/UC/data/review/review_test_tag.pkl"
    # file_token = "/Users/qianqiu/Documents/CMPUT499/Unsupervised_Parsing/data/CoNLL2003/new_test/german_test2_token.pkl"
    # file_tag = "/Users/qianqiu/Documents/CMPUT499/Unsupervised_Parsing/data/CoNLL2003/new_test/german_test2_tag.pkl"
    numcuts, tag = findnumcuts(file_tag)
    with open(file_token, "rb") as f:
        data = pickle.load(f)
        print(len(data), len(data[0]))
    writetotxt(data, tag)