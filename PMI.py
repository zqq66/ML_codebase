import pickle
import math
import time
from collections import Counter
from transformers import *
from data.CoNLL_dataset import Dataset
from queue import PriorityQueue


class DualPriorityQueue(PriorityQueue):
    def __init__(self, maxPQ=False):
        PriorityQueue.__init__(self)
        self.reverse = -1 if maxPQ else 1

    def put(self, priority, data):
        PriorityQueue.put(self, (self.reverse * priority, data))

    def get(self, *args, **kwargs):
        priority, data = PriorityQueue.get(self, *args, **kwargs)
        return self.reverse * priority, data


def findoccurence(data, numcuts=None, tag=None):
    # model_class, tokenizer_class, model_config, pretrained_weights = (BertModel, BertTokenizer, BertConfig, 'bert-large-cased')
    # lm_cache_path = '/Users/qianqiu/Documents/CMPUT499/Unsupervised_Parsing/trees_from_transformers/data/transformers'
    # tokenizer = tokenizer_class.from_pretrained(
    #     pretrained_weights, cache_dir=lm_cache_path, force_download=True)
    # data = Dataset(path=data_path, tokenizer=tokenizer)
    all_phrase = []
    all_words = []
    words_sum = 0
    phrase_sum = 0
    for sent in data:
        # print(sent)
        # sent = sent.lower().split()
        sent = [i.lower() for i in sent]
        words_sum += len(sent)
        phrase_sum += len(sent) - 1
        all_words += sent
        all_phrase += [sent[i] + sent[i+1] for i in range(len(sent)-1)]

    # print(len(data[0]))
    freq_words = Counter(all_words)
    freq_phrase = Counter(all_phrase)
    # print(all_phrase)
    # print(freq_phrase)
    # print(all_words)
    # print(freq_words)

    maxQ = DualPriorityQueue(maxPQ=False)
    sent_len = dict()
    for i in range(len(data)):
        sent = data[i]
        sent = [i.lower() for i in sent]
        sent_len[i] = len(sent)-1
        # print(sent)
        for j in range(len(sent)-1):
            word1 = sent[j]
            word2 = sent[j+1]
            p_ab = freq_phrase[word1+word2] / phrase_sum
            p_a = freq_words[word1] / words_sum
            p_b = freq_words[word2] / words_sum
            # print("sum", phrase_sum, words_sum)
            # print("p_ab", p_ab, "p_a", p_a, "p_b", p_b, word1, word2)
            p = math.log10(p_ab/p_a / p_b)
            # print(p_ab/p_a / p_b)
            maxQ.put(p, (i, j))

    cuts_dic = []
    for i in range(numcuts):
        value, (sent, phrase) = maxQ.get()
        # print(value, (sent, phrase))
        cuts_dic.append((sent, phrase))

    # print(len(cuts_dic))
    print(cuts_dic)
    result_path = '/Users/qianqiu/Documents/CMPUT499/Unsupervised_Parsing/trees_from_transformers/chunking_output'
    with open(f'{result_path}/output_review_PMI.txt', "w") as f:
        for i in range(len(data)):
            f.write("x y B B")
            f.write("\n")
            for j in range(sent_len[i]):
                if (i, j) in cuts_dic:
                    f.write("x y B ")
                else:
                    f.write("x y I ")
                f.write(tag[i][j + 1].upper())
                f.write("\n")
            f.write("\n")


def findnumcuts(tag_data):
    with open(tag_data, 'rb') as f:
        data = pickle.load(f)
        numcuts = 0
        for i in data:
            numcuts += i.count("B") - 1
    print(len(data), len(data[0]))
    return numcuts, data


if __name__ == '__main__':
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    file_token = "/Users/qianqiu/Documents/CMPUT499/UC/data/review/review_test_token.pkl"
    file_tag = "/Users/qianqiu/Documents/CMPUT499/UC/data/review/review_test_tag.pkl"

    output_tag = "/Users/qianqiu/Documents/CMPUT499/Unsupervised_Parsing/data/CoNLL2003/able2use/GT_tag.txt"
    start = time.time()
    numcuts, tag = findnumcuts(file_tag)
    with open(file_token, "rb") as f:
        data = pickle.load(f)
        print(len(data), len(data[0]))
        findoccurence(data, numcuts, tag)
    end = time.time()
    print(end - start)