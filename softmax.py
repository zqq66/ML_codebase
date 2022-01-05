# -*- coding: utf-8 -*-


import numpy as np
import struct
import matplotlib.pyplot as plt
from scipy.special import expit

def readMNISTdata():

    with open('t10k-images-idx3-ubyte','rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        test_data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        test_data = test_data.reshape((size, nrows*ncols))
    
    with open('t10k-labels-idx1-ubyte','rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        test_labels = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        test_labels = test_labels.reshape((size,1))
    
    with open('train-images-idx3-ubyte','rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        train_data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        train_data = train_data.reshape((size, nrows*ncols))
    
    with open('train-labels-idx1-ubyte','rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        train_labels = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        train_labels = train_labels.reshape((size,1))

    # augmenting a constant feature of 1 (absorbing the bias term)
    train_data = np.concatenate( ( np.ones([train_data.shape[0],1]), train_data ), axis=1)
    test_data  = np.concatenate( ( np.ones([test_data.shape[0],1]),  test_data ), axis=1)
    np.random.seed(314)
    np.random.shuffle(train_labels)
    np.random.seed(314)
    np.random.shuffle(train_data)

    X_train = train_data[:50000] / 256
    t_train = train_labels[:50000]

    X_val   = train_data[50000:] /256
    t_val   = train_labels[50000:]

    return X_train, t_train, X_val, t_val, test_data, test_labels

def softmax(z):
    # 100 x 10
    max_z = np.max(z, axis=1)
    max_z = max_z.reshape((max_z.shape[0], 1))  # 100 x 1
    # print('z', z)
    exp = np.exp(z - max_z)
    for i in range(z.shape[0]):
        exp[i] /= np.sum(exp[i])
    # denominator = np.sum(np.exp(z), axis=1).reshape((max_z.shape[0], 1))
    return exp


def predict(X, W, t, t_onehot):
    # X_new: Nsample x (d+1) # 100 x 785
    # W: (d+1) x K --- 785 x 10
    Nsample = X.shape[0]
    y = softmax(np.dot(X, W))
    t_hat = np.argmax(y, axis=1).reshape((Nsample, 1))

    # print(t.shape, t_hat.shape)
    acc = (t_hat == t).sum()/Nsample
    loss = 0
    for i in range(Nsample):
        loss -= np.log(y[i, t[i]])
    loss /= Nsample
    return y, t_hat, loss, acc

def onehot(t):
    onehot_t = np.zeros((t.shape[0], N_class))

    for i in range(t.shape[0]):
        onehot_t[i, t[i]] = 1  # N x 10

    return onehot_t

def train(X_train, y_train, X_val, t_val):
    N_train = X_train.shape[0]
    N_val   = X_val.shape[0]
    #TODO Your code here
    d = X_train.shape[1]
    w = np.random.normal(0, 1, size=[d, N_class])  # 785 x 10
    y_one_hot_tr = onehot(y_train)
    y_one_hot_val = onehot(t_val)
    acc_best = 0
    W_best = None
    epoch_best = 0
    acc_train = 0
    losses = []
    accs = []
    for epoch in range(MaxEpoch):
        loss_this_epoch = 0
        acc_this_epoch = 0
        num_batch = int(np.ceil(N_train / batch_size))
        for i in range(num_batch):
            X_batch = X_train[i * batch_size: (i + 1) * batch_size]
            y_batch = y_train[i * batch_size: (i + 1) * batch_size]
            # print('y_batch', y_batch.shape)
            y_one_hot_batch = y_one_hot_tr[i * batch_size: (i + 1) * batch_size]
            # print('y_one_hot', y_one_hot.shape)
            y_hat_batch, t_hat_batch, loss_batch, acc = predict(X_batch, w, y_batch, y_one_hot_batch)
            # print('y_one_hot_batch', y_one_hot_batch)
            loss_this_epoch += loss_batch * batch_size
            acc_this_epoch += acc * batch_size
            M = X_batch.shape[0]  # 100 x 785
            # print('y_hat_batch', y_hat_batch.shape) # 100 x 10
            gradient_w = 1 / M * np.dot(np.transpose(X_batch), (y_hat_batch - y_one_hot_batch))
            # print(gradient_b)
            w = w - gradient_w * alpha
        loss_this_epoch /= N_train
        losses.append(loss_this_epoch)
        acc_this_epoch /= N_train
        print('acc for this epoch', epoch, acc_this_epoch)
        y_hat_val, t_hat_val, loss_val, acc_val = predict(X_val, w, t_val, y_one_hot_val)
        accs.append(acc_val)
        if acc_val > acc_best:
            acc_best = acc_val
            W_best = w
            epoch_best = epoch
            acc_train = acc_this_epoch
    plt.title("Training_Loss")
    plt.xlabel("# Epoch")
    plt.ylabel("Training Loss")
    plt.plot(range(MaxEpoch), losses, color="red")
    plt.savefig('Training_Loss.jpg')
    plt.show()

    plt.title("Validation Risk")
    plt.xlabel("# Epoch")
    plt.ylabel("Validation Risk")
    plt.plot(range(MaxEpoch), accs, color="red")
    plt.savefig('Validation_Risk.jpg')
    plt.show()
    return epoch_best, acc_best,  W_best, acc_train

def find_best_hyper(X_train, y_train, X_val, t_val):
    N_train = X_train.shape[0]
    N_val = X_val.shape[0]
    # TODO Your code here
    d = X_train.shape[1]
    w = np.random.normal(0, 1, size=[d, N_class])  # 785 x 10
    y_one_hot_tr = onehot(y_train)
    y_one_hot_val = onehot(t_val)
    acc_best = 0
    W_best = None
    alpha_best = 0
    acc_train = 0
    losses = []
    accs = []
    for alpha in all_alpha:
        for epoch in range(MaxEpoch):
            loss_this_epoch = 0
            acc_this_epoch = 0
            num_batch = int(np.ceil(N_train / batch_size))
            for i in range(num_batch):
                X_batch = X_train[i * batch_size: (i + 1) * batch_size]
                y_batch = y_train[i * batch_size: (i + 1) * batch_size]
                # print('y_batch', y_batch.shape)
                y_one_hot_batch = y_one_hot_tr[i * batch_size: (i + 1) * batch_size]
                # print('y_one_hot', y_one_hot.shape)
                y_hat_batch, t_hat_batch, loss_batch, acc = predict(X_batch, w, y_batch, y_one_hot_batch)
                # print('y_one_hot_batch', y_one_hot_batch)
                loss_this_epoch += loss_batch * batch_size
                acc_this_epoch += acc * batch_size
                M = X_batch.shape[0]  # 100 x 785
                # print('y_hat_batch', y_hat_batch.shape) # 100 x 10
                gradient_w = 1 / M * np.dot(np.transpose(X_batch), (y_hat_batch - y_one_hot_batch))
                # print(gradient_b)
                w = w - gradient_w * alpha

        y_hat_val, t_hat_val, loss_val, acc_val = predict(X_val, w, t_val, y_one_hot_val)
        y_hat_tr, t_hat_tr, loss_tr, acc_tr = predict(X_train, w, y_train, y_one_hot_tr)
        print('acc for this parameter on training set', alpha, acc_tr)
        print('acc for this parameter on validation set', alpha, acc_val)
        accs.append(acc_val)
        if acc_val > acc_best:
            acc_best = acc_val
            W_best = w
            alpha_best = alpha
            acc_train = acc_tr

    return alpha_best, acc_best, W_best, acc_train
##############################
#Main code starts here
X_train, t_train, X_val, t_val, X_test, t_test = readMNISTdata()


print(X_train.shape, t_train.shape, X_val.shape, t_val.shape, X_test.shape, t_test.shape)

N_class = 10

alpha = 0.1      # learning rate
all_alpha = [0.5, 0.1, 0.05, 0.01, 0.005]
batch_size = 100    # batch size = 100
MaxEpoch = 50    # Maximum epoch = 50
decay = 0.          # weight decay

# epoch_best, acc_best, W_best, acc_train = train(X_train, t_train, X_val, t_val)
y_one_hot_test = onehot(t_test)  # N x 10
# _, _, _, acc_test = predict(X_test, W_best, t_test, y_one_hot_test)
# print('At epoch', epoch_best, 'val: ', acc_best, 'test:', acc_test, 'train:', acc_train)

alpha_best, acc_best, W_best, acc_train = find_best_hyper(X_train, t_train, X_val, t_val)
_, _, _, acc_test = predict(X_test, W_best, t_test, y_one_hot_test)
print('At parameter', alpha_best, 'val: ', acc_best, 'test:', acc_test, 'train:', acc_train)
