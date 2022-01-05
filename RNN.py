import torch
import torch.nn as nn
import torch.optim as optim
from knn.knn_inference import evaluation
from torchtext.data import Field, BucketIterator, TabularDataset
import matplotlib.pyplot as plt
import math


BATCHSIZE = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers)
        self.fc_out = nn.Linear(hid_dim, 1)
        self.activation = nn.Sigmoid()
        # self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len):
        # src = [src len, batch size]
        embedded = self.embedding(src)
        # embedded = [src len, batch size, emb dim]
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len.cpu())
        packed_outputs, hidden = self.rnn(packed_embedded)
        # packed_outputs is a packed sequence containing all hidden states
        # hidden is now from the final non-padded element in the batc
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)
        # outputs = [src len, batch size, hid dim * n directions]

        # outputs are always from the top hidden layer
        output = self.fc_out(hidden.squeeze(0))
        # print('output', output.shape)
        prob = self.activation(output)
        # print(prob)

        return prob

    def initHidden(self):
        return torch.zeros(1, BATCHSIZE, self.hid_dim, device=device)


def train(model, iterator, optimizer, criterion, clip):
    model.train()

    epoch_loss = 0
    preds = []
    targets = []
    for i, batch in enumerate(iterator):
        src, src_len = batch.src
        trg = batch.trg
        # print("src.shape", src.shape)
        optimizer.zero_grad()
        output = model(src, src_len)
        # trg = [1, batch size]
        # output = [1, batch size]
        # pred = torch.argmax(output, dim=1)
        pred = []
        for p in output:
            if p > 0.5:
                pred.append(1)
            else:
                pred.append(0)
        preds += pred
        targets += trg.tolist()
        loss = criterion(output, trg.unsqueeze(1).to(torch.float32))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    print('-------------------Measurements on Training Set-------------------')
    F1 = evaluation(preds, targets)
    return epoch_loss / len(iterator), F1


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    preds = []
    targets = []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src, src_len = batch.src
            trg = batch.trg
            output = model(src, src_len)  # turn off teacher forcing
            # pred = torch.argmax(output, dim=1)
            # pred = output > 0.5
            # preds += pred.tolist()
            pred = []
            for p in output:
                if p > 0.5:
                    pred.append(1)
                else:
                    pred.append(0)
            preds += pred
            targets += trg.tolist()
            loss = criterion(output, trg.unsqueeze(1).to(torch.float32))
            epoch_loss += loss.item()  # add loss iteratively from batch
    print('-------------------Measurements on Validation Set-------------------')
    F1 = evaluation(preds, targets)
    return epoch_loss / len(iterator), F1

def prediction(model, iterator):
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src, src_len = batch.src
            trg = batch.trg
            output = model(src, src_len)  # turn off teacher forcing
            # pred = torch.argmax(output, dim=1)
            # pred = output > 0.5
            # preds += pred.tolist()
            pred = []
            for p in output:
                if p > 0.5:
                    pred.append(1)
                else:
                    pred.append(0)
            preds += pred
            targets += trg.tolist()
    evaluation(preds, targets)


if __name__ == '__main__':
    train_path = './data/train_data.csv'
    test_path = './data/test_data.csv'
    valid_path = './data/val_data.csv'
    SRC = Field(lower=True,
                include_lengths=True)
    LABEL = Field(sequential=False, use_vocab=False)

    train_data = TabularDataset(train_path,
                                format="csv",
                                skip_header=True,
                                fields=[("src", SRC), ('trg', LABEL)])

    valid_data = TabularDataset(valid_path,
                                format="csv",
                                skip_header=True,
                                fields=[("src", SRC), ('trg', LABEL)])

    test_data = TabularDataset(test_path,
                               format="csv",
                               skip_header=True,
                               fields=[("src", SRC), ('trg', LABEL)])
    print(len(train_data))
    SRC.build_vocab(train_data)
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=BATCHSIZE,
        sort_within_batch=True,
        sort_key=lambda x: len(x.src),
        sort=True,
        device=device)
    # size of vocabulary
    INPUT_DIM = len(SRC.vocab)
    ENC_EMB_DIM = 32
    HID_DIM = 5
    N_LAYERS = 1
    LEARNING_RATE = 0.001
    W_DECAY = 0.0005

    model = Model(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE,  weight_decay=W_DECAY)

    N_EPOCHS = 25
    CLIP = 1

    best_valid_loss = float('inf')
    best_valid_F1 = float('-inf')
    train_losses = []
    val_losses = []
    train_F1s = []
    val_F1s = []
    for epoch in range(N_EPOCHS):
        train_loss, train_F1 = train(model, train_iterator, optimizer, criterion, CLIP)
        valid_loss, val_F1 = evaluate(model, valid_iterator, criterion)
        if val_F1 > best_valid_F1:
            best_valid_F1 = val_F1
            torch.save(model.state_dict(), './model/tut4-model.pt')
            print('model saved in', epoch)
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
        train_losses.append(train_loss)
        train_F1s.append(train_F1)
        val_losses.append(valid_loss)
        val_F1s.append(val_F1)
    model.load_state_dict(torch.load('./model/tut4-model.pt'))
    print('-------------------Measurements on TestSet-------------------')
    prediction(model, test_iterator)
    plt.title("Loss")
    plt.xlabel("# Epoch")
    plt.ylabel("Loss")
    plt.plot(range(N_EPOCHS), train_losses, color="red")
    plt.plot(range(N_EPOCHS), val_losses, color="blue")
    plt.savefig('Loss.png')
    plt.show()

    plt.title("Accuracy")
    plt.xlabel("# Epoch")
    plt.ylabel("Accuracy")
    plt.plot(range(N_EPOCHS), train_F1s, color="red")
    plt.plot(range(N_EPOCHS), val_F1s, color="blue")
    plt.savefig('Accuracy.png')
    plt.show()