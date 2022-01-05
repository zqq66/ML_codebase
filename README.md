# ML_codebase

## Naive Bayes Classifier NBC
```
python3 nbc_train.py ./data/train_data.json ./data/val_data.json ./data/test_data.json
```

## K Nearest Neighbour Classifier KNN
Training
```
python3 ./knn/knn_create_model.py ./data/train.json ./model/knn_model.tsv
```
Inferencing
```
python3 ./knn/knn_inference.py ./model/knn_model.tsv ./data/test_data.json ./data/val_data.json
```

## Neural Network (GRU)
```
python3 RNN.py
```
