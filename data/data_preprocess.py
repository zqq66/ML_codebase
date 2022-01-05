import json
import pandas as pd

# train Data
data = pd.read_csv("raw_train.csv")
data = data.rename(columns={"Label": "category", "Content": "text"})  # rename the data column
data = data.sample(frac=1)
train_data = data.tail(1400) # shuffle the df
val_data = data.head(400)
result = train_data.to_json(orient="records")
parsed = json.loads(result)
with open('train_data.json', 'w') as f:
    json.dump(parsed, f)
result = val_data.to_json(orient="records")
parsed = json.loads(result)
with open('val_data.json', 'w') as f:
    json.dump(parsed, f)
# print(train_json)
# for rnn
label_map = {'pos': 1, 'neg': 0}
train_data['label'] = train_data['category'].map(label_map)
train_data.drop(['category'], inplace=True, axis=1)
train_data.to_csv('train_data.csv', encoding='utf-8', index=False)

val_data['label'] = val_data['category'].map(label_map)
val_data.drop(['category'], inplace=True, axis=1)
val_data.to_csv('val_data.csv', encoding='utf-8', index=False)

test_data = pd.read_csv("raw_test.csv")
test_data = test_data.sample(frac=1) # shuffle the df
test_data = test_data.rename(columns={"Label": "category", "Content": "text"})  # rename the data column

result = test_data.to_json(orient="records")
parsed = json.loads(result)
with open('test_data.json', 'w') as f:
    json.dump(parsed, f)
print(len(parsed))
test_data['label'] = test_data['category'].map(label_map)
test_data.drop(['category'], inplace=True, axis=1)
test_data.to_csv('test_data.csv', encoding='utf-8', index=False)