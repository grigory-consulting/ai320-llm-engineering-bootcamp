
../.venv/bin/jupyter lab --port=8888 --no-browser --IdentityProvider.token=''




from sklearn.feature_extraction.text import CountVectorizer
dataset = dataset['train'].train_test_split(test_size=0.2)
vectorizer = CountVectorizer( binary=False) 
X_train = vectorizer.fit_transform([x['sms'] for x in dataset['train']]).toarray()
X_test = vectorizer.transform([x['sms'] for x in dataset['test']]).toarray()
y_train = [x['label'] for x in dataset['train']]
y_test = [x['label'] for x in dataset['test']]



import torch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

from torch.utils.data import DataLoader, TensorDataset

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(dataset, batch_size=64, shuffle=False)
