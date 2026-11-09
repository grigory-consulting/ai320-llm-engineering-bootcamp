
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




model.eval() # Evaluierungsmodus
with torch.no_grad(): # Gradient wird nicht mehr benötigt 
    logits = model(X_test_tensor) # Logits, Shape (N, 2)
    predicted = logits.argmax(dim=1) # Klasse mit dem höchsten Logit
    acc = (predicted == y_test_tensor).float().mean().item() 
    print(f'Test set accuracy: {acc*100:.2f}%')


def predict_spam(msg):
    x = vectorizer.transform([msg]).toarray()
    x_tensor = torch.tensor(x, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        pred = model(x_tensor).argmax(dim=1).item()
        return 'SPAM' if pred == 1 else 'Kein Spam'

print(predict_spam('Congratulations, you have won a prize!'))
print(predict_spam('Are we meeting at 7pm for dinner?'))
