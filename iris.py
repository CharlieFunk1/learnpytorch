import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


class Model(nn.Module):
    def __init__(self, in_features=4, h1=8, h2=9, out_features=3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)

        return x

iris_x_data = np.loadtxt("irisx.txt", dtype=float, delimiter=",")
iris_y_data = np.loadtxt("irisy.txt", dtype=float, delimiter=",")

print(iris_x_data.shape[1])
#print(iris_x_data)
#print(iris_y_data)

torch.manual_seed(41)
model = Model()

train_x_split = int(0.8 * len(iris_x_data))
train_y_split = int(0.8 * len(iris_y_data))
X_train, X_test = iris_x_data[:train_x_split], iris_x_data[train_x_split:]
y_train, y_test = iris_y_data[:train_y_split], iris_y_data[train_y_split:]

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

#print (X_train)
#print (X_test)
#print (y_train)
#print (y_test)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print(model.parameters)

epoch = 100
losses = []
for i in range(epoch):
    y_pred = model.forward(X_train)

    loss = criterion(y_pred, y_train)

    losses.append(loss.detach().numpy())

    if i % 10 == 0:
        print(f'Epoch: {i} and loss: {loss}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    
plt.plot(range(epoch), losses)
plt.ylabel("loss/error")
plt.xlabel("Epochs")
plt.show()
