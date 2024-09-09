import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import random
import interslice

#Create class for nural network.  4 inputs, 2 hidden layers with 8 and 9 nodes respectivley, and output layer with 3 outputs.
class Model(nn.Module):
    def __init__(self, in_features=4, h1=8, h2=9, out_features=3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)
#Sets up our forward pass using relu activation function.  Better than sigmoid.
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)

        return x
#Set manual seed and instantiates model class
torch.manual_seed(41)
model = Model()

#Interslice all data
X_train, y_train, X_test, y_test, output_key = interslice.interslice("iris.data")

#Convert data to tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

y_train = torch.LongTensor(y_train)
y_train = torch.squeeze(y_train)
y_test = torch.LongTensor(y_test)
y_test = torch.squeeze(y_test)

#Prints our testing data (predictions)
i = 0
while i < 30:
    print(X_test[i], y_test[i], i)
    i += 1
    
#Measures the error (How far off the data is from our predictions)    
criterion = nn.CrossEntropyLoss()

#Re-adjusts model based on error data and sets our learning rate.  Using the Adam optimiser.
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print(model.parameters)

#This is where the training happens.  Epochs represent how many training passes we make and losses will record a time based record of our losses
epoch = 100
losses = []
for i in range(epoch):
    #Our forward pass.  Lets tty and predict the output using the training data.
    y_pred = model.forward(X_train)
    #calculate loss for this epoch
    loss = criterion(y_pred, y_train)
    #Record loss for plotting
    losses.append(loss.detach().numpy())
    #Show loss rate every 10 epochs
    if i % 10 == 0:
        print(f'Epoch: {i} and loss: {loss}')

    #Calculate our gradients
    optimizer.zero_grad()
    #Apply the gradients to our weights and bias
    loss.backward()
    #Tell the optimizer its time for the next epoch
    optimizer.step()

#Plotting error    
plt.plot(range(epoch), losses)
plt.ylabel("loss/error")
plt.xlabel("Epochs")
plt.show()

#********************TEST********************
#Telling pytorch to not record loss or correct itself
with torch.no_grad():
    #Forward pass to predict
    y_eval = model.forward(X_test)
    #Recording a loss for comparison to training
    loss = criterion(y_eval, y_test)

print(loss)

#Creates a chart of results
correct = 0
with torch.no_grad():
    for i, data in enumerate(X_test):
        y_val = model.forward(data)

        print(f'{i+1}.)  {str(y_val)} \t {y_test[i]} \t {y_val.argmax().item()}')

        if y_val.argmax().item() == y_test[i]:
            correct += 1

print(f'We got {correct} correct!')


