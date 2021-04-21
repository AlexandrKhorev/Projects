import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as torch
import numpy as np


def generator(amount=1):
    # x_array = np.random.randint(0, 100, size=4)
    # x_array = np.random.random(size=4) - 0.5

    x_array = np.abs(np.random.normal(0, 1, (amount, 4)))
    y_array = np.array([[x[:2].mean(), x[2:].mean()] for x in x_array])

    return x_array, y_array


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 3)
        self.fc2 = nn.Linear(3, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x


net = Net()
x, _ = generator()
x = torch.tensor(x, dtype=torch.float)

y = torch.ones([1, 4])

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

outputs = net(x)
print(x)
print(outputs)

#  for epoch in range(2):  # loop over the dataset multiple times

#     running_loss = 0.0
#     for i, data in enumerate(trainloader, 0):
#         # get the inputs; data is a list of [inputs, labels]
#         inputs, labels = data

#         # zero the parameter gradients
#         optimizer.zero_grad()

#         # forward + backward + optimize
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         # # print statistics
#         # running_loss += loss.item()
#         # if i % 2000 == 1999:    # print every 2000 mini-batches
#         #     print('[%d, %5d] loss: %.3f' %
#         #           (epoch + 1, i + 1, running_loss / 2000))
#         #     running_loss = 0.0

# print('Finished Training')
