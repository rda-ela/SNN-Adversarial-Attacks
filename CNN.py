import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import foolbox as fb
import foolbox.attacks as fa
from foolbox import PyTorchModel, accuracy, samples

import numpy as np

import statistics


n_epochs = 1
batchsize = 64
learning_rate = 0.001
momentum = 0.9
log_interval = 100


transform = transforms.Compose(
    [transforms.Resize(32),
     transforms.ToTensor(),
     torchvision.transforms.Normalize((0.1307,), (0.3081,))])

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batchsize,
                                          shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batchsize,
                                         shuffle=False)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5,stride=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(120,84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)),(2,2))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


network = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(network.parameters(), lr=learning_rate)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

def train(epoch):
  network.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = network(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
      train_losses.append(loss.item())
      train_counter.append(
        (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))

def test():
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      output = network(data)
      test_loss += criterion(output, target).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

print(" ")
for epoch in range(1, n_epochs + 1):
  train(epoch)
  test()


#This function does the same work as foolbox, it allows us to compare the values of foolbox
#It's only used to check if there is no issues with the model given to foolbox

def benchmark():
    listeimg = []
    listeadv = []
    for i in range(len(images)):
      imaj = images[i]
      img = imaj
      listeimg.append(img)
      tempadvlist = []
      for e in range(len(epsilons)):
        iadvs = advs[e][i]
        adv = iadvs
        tempadvlist.append(adv)
      listeadv.append(tempadvlist)

    results = []

    model.eval()
    for e in range(len(epsilons)):
      tempepsilons = []
      for i in range(len(listeadv)):
        imgadvtest = listeadv[i][e]
        lb = labels[i].item()
        with torch.no_grad():
          output = model(imgadvtest.unsqueeze(0))
        pred = output.data.max(1, keepdim=True)[1][0].item()
        prob = torch.nn.functional.softmax(output, dim=1)
        top_p, top_class = prob.topk(1, dim = 1)
        if(str(lb)==str(pred)):
          tempepsilons.append("FALSE")
        else:
          tempepsilons.append("TRUE")
      results.append(tempepsilons)

    for i in range(len(results)):
        v = 0
        for e in range(len(results[0])):
            if results[i][e] == "FALSE":
                v = v+1
        print("eps: "+ str(epsilons[i]), end=' | ')
        print("pred en %: " + str(float(v/len(results[0]))*100))
        print("")


adv_loader = torch.utils.data.DataLoader(testset, batch_size=batchsize,
                                         shuffle=False)

examples = enumerate(adv_loader)
batch_idx, (images, labels) = next(examples)


model = network
print(model)


fmodel = fb.PyTorchModel(model, bounds=(0, 1))

print(" ")
print("accuracy", end=' | ')
print(accuracy(fmodel, images, labels))
print("")

attacks = [fb.attacks.PGD()]

epsilons = [
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
    1.1,
    1.2,
    1.3,
    1.4,
    1.5,
    1.6,
    1.7,
    1.8,
    1.9,
    2.0,
]

print("epsilons")
print(epsilons)
print("")


lbyat = []
fulatac = []
for i, attack in enumerate(attacks):
    _, advs, success = attack(fmodel, images, labels, epsilons=epsilons)
    benchmark()
    atacc = []
    robust_accuracy = 1 - success.float().mean(axis=-1)
    for eps, acc in zip(epsilons, robust_accuracy):
        print(attack, eps, acc.item())
        atacc.append(acc.item())
    print(" ")
    fulatac.append(atacc)
    lfull = []
    for e in range(len(epsilons)):
        l = []
        for i in range(len(images)):
            perturbation = advs[e][i][0].numpy() - images[i][0].numpy()
            l.append(float(format(np.linalg.norm(perturbation.flatten()))))
        lfull.append(statistics.mean(l))
    lbyat.append(lfull)
