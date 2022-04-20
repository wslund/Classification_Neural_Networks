import torch
from torchvision import datasets, transforms
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


trainSet = torchvision.datasets.FashionMNIST("./data", download=True, transform=
                                                transforms.Compose([transforms.ToTensor()]))
testSet = torchvision.datasets.FashionMNIST("./data", download=True, train=False, transform=
                                               transforms.Compose([transforms.ToTensor()]))


trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=100)
testLoader = torch.utils.data.DataLoader(testSet, batch_size=100)


print(f' The dataset have : {len(trainSet.classes)} Outcomes')



class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 =  nn.Linear(784, 256)
        self.l2 =  nn.Linear(256, 128)
        self.l3 =  nn.Linear(128, 10)


    def forward(self,x):
        x = x.view(x.shape[0],-1)

        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.log_softmax(self.l3(x), dim = 1)

        return x


def create_classification_model(lossFunction):
  model = Classifier()
  model.to(device)

  if lossFunction == "NLLLoss":
    error = nn.NLLLoss()
  else:
    error = nn.CrossEntropyLoss()

  learningRate = 0.001
  optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

  numEpochs = 5
  count = 0

  lossList = []
  iterationList = []
  accuracyList = []

  predictionsList = []
  labelsList = []

  for epoch in range(numEpochs):
      for images, labels in trainLoader:
          images, labels = images.to(device), labels.to(device)

          train = Variable(images.view(100, 1, 28, 28))
          labels = Variable(labels)

          outputs = model(train)
          loss = error(outputs, labels)

          optimizer.zero_grad()


          loss.backward()

          optimizer.step()

          count += 1

          if not (count % 50):
              total = 0
              correct = 0

              for images, labels in testLoader:
                  images, labels = images.to(device), labels.to(device)
                  labelsList.append(labels)

                  test = Variable(images.view(100, 1, 28, 28))

                  outputs = model(test)

                  predictions = torch.max(outputs, 1)[1].to(device)
                  predictionsList.append(predictions)
                  correct += (predictions == labels).sum()

                  total += len(labels)

              accuracy = correct * 100 / total
              lossList.append(loss.data)
              iterationList.append(count)
              accuracyList.append(accuracy)

          if not (count % 500):
              print("Iteration: {}, Loss: {}, Accuracy: {}%".format(count, loss.data, accuracy))
  return accuracy


NLLLossAccuracy = create_classification_model("NLLLoss")

CrossEntropyLossAccuracy = create_classification_model("CrossEntropyLoss")


print(f"Negative Log Likelihood Loss Accuracy: {NLLLossAccuracy}")
print(f"Cross Entropy Loss Accuracy: {CrossEntropyLossAccuracy}")