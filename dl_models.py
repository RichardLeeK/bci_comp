import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss

class CNN3d(nn.Module):
    def __init__(self):
        super(CNN3d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=32, kernel_size=(3, 3, 3)),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(512, 100),
            nn.ReLU(),
            nn.Linear(100, 4)
        )
        
    def forward(self, x):
        out = self.conv(x)
        dim = 1
        for d in out.size()[1::]:
            dim = dim * d
        out = out.view(-1, dim)
        out = self.fc(out)
        return F.softmax(out, dim=1)

class CNN2d(nn.Module):
  def __init__(self):
    super(CNN2d, self).__init__()
    self.conv = nn.Sequential(
        nn.Conv2d(1, 32, 3, 1),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )
    self.fc = nn.Sequential(
        nn.Linear(1000, 100),
        nn.ReLU(),
        nn.Linear(100, 4)
    )
  def forward(self, x):
    out = self.conv(x)
    dim = 1
    for d in out.size()[1::]:
      dim = dim * d
    out = out.view(-1, dim)
    out = self.fc(out)
    return F.softmax(out, dim=1)



def train(model, device, train_loader, optimizer, epoch):
  model.to(device)
  model.train()

  criterion = CrossEntropyLoss()

  for batch_idx, (data, target) in enumerate(train_loader):
    data, target = Variable(data.to(device)), Variable(target.to(device))
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target.long())
    loss.backward()
    optimizer.step()
    if batch_idx % 5 == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
  model.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(device), target.to(device)
      output = model(data)
      test_loss = CrossEntropyLoss(output, target, reduction='sum').item()
      pred = output.max(1, keepdim=True)[1]
      correct += pred.eq(target.view_as(pred)).sum().item()
  test_loss /= len(test_loader.dataset)
  print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
