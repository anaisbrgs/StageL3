from __future__ import print_function
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import matplotlib.pyplot as plt
import torchvision


# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=10, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', #default = 0.5
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--dimension', type=int, default = 120, metavar='D',
                    help='the dimension of the second neuron network') #ajout de l'argument dimension représentant le nombre de neurone dans la deuxième couche. 
parser.add_argument('--boucle', type=int, default=0, metavar='B',
                   help='boucle pour faire différents couche de la deuxième couche de neurone')# ajout de boucle pour automatiser le nombre de neurone dans la deuxieme couche
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print ('cuda?', args.cuda)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
dataset = torchvision.datasets.ImageFolder('dataset',
                                        transforms.Compose([
                                        transforms.CenterCrop(150),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                                        ]))
train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
test_loader= torch.utils.data.DataLoader(dataset, batch_size=args.test_batch_size, shuffle=True, num_workers=2)

classes = 'blink','left','right','center'

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)#
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(12800, args.dimension)# args.dimension prend la valeur default de l'argument dimension.
        self.fc2 = nn.Linear(args.dimension, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 4))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 4))
        x = x.view(-1, 12800)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))#
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)#
        #x = self.fc2(x)
        return F.log_softmax(x, dim=1)#x

model = Net()


#optimizer = optim.Adam(model.parameters(), lr=opt.lr)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
# define loss function (criterion) and optimizer
criterion = F.nll_loss()
# criterion = nn.CrossEntropyLoss()
# criterion = nn.MSELoss()

if args.cuda:
    model = model.cuda()
    criterion = criterion.cuda()

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader,0):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        # data, target = Variable(data).float(), Variable(target).float()
        data, target = Variable(data).long(), Variable(target).long()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if args.log_interval>0: # rajout de la commande pour pouvoir print ou non les différents epoch ou juste le résultat
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data[0]))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    if args.log_interval>0: print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct / len(test_loader.dataset)

def protocol():
    for epoch in range(1, args.epochs + 1):
        train(epoch)
    return test()

def main():
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        Accuracy = test()
    print('Test set: Final Accuracy: {:.3f}%'.format(Accuracy*100)) # print que le pourcentage de réussite final
    
    
if __name__ == '__main__':  
    if args.boucle == 1: # Pour que la boucle se fasse indiquer --boucle 1
        rho = 10**(1/3) 
        for i in [int (k) for k in rho**np.arange(2,9)]:# i prend les valeur en entier du tuple rho correspondra au nombre de neurone
            args.dimension = i
            print ('La deuxième couche de neurone comporte',i,'neurones')
            main()
    else:
        t0 = time.time () # ajout de la constante de temps t0

        main()

        t1 = time.time () # ajout de la constante de temps t1

        print ("Le programme a mis",t1-t0, "secondes à s'exécuter.") #compare t1 et t0, connaitre le temps d'execution du programme