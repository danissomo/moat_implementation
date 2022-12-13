import torch
import torchvision
from torchvision import transforms
import multiprocessing
import wandb
import time
import sys
import torch.optim as optim
from model_torch import MOAT_block
import torch.nn as nn
def draw_progressbar(count, total, status=''):
    '''
    modified
    https://gist.github.com/vladignatyev/06860ec2040cb497f0f3
    '''
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total -1), 1)
    bar = '=' * filled_len + '>' + '-' * (bar_len - filled_len-1)

    sys.stdout.write(' [%s] %s%s ... %s\r' % (bar, percents, '%', status))
    sys.stdout.flush()
    
class MOAT_project:
    def __init__(self) -> None:
        wandb.init(project="moat_project")
        cpu_count = multiprocessing.cpu_count()
        batch_size = 200
        transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle=True, num_workers=cpu_count)

        self.testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform).__len__()
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size, shuffle=False, num_workers=cpu_count)

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        self.best_weights = None
        self.best_accur = None
        self.start_time = time.time()
        self.model = MOAT_block(len(self.classes), 256, 32)



    def publish_data(self):
        pass

    def launch_train(self):
        self.start_time = time.time()
        epochs = 100
        optimizer = optim.SGD(self.model.parameters(), lr=0.001)
        crit = nn.CrossEntropyLoss()
        correct = 0
        for epoch in range(1, epochs + 1):
            for n, (data, target) in enumerate(self.trainloader):
                data = data.reshape((len(target), 3, 256, 256))
                # print(data.shape)
                
                rs = self.model.forward(data)
                optimizer.zero_grad()
                loss = crit(rs, target)
                loss.backward()
                pred = rs.data.max(1)[1]
                correct += pred.eq(target.data).sum()
                batch_acc = pred.eq(target.data).sum() / len(target)
                optimizer.step()
                wandb.log({'batch_acc': batch_acc, 'batch_loss':loss})
                draw_progressbar(n, self.trainset.__len__(), f'loss {loss : 0.4f}, acc {batch_acc : 0.4f},  time {time.time()-self.start_time : 0.4f}')
            epoc_acc = correct/ len(self.trainloader.dataset)

    

    def print_epoch_summary(self):
        pass


if __name__ == "__main__":
    m = MOAT_project()
    m.launch_train()
