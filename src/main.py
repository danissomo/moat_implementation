import torch
import torchvision
from torchvision import transforms
import multiprocessing
import wandb
import time
import sys
import torch.optim as optim
from model_torch import MOAT_block, CustomMOAT_torch
import torch.nn as nn
from einops import rearrange
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
        self.device = torch.device('cuda')
        wandb.init(project="moat_project")
        cpu_count = multiprocessing.cpu_count()
        batch_size = 100
        self.batch_size = batch_size
        self.imsize = 64
        # уменьшение входного размера картинки существенно улучшило работу
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.Resize((self.imsize, self.imsize)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.Resize((self.imsize, self.imsize)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize((self.imsize, self.imsize)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=cpu_count)
        
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=cpu_count)

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        self.best_weights = None
        self.best_accur = None
        self.start_time = time.time()
        patch_size, channels, dim_head , heads, mlp_hidden_dim = 8, 10, 128, 4, 1000
        self.model = CustomMOAT_torch(len(self.classes), 
                                    self.imsize, 
                                    patch_size, 
                                    channels, 
                                    dim_head , 
                                    heads, 
                                    mlp_hidden_dim).to(self.device)
        wandb.config.update({
            "patch_size"        :       patch_size,      
            "channels"          :       channels,  
            "dim_head"          :       dim_head ,  
            "heads"             :       heads,  
            "mlp_hidden_dim"    :       mlp_hidden_dim,         
        })
        


    def publish_data(self):
        pass
    def launch_train(self):
        self.start_time = time.time()
        epochs = 1000
        optimizer =  optim.AdamW(self.model.parameters(), lr=0.0001)
    
        crit = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10)
        last_acc_train, last_acc_test = 0, 0 
        
        print('dataset len ', len(self.trainloader.dataset))
        for epoch in range(1, epochs + 1):
            correct_train = 0
            correct_test = 0
            loss_train = 0
            loss_test = 0
            self.model.train()
            print("train")
            for n, (data, target) in enumerate(self.trainloader):
                data = data.to(self.device)
                target = target.to(self.device)
                rs = self.model(data)
                optimizer.zero_grad()
                loss = crit(rs, target)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    loss_train += loss
                pred = rs.data.max(1)[1]
                correct_train += pred.eq(target.data).sum()
                batch_acc = pred.eq(target.data).sum() / len(target)
                
                draw_progressbar(n, len(self.trainloader.dataset)/len(target), f'loss {loss : 0.4f}, acc {batch_acc : 0.4f},  time {time.time()-self.start_time : 0.4f}')
                scheduler.step()

            print()
            print("eval")
            self.model.eval()
            for n, (data, target) in enumerate(self.testloader):
                with torch.no_grad():
                    data = data.to(self.device)
                    target = target.to(self.device)
                    rs = self.model(data)
                    loss = crit(rs, target)
                    loss_test+= loss
                    pred = rs.data.max(1)[1]
                    correct_test += pred.eq(target.data).sum()
                    batch_acc = pred.eq(target.data).sum() / len(target)
                draw_progressbar(n, len(self.testloader.dataset)/len(target), f'loss {loss : 0.4f}, acc {batch_acc : 0.4f},  time {time.time()-self.start_time : 0.4f}')

            epoch_acc_test = correct_test/len(self.testloader.dataset)
            epoch_acc_train = correct_train/ len(self.trainloader.dataset)
            loss_test /= len(self.testloader.dataset)/(self.batch_size)
            loss_train /= len(self.trainloader.dataset)/(self.batch_size)
            
            if self.best_accur is None or self.best_accur > epoch_acc_test:
                torch.save(self.model, "model")
                self.best_accur = epoch_acc_test
            print()
            print(f"acc train {epoch_acc_train}, acc test {epoch_acc_test}")
            wandb.log({ 'acc_train_epoch'   :   epoch_acc_train, 
                        'acc_test_epoch'    :   epoch_acc_test,
                        'loss_tarin_epoch'  :   loss_train,
                        'loss_test_epoch'   :   loss_test,
                        # 'lr'                :   optimizer.param_groups[0]['lr'],
                        'inc_acc_test'      :   epoch_acc_test - last_acc_test,
                        'inc_acc_train'     :   epoch_acc_train - last_acc_train,
                        })
            last_acc_test = epoch_acc_test
            last_acc_train = epoch_acc_train
            

    

    def print_epoch_summary(self):
        pass


if __name__ == "__main__":
    
    m = MOAT_project()
    m.launch_train()
