import torch
import matplotlib.pyplot as plt
import multiprocessing
import torchvision
from torchvision import transforms
from einops import rearrange
import matplotlib.patches as mpatches

if __name__ == "__main__":
    batch_size = 12*2
    cpu_count = multiprocessing.cpu_count()
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize((64, 64)),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=cpu_count)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    f, axs = plt.subplots(batch_size // 3, 3)
    f_scale = 5
    f.set_figheight(f_scale* batch_size // 3)
    f.set_figwidth(f_scale*3)
     
    model = torch.load('model')
    for data, target in testloader:
        data = data.to('cuda')
        rs = model(data)
        pred = rs.data.max(1)[1]
        for one_pred, one_target, img, ax in zip(pred, target, data, axs.flatten()):
            ax.imshow(rearrange(img.cpu(), 'c h w -> h w c'))
            ax.legend(handles = [mpatches.Patch(label = f'pred {classes[one_pred]}'),
            mpatches.Patch( label = f'target {classes[one_target]}') ])
        plt.savefig('output.jpg')
        break