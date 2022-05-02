import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from torch import nn
import time

def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=(3, 3), padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)
def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    # 卷积层部分
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        # 全连接层部分
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10))
class Accumulator:  #@save
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
def accuracy(y_hat, y):  #@save
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())
def evaluate_accuracy(net, data_iter):  #@save
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            X = X.cuda()
            y = y.cuda()
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]
def get_fashion_mnist_labels(labels):  #@save
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]
def train(net, train_iter,test_iter, loss, trainer,num_epochs=10):

    start_time = time.time()
    if isinstance(net, torch.nn.Module):
        net.train()
        # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for epoch in range(num_epochs):
      for X, y in train_iter:
        X=X.cuda()
        y=y.cuda()
        l = loss(net(X) ,y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
        metric.add(float(l.sum()), accuracy(net(X), y), y.numel())
    # 返回训练损失和训练精度
      end_time = time.time()
      test_acc = evaluate_accuracy(net, test_iter)
      print(f'epoch {epoch + 1}, loss {metric[0] / metric[2]:f}, train_acc {metric[1] / metric[2]:f}, test_acc {test_acc:f}, time {end_time - start_time:f}')
def predict(net, test_iter, n=20):  #@save
    """预测标签（定义见第3章）"""
    for X, y in test_iter:
        X = X.cuda()
        y = y.cuda()
        break
    trues = get_fashion_mnist_labels(y)
    preds = get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true +'    ' + pred for true, pred in zip(trues, preds)]
    print(X.shape)
    print(titles[0:n])
if __name__ == "__main__":
    batch_size=645
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0,], [1,]),
         transforms.Resize(224)])
    mnist_train = torchvision.datasets.FashionMNIST(
    root="../data", train=True, transform=transform, download=False)
    mnist_test = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=transform, download=False)
    train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                             num_workers=4)
    test_iter = data.DataLoader(mnist_test, batch_size, shuffle=True,
                             num_workers=4)
    conv_arch = ((2, 64), (2, 128), (3, 256), (3, 512), (3, 512))
    net = vgg(conv_arch).cuda()
    loss = nn.CrossEntropyLoss().cuda()
    trainer = torch.optim.SGD(net.parameters(), lr=0.05)
    train(net, train_iter,test_iter, loss, trainer)
    predict(net, test_iter)