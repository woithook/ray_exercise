import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import ray
from ray import tune
import numpy as np
import os


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class Net(nn.Module):
    def __init__(self, l1=120, l2=84):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, l1)
        self.fc2 = nn.Linear(l1, l2)
        self.fc3 = nn.Linear(l2, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(net, optimizer):

    criterion = nn.CrossEntropyLoss()

    for epoch in range(10):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

        acc = test(net)
        with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save(
                (net.state_dict(), optimizer.state_dict()), path
            )
        tune.report(accuracy=acc)

    print('Finished Training')


def test(net):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = correct / total

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * acc))
    return acc


def tunning_func(config, checkpoint_dir=None):
    # Hyperparameters
    l1, l2 = config["l1"], config["l2"]
    lr = config["lr"]
    momentum = config["momentum"]

    net = Net(l1, l2)
    optimizer = optim.SGD(net.parameters(), lr, momentum)

    # The `checkpoint_dir` parameter gets passed by Ray Tune when a checkpoint
    # should be restored.
    if checkpoint_dir:
        checkpoint = os.path.join(checkpoint_dir, "checkpoint")
        model_state, optimizer_state = torch.load(checkpoint)
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    train(net, optimizer)


if __name__ == "__main__":
    checkpoint_dir = r"C:\Users\zzw93\ray_results\tunning_func_2020-12-08_11-00-49\tunning_func_37929_00001_1_l1=4,l2=256,lr=0.051453,momentum=0.72314_2020-12-08_11-02-08\checkpoint_9"
    ray.init(num_cpus=2, num_gpus=0)
    result = tune.run(
        tunning_func,
        config={
            "l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
            "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
            "lr": tune.loguniform(0.001, 0.1),
            "momentum": tune.uniform(0.1, 0.9),
        },
        num_samples=2,
        restore=checkpoint_dir
    )

    df = result.results_df
    best_trial = result.get_best_trial("accuracy", "max", "last")
    print("Best trial config: {}".format(best_trial.config))


# if __name__ == "__main__":
#     net = Net(10, 20)
#     train(net, lr=0.001, momentum=0.9)
#     acc = test(net)
#     print("accuracy = ", acc)
