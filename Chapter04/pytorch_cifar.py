import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

# Training dataset
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225])
])

train_data = datasets.CIFAR10(
    root='data',
    train=True,
    download=True,
    transform=train_transform)

batch_size = 50
train_loader = DataLoader(
    dataset=train_data,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2)

# Validation dataset
validation_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225])
])

validation_data = datasets.CIFAR10(
    root='data',
    train=False,
    download=True,
    transform=validation_transform)

validation_loader = DataLoader(
    dataset=validation_data,
    batch_size=100,
    shuffle=True)

torch.manual_seed(1234)

from torch.nn import Sequential, Conv2d, BatchNorm2d, GELU, MaxPool2d, Dropout2d, Linear, Flatten

model = Sequential(
    Conv2d(in_channels=3, out_channels=32,
           kernel_size=3, padding=1),
    BatchNorm2d(32),
    GELU(),
    Conv2d(in_channels=32, out_channels=32,
           kernel_size=3, padding=1),
    BatchNorm2d(32),
    GELU(),
    MaxPool2d(kernel_size=2, stride=2),
    Dropout2d(0.2),

    Conv2d(in_channels=32, out_channels=64,
           kernel_size=3, padding=1),
    BatchNorm2d(64),
    GELU(),
    Conv2d(in_channels=64, out_channels=64,
           kernel_size=3, padding=1),
    BatchNorm2d(64),
    GELU(),
    MaxPool2d(kernel_size=2, stride=2),
    Dropout2d(p=0.3),

    Conv2d(in_channels=64, out_channels=128,
           kernel_size=3),
    BatchNorm2d(128),
    GELU(),
    Conv2d(in_channels=128, out_channels=128,
           kernel_size=3),
    BatchNorm2d(128),
    GELU(),
    MaxPool2d(kernel_size=2, stride=2),
    Dropout2d(p=0.5),
    Flatten(),

    Linear(512, 10),
)


def train_model(model, cost_function, optimizer, data_loader):
    # send the model to the GPU
    model.to(device)

    # set model to training mode
    model.train()

    current_loss = 0.0
    current_acc = 0

    # iterate over the training data
    for i, (inputs, labels) in enumerate(data_loader):
        # send the input/labels to the GPU
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            # forward
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)
            loss = cost_function(outputs, labels)

            # backward
            loss.backward()
            optimizer.step()

        # statistics
        current_loss += loss.item() * inputs.size(0)
        current_acc += torch.sum(predictions == labels.data)

    total_loss = current_loss / len(data_loader.dataset)
    total_acc = current_acc.double() / len(data_loader.dataset)

    print('Train Loss: {:.4f}; Accuracy: {:.4f}'.format(total_loss, total_acc))


def test_model(model, cost_function, data_loader):
    # send the model to the GPU
    model.to(device)

    # set model in evaluation mode
    model.eval()

    current_loss = 0.0
    current_acc = 0

    # iterate over  the validation data
    for i, (inputs, labels) in enumerate(data_loader):
        # send the input/labels to the GPU
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)
            loss = cost_function(outputs, labels)

        # statistics
        current_loss += loss.item() * inputs.size(0)
        current_acc += torch.sum(predictions == labels.data)

    total_loss = current_loss / len(data_loader.dataset)
    total_acc = current_acc.double() / len(data_loader.dataset)

    print('Test Loss: {:.4f}; Accuracy: {:.4f}'.format(total_loss, total_acc))

    return total_loss, total_acc


cost_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

epochs = 50
for epoch in range(epochs):
    print('Epoch {}/{}'.format(epoch + 1, epochs))
    train_model(model, cost_func, optimizer, train_loader)
    test_model(model, cost_func, validation_loader)
