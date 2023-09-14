print("Classifying MNIST with a fully-connected PyTorch network with one hidden layer")

import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose

train_data = datasets.MNIST(
    root='data',
    train=True,
    transform=Compose(
        [ToTensor(),
         Lambda(lambda x: torch.flatten(x))]),
    download=True,
)
validation_data = datasets.MNIST(
    root='data',
    train=False,
    transform=Compose(
        [ToTensor(),
         Lambda(lambda x: torch.flatten(x))]),
)

from torch.utils.data import DataLoader

train_loader = DataLoader(
    dataset=train_data,
    batch_size=100,
    shuffle=True)

validation_loader = DataLoader(
    dataset=validation_data,
    batch_size=100,
    shuffle=True)

torch.manual_seed(1234)

hidden_units = 100
classes = 10

model = torch.nn.Sequential(
    torch.nn.Linear(28 * 28, hidden_units),
    torch.nn.BatchNorm1d(hidden_units),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_units, classes),
)

cost_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())


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


epochs = 20
for epoch in range(epochs):
    print('Epoch {}/{}'.format(epoch + 1, epochs))
    train_model(model, cost_func, optimizer, train_loader)

test_model(model, cost_func, validation_loader)
