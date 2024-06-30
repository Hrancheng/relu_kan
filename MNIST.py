
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch_relu_kan import ReLUKANLayer, ReLUKAN
import numpy as np
import os

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True, transform=transform),
    batch_size=64, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, download=True, transform=transform),
    batch_size=64, shuffle=False)

input_size = 28 * 28
relu_kan = ReLUKAN([input_size, 10], 61, 3)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(relu_kan.parameters())

plt.ion()
for epoch in range(100):
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        
        #print("data shape")
        #print(data.shape) #torch.Size([64, 1, 28, 28])
        if data.size(0) != 64:
            continue
        data = data.view(64, input_size, 1)
        target = target.view(64, 1)
        
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        
        output = relu_kan(data)

        loss = criterion(output, target)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if batch_idx % 100 == 99:
            print(f'Epoch {epoch + 1}, Batch {batch_idx + 1}, Loss: {running_loss / 100}')
            with open('loss.csv', 'a') as f:
                f.write(f'{running_loss / 100}\n')
            running_loss = 0.0

    with torch.no_grad():
        example_data, _ = next(iter(test_loader))
        example_data = example_data.view(example_data.size(0), -1)
        if torch.cuda.is_available():
            example_data = example_data.cuda()
        if example_data.size(0) != 64:
            continue
        example_data = example_data.view(64, input_size, 1)
        predictions = relu_kan(example_data).argmax(dim=1)
        plt.clf()
        plt.imshow(example_data[0].cpu().view(28, 28), cmap='gray')
        plt.title(f'Prediction: {predictions[0].item()}')
        if not os.path.exists('pred_result'):
            os.makedirs('pred_result')
        plt.savefig(f'pred_result/{epoch}_{batch_idx}.png')
        plt.pause(0.1)
        
plt.ioff()
plt.show()


