import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import random
from softmaxModel import SoftmaxModel


device = "cuda"
torch.cuda.manual_seed_all(777)

mnist_test = dsets.MNIST(root='MNIST_data/',
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)

model = SoftmaxModel()
model.load_state_dict(torch.load("softmax/out/softmaxModel"))
model.eval()

# 테스트 데이터를 사용하여 모델을 테스트한다.
with torch.no_grad():  # torch.no_grad()를 하면 gradient 계산을 수행하지 않는다.
    X_test = mnist_test.test_data.view(-1, 28 * 28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)

    prediction = model.forward(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())

    # MNIST 테스트 데이터에서 무작위로 하나를 뽑아서 예측을 해본다
    for i in range(20):
        r = random.randint(0, len(mnist_test) - 1)
        X_single_data = mnist_test.test_data[r:r + 1].view(-1, 28 * 28).float().to(device)
        Y_single_data = mnist_test.test_labels[r:r + 1].to(device)

        print('Label: ', Y_single_data.item())
        single_prediction = model.forward(X_single_data)
        print('Prediction: ', torch.argmax(single_prediction, 1).item())

        plt.imshow(mnist_test.test_data[r:r + 1].view(28, 28), cmap='Greys', interpolation='nearest')
        plt.show()
