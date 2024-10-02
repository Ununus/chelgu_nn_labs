import torch
import matplotlib.pyplot as plt

class MyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(1, 20)
        self.act1 = torch.nn.Sigmoid()
        self.fc2 = torch.nn.Linear(20, 1)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        return x

def main():
    # Подготовка данных
    x_train = torch.rand(50) * 20 - 10
    y_train = torch.sin(x_train) 
    plt.plot(x_train.numpy(), y_train.numpy(), 'o')
    plt.show()
    x_test = torch.linspace(-10, 10, 50)
    my_net = MyNet()
    optimizer = torch.optim.Adam(my_net.parameters(), lr=0.01)
    loss = torch.nn.MSELoss()
    x_train.unsqueeze_(1)
    y_train.unsqueeze_(1)
    # Обучение
    for epoch_number in range(1000):
        optimizer.zero_grad()
        y_pred = my_net(x_train)
        loss_val = loss(y_train, y_pred)
        loss_val.backward()
        optimizer.step()
    # Проверка
    with torch.no_grad():
        y_test = torch.sin(x_test)
        x_test.unsqueeze_(1)
        y_test.unsqueeze_(1)
        y_pred = my_net(x_test)
        plt.plot(x_test.numpy(), y_pred.numpy(), 'o')
        plt.plot(x_test.numpy(), y_test.numpy(), 'v')
        plt.show()

if __name__ == '__main__':
    main()