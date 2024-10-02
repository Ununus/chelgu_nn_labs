import torch
import torch.nn as nn # Такие подключения часто используются в программах на torch 
import torch.nn.functional as F # Чтобы не плодить слои, у которых нет обучающих параметров (активации и т.п.), можно вызывать их непосредственно
import torch.optim as optim # Здесь находятся оптимизаторы
from torchvision import datasets, transforms # Это поможет загрузить MNIST датасет
from tqdm import tqdm # Нужен, чтобы красиво отобразить процесс обучения
import matplotlib.pyplot as plt # Для отображения примера циферки

N_EPOCHES = 10 # Кол-во эпох для тренировки
BATCH_SIZE = 500 # Размер батча
DEVICE = 'cpu' # можно 'cuda', если работает

# У меня обучилась до 96.88% за 10 эпох (Мат ожидание 10%)
# Рекорд обучения 99.87% https://paperswithcode.com/sota/image-classification-on-mnist
# Можно модифицировать структуру, чтоб улучшить качество (добавить ещё свёртку, дропаутов)
# Реализация пайторчевских мастеров: https://github.com/pytorch/examples/blob/main/mnist/main.py


class Net(nn.Module): # Класс нашей нейронной сети
    def __init__(self): # Это конструктор
        super(Net, self).__init__() # Вызовем конструктор базового класса (nn.Module)
        self.conv1 = nn.Conv2d(1, 16, 3) # Это свёртка (число_входных_каналов=1, число_выходных_каналов=16, ядро=3, stride=1)
        self.fc1 = nn.Linear(2704, 128) # Полносвязный слой
        self.fc2 = nn.Linear(128, 10) # Ещё один, число_выходов=числу_классов

    def forward(self, x): # Здесь опишем как прогоняется входной тензор через нейросеть
        x = self.conv1(x) # Сначала выполняем свёртку
        x = F.relu(x) # Слой активации ReLU
        x = F.max_pool2d(x, 2) # Можно добавить MaxPool
        x = torch.flatten(x, 1) # Здесь преобразуем в одномерный тензор для передачи в полносвязный слой
        x = self.fc1(x) # Полносвязный слой 1
        x = F.relu(x) # ReLU
        x = self.fc2(x) # Полносвязный слой 2
        output = F.log_softmax(x, dim=1) # можно просто softmax
        return output


def train(model, train_loader, optimizer): # Функция для тренировки сети
    epoch_loss = 0.
    model.train() # Переведём сеть в режим обучения
    for data, target in tqdm(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad() # Шаги всегда одинаковые: 1. Обнуляем градиенты
        output = model(data) # 2. Делам прямой проход
        loss = F.nll_loss(output, target, reduction='sum') # 3. Вычисляем NLLLoss https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html
        # loss = F.cross_entropy(output, target, reduction='sum') можно было так
        loss.backward() # 4. Вычисляем градиенты
        optimizer.step() # 5. Делаем шаг градиентного спуска
        epoch_loss += loss.item() # Подсчитаем loss для статистики
    epoch_loss /= len(train_loader.dataset) # Если loss падает, то сеть обучается, однако может переобучатся, поэтому нужно тестовое множество
    print('\tLoss: {:.6f}'.format(epoch_loss))


def test(model, test_loader): # Функция для теста сети
    model.eval() # Переведём сеть в режим вычисления
    test_loss = 0
    correct = 0
    with torch.no_grad(): # Это обязательно, если мы не обучаем сеть. Так не будут делаться вычисления для градиентов
        for data, target in test_loader: # Как в тренировке, но проще
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data) # Делам прямой проход
            #test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def test1(model, test_dataset): # Функция проверки сети на конкретном примере
    model.eval() # Переведём сеть в режим вычисления
    with torch.no_grad():
        data, target = test_dataset[0] # Просто какой-то пример
        plot_data = data.permute(1, 2, 0).numpy() # Для matplotlib надо в формате (W, H, C) numpy массив
        output = model(data.unsqueeze(0)) # Делам прямой проход
        output = output.argmax(dim=1, keepdim=True).item() # Выбираем наиболее вероятный ответ
        print('Должно быть: {}. Сеть выдала: {}'.format(target, output))
        plt.imshow(plot_data) # Посмотрим картинку
        plt.show()

def main():
    transform=transforms.Compose([ # Это нужно, чтобы преобразовать изображение из базы в тензор
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('./data', train=True, download=True, transform=transform) # Здесь загружаем MNIST датасет для тренировки
    dataset2 = datasets.MNIST('./data', train=False, transform=transform) # Здесь загружаем MNIST датасет для теста
    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=BATCH_SIZE) # Этот класс поможет в обучении (разобьёт на батчи и т.п.)
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size=BATCH_SIZE) # То же для теста
    model = Net().to(DEVICE) # Создаём объект класса нашей сети
    #model = torch.load('./model.pth').to(DEVICE) # Можем загрузить предобученную сеть, если сохранили
    optimizer = optim.Adam(model.parameters(), lr=1e-4) # Этот делает градиентные шаги, здесь мог быть Adam или SGD
    test1(model, dataset2) # Проверим, что выдаёт необученная нейронная сеть на конкретном примере
    for epoch in range(0, N_EPOCHES): # Цикл по эпохам
        print('Epoch {}/{}'.format(epoch + 1, N_EPOCHES))
        train(model, train_loader, optimizer) # Тренируем одну эпоху
        test(model, test_loader) # Тестируем после эпохи
    test1(model, dataset2) # Проверим, что выдаёт обученная нейронная сеть на конкретном примере
    torch.save(model, './model.pth')
        
if __name__ == '__main__':
    main() # Так принято в питоне
