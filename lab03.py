import torch
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

DATASET_PATH = 'PetImages/'
CAT_PATH = os.path.join(DATASET_PATH, 'Cat/')
DOG_PATH = os.path.join(DATASET_PATH, 'Dog/')
MAX_IMAGES = 10000
RESIZE_W = 128
RESIZE_H = 128
TRAIN_COUNT = MAX_IMAGES * 80 // 100
BATCH_SIZE = 10
LEANING_RATE = 0.001

# Считываем пути изображений в папке folder_path
def readImageNames(folder_path):
    cur_data = []
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if os.path.isfile(filepath) and filename.endswith('.jpg'):
            cur_data.append(filepath)
            #print(filename)
    return cur_data

# Считываем изображение opencv из img_path
def readImage(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if img.shape[0] < RESIZE_H or img.shape[1] < RESIZE_W:
            img = cv2.resize(img, (RESIZE_H, RESIZE_W), interpolation=cv2.INTER_LINEAR)
        else:
            img = cv2.resize(img, (RESIZE_H, RESIZE_W), interpolation=cv2.INTER_AREA)

        img = img.astype(np.float32) / 255.
        return img
    return None

# Считываем пути до котов, получаем список строк
cats_data = readImageNames(CAT_PATH)
dogs_data = readImageNames(DOG_PATH)

# Считываем все изображения из списка с путями
def readImages(data):
    images = []
    k = 0
    for i in data:
        img = readImage(i)
        if img is not None:
            images.append(img)
            k += 1
        if k + 1 == MAX_IMAGES:
            break
    return images

# Проверили один файл и отобразили его
img1 = readImage(cats_data[38])
plt.imshow(img1)
plt.show()

# Cчитывам все изображения opencv
cat_images = readImages(cats_data)
dog_images = readImages(dogs_data)

# Класс датасета, в него передаём считанные изображения
    # можно было считать изобраения прям в этом классе.
    # Обычно если размер датесета превышает размер оперативной памяти,
    #  то изображения считывает по ходу.
class CatsAndDogsDataset(torch.utils.data.Dataset):
    def __init__(self, cats_list, dogs_list):
        # Вызываем конструктор родительского класса
        super().__init__()
        # Создаём поле data_list - список в который будем складываеть
        #   пары (тензоры исходных изображений, номер класса)
        self.data_list = []
        for i in cats_list:
            # добавляем в data_list котов
            t = torch.from_numpy(i).permute(2, 0, 1)
            self.data_list.append((t, 0,))
        for i in dogs_list:
            # добавляем в data_list собак
            t = torch.from_numpy(i).permute(2, 0, 1)
            self.data_list.append((t, 1,))

    # Питоновская функция, нужна torch
    def __len__(self):
        # Возвращаем размер нашего датасета
        return len(self.data_list)

    # Получение элемента по индексу
    def __getitem__(self, index):
        return self.data_list[index]

# Питоновские слайсы [начало(с 0):конец(не включительно):шаг]
train_cat_images = cat_images[:TRAIN_COUNT]
test_cat_images = cat_images[TRAIN_COUNT:]
train_dog_images = dog_images[:TRAIN_COUNT]
test_dog_images = dog_images[TRAIN_COUNT:]
# Создаём датесеты train и test, потом предадим их в dataloader
train_dataset = CatsAndDogsDataset(train_cat_images, train_dog_images)
test_dataset = CatsAndDogsDataset(test_cat_images, test_dog_images)

# Класс нашей нейронной сети
class MyNet(torch.nn.Module):
    # Пишем конструктор,
    #   num_classes - количество классов для классификации 
    #   num_chanels=3, если rgb; num_chanels=1, если градации серого
    def __init__(self, num_classes : int = 2, num_chanels : int = 3) -> None:
        # Вызываем консруктор torch.nn.Module
        super().__init__()
        # Создаём поле self.conv1 инициализируем 
        #   объектом класса свёртки torch.nn.Conv2d, 
        #   num_chanels - кол-во входных каналов,
        #   num_out_channels - кол-во выходных каналов,
        #   kernel_size - размер ядра,
        #   stride - шаг окна свёрктки,
        #   padding - заполнения нулями по краям
        #       был (3x128х128), стал (64x30х30)
        #       30х30 - посчитали по формуле,
        #   num_out_channels, kernel_size, stride, padding - берём произвольно
        num_out_channels = 64 # так захотели
        self.conv1 = torch.nn.Conv2d(num_chanels, num_out_channels, 
                                     kernel_size=11, stride=4, padding=2)
        # Создаём поле self.act1 - активация, инициализируем 
        #   объектом класса torch.nn.ReLU, 
        #   inplace=True - не знаю что значит
        self.act1 = torch.nn.ReLU(inplace=True)
        # Создаём полносвязный слой
        #   первым параметром размер тензора на входе
        #   вторым параметром размер тензора на выходе
        #   т.к. сейчас делаем один слой, то выход=num_classes
        #   на последнем слое всегда выход=num_classes
        self.fc1 = torch.nn.Linear(31*31*64, num_classes)
        # softmax будет потом отдельно
    
   # Пишем прямой проход,
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        # применяем свёрточный слой ко входному тензору
        x = self.conv1(x)
        # применяем слой активации
        x = self.act1(x)
        # преобразуем в одномерный тензор
        x = torch.flatten(x, 1)
        # применяем выходной полносвязный слой
        x = self.fc1(x)
        # возвращаем итоговый тензор
        return x

# Cоздаём dataloder
train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=BATCH_SIZE,
                                               shuffle=False)
# Cоздаём объект сети
model = MyNet()
# Cоздаём loss функцию
criterion = torch.nn.CrossEntropyLoss()
# Cоздаём оптимизатор, для градиентных шагов, который будет обучать нашу сеть
optimizer = torch.optim.Adam(model.parameters(), lr=LEANING_RATE)

# Функция для вычисления loss на тестовом множестве
def eval_test_metrics(dataloader):
    model.eval() # Переводим сеть в eval, говорим что мы не будем обучать
    correct, total, test_loss = 0, 0, 0.0
    y_true, y_pred_classes = [], []
    print("eval metrics...")
    with torch.no_grad(): # Выключаем вычисления градиентов
        for batch in dataloader:
            inputs, labels = batch
            # Выполняем прямой проход
            outputs = model(inputs)
            # Аккумулируем loss
            test_loss += criterion(outputs, labels).item()

            outputs = torch.nn.functional.softmax(outputs, dim=1)
            answer = (outputs.argmax(1) == labels)
            correct += answer.sum().item()
            total += len(labels)

    avg_test_loss = test_loss / len(dataloader)
    accuracy = correct / total
    return avg_test_loss, accuracy

test_loss, test_accuracy = eval_test_metrics(test_dataloader)
print("accuracy=", test_accuracy)

# Функция обучения нейронной сети
def train_net():
    best_test_acc = 0.
    for epoch in range(100):
        model.train()
        correct, total, train_loss = 0, 0, 0.
        for batch in train_dataloader:
            inputs, labels = batch
            # Обнуляем градиенты
            optimizer.zero_grad()
            # Выполняем прямой проход
            outputs = model(inputs)
            # Вычисляем loss
            loss = criterion(outputs, labels)
            # Вычисляем градиенты
            loss.backward()
            # Делаем градиентный шаг
            optimizer.step()
            # Аккумулируем loss
            train_loss += loss.detach().item()
            
            outputs = torch.nn.functional.softmax(outputs, dim=1)
            answer = (outputs.argmax(1) == labels)
            correct += answer.sum().item()
            total += len(labels)

        train_loss /= len(train_dataloader)
        accuracy = correct / total
        test_loss, test_accuracy = eval_test_metrics(test_dataloader)
        print(f"epoch={epoch}, train_loss={train_loss}, train_accuracy={accuracy}\
        test_loss={test_loss}, test_accuracy={test_accuracy}")

train_net()