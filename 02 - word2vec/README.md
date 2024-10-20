# Word2vec и классификация текста

Для запуска потребуются **torch**, **datasets**, **nltk**, **pymorphy3**, **tqdm**.
Обычно лучше создать **venv** окружение: https://code.visualstudio.com/docs/python/environments.
Запускаем **main** файл. В папке создастся **params.json** файл с настройками по умолчаню.
Все обработанные документы и обученные сети сохраняются в папку **/data**


## Порядок работы программы

- Загружается датасет https://huggingface.co/datasets/Maxstan/russian_youtube_comments_political_and_nonpolitical.
- Происходит обработка всех документов (буквы приводятся к нижнему регистру, убираются знаки препинания, стопслова и цифры, происходит токенизация и лемматизация).
- Множество документов разбивается на тренировочное, валидационное и тестовое множества. Сейчас  разбиваетмя поровну по классам, для неравномерного не настроены веса обучения.
- Создаётся словарь из слов.
- Создаётся датасет для обучения word2vec: проходим скользящим окном с фиксированным размером (по умолчанию 2) и добавляем слова в контекст, если они есть в словаре.
- Обучаем word2vec.
- Создаём эмбеддинги документов.
- Обучаем классификатор на тренировочном множестве. На валидационном смотрим качество, если не улучшалось на протяжении 5 эпох - останавливаем обучение.
- Запускаем на тестовом, там сейчас просто 200 случайных документов.
- Запускаем inference на кастомных комментариях. Они находятся в файле **custom_political.txt**. Каждая строка интерпретируется как отдельный документ. Для каждого предсказывает политическое либо фоновое.
