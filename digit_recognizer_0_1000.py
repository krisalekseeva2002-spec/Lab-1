"""Распознавание рукописных чисел от 0 до 1000 с помощью CNN на PyTorch.

Нейросеть обучается классифицировать ОТДЕЛЬНЫЕ цифры от 0 до 9 (MNIST).
Перед распознаванием числа программа выделяет его цифры по промежуткам
между символами, распознаёт каждую и объединяет результат в целое число.

ВАЖНО: цифры должны быть написаны раздельно, не касаясь друг друга.
Для распознавания слитного рукописного текста нужна более сложная модель.

Установка: pip install torch torchvision pillow numpy

Примеры запуска:
    python digit_recognizer_0_1000.py train --epochs 3
    python digit_recognizer_0_1000.py draw
    python digit_recognizer_0_1000.py predict number.png
"""

from __future__ import annotations

# Библиотеки для обработки аргументов командной строки и путей к файлам.
import argparse
from pathlib import Path

# NumPy обрабатывает изображения как массивы чисел.
# PyTorch нужен для создания, обучения и использования нейронной сети.
# Pillow используется для чтения изображений и рисования цифр.
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn


# Папка проекта и стандартный путь к файлу с обученными весами нейросети.
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = BASE_DIR / "digit_model.pth"


# 1. АРХИТЕКТУРА НЕЙРОННОЙ СЕТИ ДЛЯ ОДНОЙ ЦИФРЫ
# CNN (Convolutional Neural Network) — свёрточная нейросеть.
# На вход она получает чёрно-белое изображение размером 28x28 пикселей.
class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # СВЁРТОЧНАЯ ЧАСТЬ: ищет линии, края и элементы формы цифры.
        # 1 -> 32: один входной канал (оттенки серого), 32 карты признаков.
        # ReLU добавляет нелинейность, MaxPool2d уменьшает размер изображения вдвое.
        # Размеры данных: 1x28x28 -> 32x14x14 -> 64x7x7.
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # КЛАССИФИКАТОР: по найденным признакам определяет цифру.
        # Flatten превращает карты признаков в одномерный массив из 3136 чисел.
        # Linear — полносвязный слой; Dropout снижает риск переобучения.
        # Последний слой возвращает 10 оценок — по одной для цифр 0-9.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 10),
        )

    # Определяем, как изображение проходит через нейросеть.
    def forward(self, x):
        return self.classifier(self.features(x))


# 2. ОБУЧЕНИЕ НЕЙРОСЕТИ
# epochs — количество проходов по всему обучающему набору.
# output_path — куда сохранить обученную модель.
def train(epochs: int, output_path: Path):
    if epochs < 1:
        raise ValueError("Число эпох должно быть не меньше 1.")

    # Импортируем инструменты загрузки MNIST только при запуске обучения.
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    # MNIST содержит 60 000 изображений для обучения и 10 000 для тестирования.
    # download=True скачивает набор данных при первом запуске.
    # ToTensor переводит изображения в тензоры со значениями пикселей от 0 до 1.
    dataset_path = BASE_DIR / "data"
    train_data = datasets.MNIST(
        dataset_path, train=True, download=True,
        transform=transforms.ToTensor()
    )
    test_data = datasets.MNIST(
        dataset_path, train=False, download=True,
        transform=transforms.ToTensor()
    )

    # DataLoader выдаёт изображения небольшими группами (batch).
    # shuffle=True перемешивает обучающие примеры, чтобы менять их порядок.
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=0)

    # Если доступна видеокарта NVIDIA с CUDA, обучаем на ней; иначе — на CPU.
    # Adam изменяет веса нейросети; CrossEntropyLoss измеряет ошибку классификации.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DigitCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Основной цикл обучения: сеть предсказывает цифры и исправляет свои ошибки.
    print(f"Устройство: {device}. Обучение на {len(train_data)} изображениях.")
    for epoch in range(1, epochs + 1):
        model.train()  # Режим обучения: Dropout включён.
        total_loss = 0.0

        for images, labels in train_loader:
            # Переносим изображения и правильные ответы на выбранное устройство.
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()       # Сбрасываем градиенты предыдущего шага.
            logits = model(images)      # Получаем оценки 10 возможных цифр.
            loss = criterion(logits, labels)  # Считаем ошибку предсказаний.
            loss.backward()             # Вычисляем градиенты (backpropagation).
            optimizer.step()            # Корректируем веса сети для уменьшения ошибки.

            total_loss += loss.item() * images.size(0)

        # Средняя ошибка за эпоху: чем ниже, тем лучше на обучающем наборе.
        print(f"Эпоха {epoch}/{epochs}: loss = {total_loss / len(train_data):.4f}")

    # 3. ПРОВЕРКА ТОЧНОСТИ НА НЕЗНАКОМЫХ ИЗОБРАЖЕНИЯХ
    model.eval()  # Режим оценки: Dropout отключён.
    correct = 0

    # Для тестирования вычислять градиенты не нужно.
    with torch.inference_mode():
        for images, labels in test_loader:
            logits = model(images.to(device))
            # argmax выбирает цифру с самой высокой оценкой.
            # Сравниваем предсказания с правильными ответами.
            correct += (logits.argmax(dim=1).cpu() == labels).sum().item()

    accuracy = 100 * correct / len(test_data)
    print(f"Точность на тестовой части MNIST: {accuracy:.2f}%")

    # Сохраняем обученные веса в файл .pth, чтобы не обучать модель заново.
    # Переносим веса на CPU, чтобы файл можно было загрузить без видеокарты.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.cpu().state_dict(), output_path)
    print(f"Модель сохранена: {output_path}")


# 4. ПОДГОТОВКА ОДНОЙ ЦИФРЫ ДЛЯ РАСПОЗНАВАНИЯ
# Приводим отдельную цифру к формату MNIST: 28x28, белая на чёрном.
def prepare_image(image: Image.Image, *, already_normalized: bool = False) -> torch.Tensor:
    # Переводим изображение в оттенки серого (L), затем в массив пикселей.
    grayscale = np.asarray(image.convert("L"), dtype=np.uint8).copy()

    # Если фон светлый, инвертируем цвета: делаем фон чёрным, а цифру белой.
    # Значения ниже 30 считаем фоновым шумом и обнуляем.
    # После segment_digits фон уже чёрный, поэтому инверсия не нужна.
    # Это важно, например, для узкой цифры «1» с плотным белым штрихом.
    if not already_normalized and np.median(grayscale) > 127:
        grayscale = 255 - grayscale
    grayscale[grayscale < 30] = 0

    # Находим координаты всех видимых пикселей цифры.
    # Если таких пикселей нет, значит изображение пустое.
    coords = np.argwhere(grayscale > 30)
    if len(coords) == 0:
        raise ValueError("На изображении не найдена цифра.")

    # Обрезаем лишний фон вокруг цифры по её крайним пикселям.
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    digit = Image.fromarray(grayscale[y0:y1, x0:x1])

    # Масштабируем цифру так, чтобы её максимальная сторона стала 20 пикселей.
    # Пропорции сохраняются; LANCZOS сглаживает изображение при изменении размера.
    width, height = digit.size
    scale = 20 / max(width, height)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    digit = digit.resize(new_size, Image.Resampling.LANCZOS)

    # Размещаем цифру в центре чёрного квадрата размером 28x28.
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(digit, ((28 - digit.width) // 2, (28 - digit.height) // 2))

    # Нормализуем пиксели до диапазона 0-1.
    # Добавляем две оси: размер партии и количество каналов.
    # Результат имеет форму [1, 1, 28, 28], которую ожидает CNN.
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0).unsqueeze(0)


# 5. РАЗДЕЛЕНИЕ МНОГОЗНАЧНОГО ЧИСЛА НА ОТДЕЛЬНЫЕ ЦИФРЫ
# Например, изображение «205» разбивается на три изображения: «2», «0», «5».
# Это делается без отдельного обучения: анализируем пустые вертикальные колонки.
def segment_digits(image: Image.Image) -> list[Image.Image]:
    # Переводим рисунок в чёрно-белый массив с числами от 0 до 255.
    gray = np.asarray(image.convert("L"), dtype=np.uint8).copy()

    # MNIST ожидает белые штрихи на чёрном фоне.
    # Если фон светлый (белый лист с тёмными цифрами), инвертируем цвета.
    if np.median(gray) > 127:
        gray = 255 - gray
    gray[gray < 30] = 0  # Убираем очень слабые пиксели фона.

    # По оси X находим столбцы, в которых присутствуют нарисованные штрихи.
    active_x = np.flatnonzero(np.any(gray > 0, axis=0))
    if active_x.size == 0:
        raise ValueError("Изображение пустое: нарисуйте число от 0 до 1000.")

    # Разделителем между цифрами служит достаточно широкий пустой промежуток.
    # Для большого холста это несколько пикселей; для маленького — минимум два.
    min_gap = max(2, round(gray.shape[0] * 0.02))
    groups = []
    start = last = int(active_x[0])

    for x in active_x[1:]:
        x = int(x)
        # Если пустых колонок достаточно, предыдущая цифра закончилась.
        if x - last - 1 >= min_gap:
            groups.append((start, last + 1))
            start = x
        last = x
    groups.append((start, last + 1))

    # Число до 1000 может состоять максимум из четырёх цифр.
    if len(groups) > 4:
        raise ValueError(
            f"Обнаружено {len(groups)} отдельных частей. "
            "Для чисел 0–1000 нужно не более четырёх раздельных цифр."
        )

    digit_images = []
    for x0, x1 in groups:
        # Находим верхнюю и нижнюю границу каждой цифры.
        stripe = gray[:, x0:x1]
        active_y = np.flatnonzero(np.any(stripe > 0, axis=1))
        y0, y1 = int(active_y[0]), int(active_y[-1]) + 1

        # Вырезаем цифру: далее prepare_image масштабирует её до 28x28.
        digit_images.append(Image.fromarray(gray[y0:y1, x0:x1]))

    return digit_images


# 6. ЗАГРУЗКА РАНЕЕ ОБУЧЕННОЙ МОДЕЛИ
def load_model(model_path: Path) -> DigitCNN:
    # Без файла весов сеть не сможет распознавать цифры после перезапуска.
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Нет обученной модели: {model_path}\n"
            "Сначала запустите: python digit_recognizer_0_1000.py train"
        )

    # Создаём такую же архитектуру и загружаем сохранённые веса.
    # map_location='cpu' позволяет использовать модель на обычном компьютере.
    model = DigitCNN()
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    return model


# 7. РАСПОЗНАВАНИЕ ОДНОЙ ЦИФРЫ
# Возвращаем три наиболее вероятных варианта: (цифра, вероятность).
def recognize(model: DigitCNN, image: Image.Image, *, already_normalized=False):
    tensor = prepare_image(image, already_normalized=already_normalized)

    # softmax переводит 10 выходных оценок сети в вероятности.
    # inference_mode ускоряет предсказание, поскольку обучение здесь не проводится.
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0]

    # topk(3) выбирает три цифры с наибольшими вероятностями.
    values, labels = probabilities.topk(3)
    return [(int(label), float(value)) for label, value in zip(labels, values)]


# 8. РАСПОЗНАВАНИЕ ЦЕЛОГО ЧИСЛА ОТ 0 ДО 1000
# Повторно используем CNN для каждой части рисунка и объединяем её ответы.
def recognize_number(model: DigitCNN, image: Image.Image):
    digit_images = segment_digits(image)

    # Для каждой цифры получаем наиболее вероятный класс из диапазона 0–9.
    predicted_digits = []
    confidences = []
    for digit_image in digit_images:
        digit, confidence = recognize(model, digit_image, already_normalized=True)[0]
        predicted_digits.append(digit)
        confidences.append(confidence)

    # Например: [2, 0, 5] -> строка "205" -> целое число 205.
    number = int("".join(str(digit) for digit in predicted_digits))
    if number > 1000:
        raise ValueError(
            f"Получено число {number}, оно превышает 1000. "
            "Нарисуйте число в диапазоне 0–1000."
        )

    # Вероятности относятся к каждой цифре отдельно, а не ко всему числу.
    return number, list(zip(predicted_digits, confidences))


# 9. ГРАФИЧЕСКОЕ ОКНО ДЛЯ РИСОВАНИЯ ЧИСЛА МЫШЬЮ
def launch_draw_window(model: DigitCNN):
    # tkinter — стандартная библиотека Python для оконных приложений.
    import tkinter as tk
    from tkinter import messagebox

    # Создаём окно с заголовком и фиксированным размером.
    root = tk.Tk()
    root.title("Распознавание чисел 0–1000")
    root.resizable(False, False)

    # Два холста: видимый tkinter и невидимый Pillow (для нейросети).
    # Широкий холст позволяет рисовать от одной до четырёх цифр слева направо.
    width, height = 600, 280
    brush = 20
    drawing = Image.new("L", (width, height), 0)
    painter = ImageDraw.Draw(drawing)
    canvas = tk.Canvas(root, width=width, height=height, bg="black", highlightthickness=0)
    canvas.pack(padx=12, pady=(12, 8))
    previous = [None]  # Предыдущая точка движения мыши.
    result_text = tk.StringVar(value="Нарисуйте число 0–1000, оставляя промежутки между цифрами")

    # Рисуем круглую точку при нажатии мыши или во время движения.
    def make_dot(x, y):
        radius = brush / 2
        canvas.create_oval(x-radius, y-radius, x+radius, y+radius,
                           fill="white", outline="white")
        painter.ellipse((x-radius, y-radius, x+radius, y+radius), fill=255)

    # Событие: пользователь нажал левую кнопку мыши.
    def pen_down(event):
        previous[0] = (event.x, event.y)
        make_dot(event.x, event.y)

    # Событие: пользователь ведёт мышью с зажатой левой кнопкой.
    # Соединяем предыдущую и новую точки линией, чтобы штрих был непрерывным.
    def pen_move(event):
        new_point = (event.x, event.y)
        if previous[0] is not None:
            x0, y0 = previous[0]
            canvas.create_line(x0, y0, event.x, event.y,
                               fill="white", width=brush, capstyle=tk.ROUND)
            painter.line((x0, y0, event.x, event.y), fill=255, width=brush)
            make_dot(event.x, event.y)
        previous[0] = new_point

    # Событие: пользователь отпустил кнопку мыши.
    def pen_up(_event):
        previous[0] = None

    # Кнопка «Очистить»: удаляем рисунок и предыдущий результат распознавания.
    def clear():
        canvas.delete("all")
        painter.rectangle((0, 0, width, height), fill=0)
        previous[0] = None
        result_text.set("Нарисуйте число 0–1000, оставляя промежутки между цифрами")

    # Кнопка «Распознать»: делим рисунок на цифры и собираем число.
    def predict():
        try:
            number, details = recognize_number(model, drawing)
        except ValueError as exc:
            messagebox.showwarning("Не удалось распознать число", str(exc))
            return

        # Показываем результат и вероятности для ОТДЕЛЬНЫХ цифр.
        scores = ", ".join(f"{digit}: {score:.1%}" for digit, score in details)
        result_text.set(f"Число: {number}\nЦифры и их вероятности: {scores}")

    # Связываем действия мыши с функциями рисования.
    canvas.bind("<Button-1>", pen_down)
    canvas.bind("<B1-Motion>", pen_move)
    canvas.bind("<ButtonRelease-1>", pen_up)

    # Размещаем кнопки и строку с результатом в окне.
    toolbar = tk.Frame(root)
    toolbar.pack(pady=4)
    tk.Button(toolbar, text="Распознать", command=predict, width=16).pack(side="left", padx=5)
    tk.Button(toolbar, text="Очистить", command=clear, width=12).pack(side="left", padx=5)
    tk.Label(root, textvariable=result_text, justify="center", height=3).pack(padx=12, pady=(2, 12))

    # Запускаем цикл обработки событий окна; программа работает, пока оно открыто.
    root.mainloop()


# 10. ТОЧКА ВХОДА И КОМАНДЫ ЗАПУСКА
def main():
    # argparse позволяет запускать один файл в трёх режимах:
    # train — обучить модель, predict — распознать файл, draw — открыть окно.
    parser = argparse.ArgumentParser(description="Распознавание рукописных чисел 0–1000")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                        help="Путь к файлу модели (.pth)")
    commands = parser.add_subparsers(dest="command", required=True)

    training = commands.add_parser("train", help="Скачать MNIST и обучить модель")
    training.add_argument("--epochs", type=int, default=3)

    prediction = commands.add_parser("predict", help="Распознать число 0–1000 в файле")
    prediction.add_argument("image", type=Path, help="Путь к PNG/JPG")

    commands.add_parser("draw", help="Открыть окно рисования")
    args = parser.parse_args()

    # При команде train обучаем сеть и сохраняем её веса.
    if args.command == "train":
        train(args.epochs, args.model)
        return

    # Для распознавания сначала загружаем уже обученную сеть.
    model = load_model(args.model)
    if args.command == "predict":
        # Выделяем цифры на изображении, распознаём их и собираем число.
        with Image.open(args.image) as image:
            number, details = recognize_number(model, image)
        print(f"Распознанное число: {number}")
        for position, (digit, probability) in enumerate(details, 1):
            print(f"Цифра {position}: {digit} (вероятность {probability:.2%})")
    elif args.command == "draw":
        # Открываем окно, где пользователь может рисовать цифры мышью.
        launch_draw_window(model)


# Запускаем main() только при непосредственном запуске этого Python-файла.
if __name__ == "__main__":
    main()
