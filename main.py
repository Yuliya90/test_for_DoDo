"""
Прототип системы детекции уборки столиков по видео.

Скрипт обрабатывает видеофайл кадр за кадром:
  - определяет зону (ROI) — область столика на видео
  - с помощью нейросети YOLOv8 обнаруживает людей на каждом кадре
  - отслеживает, находится ли человек в зоне столика
  - фиксирует события: столик занят, пуст, к нему подошли
  - считает статистику времени между уходом гостя и приходом следующего
  - сохраняет итоговое видео и текстовый отчёт

Запуск: python main.py --video "видео 1.mp4" (замени на свое видео)
"""

import argparse
import time
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


# Константы — управляют поведением детекции

# Сколько кадров подряд должно наблюдаться, прежде чем оно считается «подтверждённым».
# Защищает от мерцания: если человек на 1-2 кадра вышел из зоны, состояние не изменится.
DEBOUNCE_FRAMES = 15

# Индекс класса «person» в датасете COCO, на котором обучена модель YOLOv8.
PERSON_CLASS_ID = 0

# Минимальная уверенность (confidence) модели, при которой детекция считается достоверной. Значение 0.4 = 40%.
# Слишком низкое → много ложных срабатываний; слишком высокое → пропуски.
YOLO_CONF = 0.4

# min доля площади ROI, которую должен перекрывать bounding box человека, чтобы считать его «у столика».
# 0.15 = достаточно 15% площади зоны — человек рядом, но не обязательно сидит.
IOU_THRESHOLD = 0.15

# Цвета рамки столика на видео (в формате BGR, не RGB — особенность OpenCV)
COLOR_EMPTY    = (0, 255, 0)    # зелёный — столик свободен
COLOR_OCCUPIED = (0, 0, 255)    # красный  — у столика есть человек
COLOR_APPROACH = (0, 165, 255)  # оранжевый — первый подход после пустоты
COLOR_TEXT     = (255, 255, 255) # белый    — цвет подписей

# Сколько кадров показывать состояние APPROACH на видео после подхода
APPROACH_DISPLAY_FRAMES = 60



# Вспомогательные функции


def parse_args():
    """
    Обязательный аргумент:
      --video путь к видеофайлу

    Необязательные аргументы:
      --output куда сохранить видео
      --report куда сохранить текстовый отчёт
      --model какую модель YOLO использовать
      --roi координаты зоны столика X Y W H
    """
    parser = argparse.ArgumentParser(description="Детекция уборки столиков по видео")
    parser.add_argument("--video", required=True, help="Путь к входному видео")
    parser.add_argument("--output", default="output.mp4", help="Путь к выходному видео")
    parser.add_argument("--report", default="report.txt", help="Путь к текстовому отчёту")
    parser.add_argument("--model", default="yolov8n.pt", help="Путь к модели YOLO")
    parser.add_argument("--roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"),
                        help="Координаты ROI вручную")
    return parser.parse_args()


def select_roi(frame):
    """
    Открывает видео и позволяет пользователю выделить зону столика мышью.

    Управление:
      - Зажмите левую кнопку мыши и растяните прямоугольник
      - ENTER — подтвердить выбор
      - C — отменить выбор

    Возвращает кортеж (x, y, w, h) — координаты левого верхнего угла и размеры выделенной зоны в пикселях.
    """
    print("\n[ROI] Выделите зону столика мышью и нажмите ENTER.")
    print("[ROI] Для отмены нажмите C.\n")
    roi = cv2.selectROI("Выберите столик", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Выберите столик")

    # Если пользователь не выделил зону — прерываем работу
    if roi == (0, 0, 0, 0):
        raise ValueError("ROI не был выбран.")

    return roi  # (x, y, w, h)


def compute_iou_with_roi(box, roi):
    """
    Вычисляет, какую долю площади ROI перекрывает bounding box человека.

    Аргументы:
      box — координаты рамки человека: (x1, y1 — левый верхний угол, x2, y2 — правый нижний)
      roi — зона столика: (x, y — левый верхний угол, w, h — ширина и высота)

    Возвращает:
      Число от 0.0 до 1.0 — доля площади ROI, которую перекрывает человек.
      Например, 0.3 означает, что перекрывает 30% зоны столика.
    """
    # Переводим ROI из формата (x, y, w, h) в формат двух точек (x1, y1, x2, y2)
    rx, ry, rw, rh = roi
    rx2, ry2 = rx + rw, ry + rh

    bx1, by1, bx2, by2 = box

    # Находим координаты прямоугольника пересечения: берём максимум левых и минимум правых краёв
    ix1 = max(bx1, rx)
    iy1 = max(by1, ry)
    ix2 = min(bx2, rx2)
    iy2 = min(by2, ry2)

    # Ширина и высота пересечения (0 если прямоугольники не пересекаются)
    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    inter_area = inter_w * inter_h

    # Площадь зоны столика
    roi_area = rw * rh
    if roi_area == 0:
        return 0.0  # защита от деления на ноль

    return inter_area / roi_area  # доля пересечения относительно ROI


def draw_overlay(frame, roi, state, frame_idx, fps):
    """
    Рисует поверх кадра подсказки:
      - цветную рамку вокруг зоны столика (зелёная / красная / оранжевая)
      - подпись «EMPTY», «OCCUPIED» или «APPROACH» над рамкой
      - текущий таймкод видео в левом верхнем углу

    Аргументы:
      frame      — текущий кадр
      roi        — зона столика (x, y, w, h)
      state      — отображаемое состояние: "EMPTY", "OCCUPIED" или "APPROACH"
      frame_idx  — порядковый номер кадра
      fps        — частота кадров видео

    Возвращает изменённый кадр.
    """
    x, y, w, h = roi

    # Выбираем цвет и текст в зависимости от состояния столика
    # OpenCV не поддерживает кириллицу — используем латиницу
    if state == "EMPTY":
        color = COLOR_EMPTY
        label = "EMPTY"
    elif state == "APPROACH":
        color = COLOR_APPROACH
        label = "APPROACH"
    else:
        color = COLOR_OCCUPIED
        label = "OCCUPIED"

    # Рисуем рамку вокруг зоны столика
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)

    # Рисуем закрашенный фон для подписи над рамкой
    text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.rectangle(frame, (x, y - text_size[1] - 10), (x + text_size[0] + 6, y), color, -1)

    # Пишем подпись состояния
    cv2.putText(frame, label, (x + 3, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_TEXT, 2)

    # Вычисляем текущее время видео и выводим таймкод в формате ЧЧ:ММ:СС
    seconds = frame_idx / fps if fps > 0 else 0
    timecode = time.strftime("%H:%M:%S", time.gmtime(seconds))
    cv2.putText(frame, f"Time: {timecode}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)

    return frame


# Основной пайплайн


def main():
    args = parse_args()

    # Открываем видеофайл
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Не удалось открыть видео: {args.video}")

    # Считываем метаданные видео; fps нужен для перевода номера кадра во время
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0  # если метаданные отсутствуют — 25 fps
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Видео: {args.video}  |  {width}x{height}  |  {fps:.1f} fps  |  {total} кадров")

    #Выбор зоны столика (ROI)
    if args.roi:
        # Если координаты переданы через аргумент --roi, используем их напрямую
        roi = tuple(args.roi)
        print(f"[ROI] Используется заданный ROI: {roi}")
    else:
        # Иначе читаем первый кадр и даём пользователю выделить зону мышью
        ret, first_frame = cap.read()
        if not ret:
            raise RuntimeError("Не удалось прочитать первый кадр видео.")
        roi = select_roi(first_frame)
        # Возвращаем позицию чтения на начало видео, чтобы обработать все кадры
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print(f"[ROI] x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}")

    #Загружаем модель YOLOv8
    print(f"[MODEL] Загрузка модели: {args.model}")
    model = YOLO(args.model)

    # Настройка записи выходного видео
    # mp4v — для формата MP4; размер и fps берём из исходного видео
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    # Переменные машины состояний
    # Машина состояний нужна, чтобы не реагировать на каждый кадр по отдельности, а подтверждать смену состояния только после DEBOUNCE_FRAMES стабильных кадров.

    current_state   = "EMPTY"  # подтверждённое текущее состояние столика
    candidate_state = "EMPTY"  # «кандидат» — состояние, которое пока накапливает кадры
    debounce_count  = 0        # сколько кадров подряд наблюдается candidate_state

    # Счётчик кадров для отображения APPROACH на видео.
    # После подхода показываем надпись APPROACH в течение APPROACH_DISPLAY_FRAMES кадров,
    # затем переключаемся на OCCUPIED.
    approach_display_count = 0

    # Список для накопления событий; потом преобразуется в DataFrame
    events = []

    frame_idx = 0  # счётчик обработанных кадров

    print("[INFO] Обработка видео...")

    # Основной цикл — обработка кадра за кадром
    while True:
        ret, frame = cap.read()
        if not ret:
            # ret = False означает конец видео — выходим из цикла
            break

        frame_idx += 1
        # Переводим номер кадра в секунды для временных меток
        timestamp = frame_idx / fps

        # Детекция людей на текущем кадре
        # Запускаем YOLOv8; просим искать только класс «person» (classes=[0])
        # verbose=False отключаем вывод логов
        results = model(frame, classes=[PERSON_CLASS_ID], conf=YOLO_CONF, verbose=False)

        person_in_roi = False  # флаг: найден ли человек в зоне столика

        for result in results:
            for box in result.boxes:
                # Получаем координаты рамки человека в формате (x1, y1, x2, y2)
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # Считаем, какую долю ROI перекрывает эта рамка
                overlap = compute_iou_with_roi((x1, y1, x2, y2), roi)

                if overlap >= IOU_THRESHOLD:
                    # Человек достаточно близко к столику — значит «в зоне»
                    person_in_roi = True
                    # Рисуем оранжевую рамку вокруг обнаруженного человека
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
                    break  # достаточно одного человека в зоне — дальше не ищем

            if person_in_roi:
                break  # выходим из цикла

        # Машина состояний с debounce состояние на этомкадре ещё не подтверждено
        raw_state = "OCCUPIED" if person_in_roi else "EMPTY"

        if raw_state == candidate_state:
            # Состояние на этом кадре совпадает с кандидатом — увеличиваем счётчик
            debounce_count += 1
        else:
            # Состояние изменилось — начинаем отсчёт заново для нового кандидата
            candidate_state = raw_state
            debounce_count  = 1

        # Если кандидат держится уже DEBOUNCE_FRAMES кадров подряд и отличается от текущего подтверждённого состояния — подтверждаем смену
        if debounce_count >= DEBOUNCE_FRAMES and raw_state != current_state:
            prev_state    = current_state
            current_state = raw_state

            if prev_state == "EMPTY" and current_state == "OCCUPIED":
                # Переход EMPTY → OCCUPIED: записываем два события.
                # Сначала APPROACH — момент появления человека после пустоты,
                # затем OCCUPIED — подтверждение, что стол занят.
                events.append({
                    "frame":      frame_idx,
                    "timestamp":  round(timestamp, 3),
                    "event_type": "APPROACH",
                })
                events.append({
                    "frame":      frame_idx,
                    "timestamp":  round(timestamp, 3),
                    "event_type": "OCCUPIED",
                })
                print(f"  [EVENT] кадр={frame_idx:5d}  t={timestamp:7.2f}s  → APPROACH")
                print(f"  [EVENT] кадр={frame_idx:5d}  t={timestamp:7.2f}s  → OCCUPIED")
                # Запускаем таймер: показывать APPROACH на видео в течение APPROACH_DISPLAY_FRAMES кадров
                approach_display_count = APPROACH_DISPLAY_FRAMES
            else:
                # Переход OCCUPIED → EMPTY: записываем одно событие EMPTY
                events.append({
                    "frame":      frame_idx,
                    "timestamp":  round(timestamp, 3),
                    "event_type": current_state,
                })
                print(f"  [EVENT] кадр={frame_idx:5d}  t={timestamp:7.2f}s  → {current_state}")

        # Визуализация и запись кадра
        # Определяем, что показывать: APPROACH (пока таймер не истёк) или текущее состояние
        if approach_display_count > 0:
            display_state = "APPROACH"
            approach_display_count -= 1
        else:
            display_state = current_state

        frame = draw_overlay(frame, roi, display_state, frame_idx, fps)
        # Записываем данные в выходной видеофайл
        out.write(frame)

        # выводим прогресс
        if frame_idx % 500 == 0:
            print(f"  [PROGRESS] {frame_idx}/{total} кадров обработано...")

    # Освобождаем ресурсы видео
    cap.release()
    out.release()
    print(f"\n[INFO] Готово. Видео сохранено: {args.output}")


    # Аналитика событий


    # Собираем все события в DataFrame для анализа
    df = pd.DataFrame(events, columns=["frame", "timestamp", "event_type"])
    print("\n[ANALYTICS] Таблица событий:")
    print(df.to_string(index=False))

    #Вычисляем задержки между уходом гостя и подходом следующего
    delays = []

    # Выбираем временные метки всех событий «столик опустел» (EMPTY)
    empty_times    = df[df["event_type"] == "EMPTY"]["timestamp"].tolist()
    # Выбираем временные метки всех событий «к столику подошли» (APPROACH)
    approach_times = df[df["event_type"] == "APPROACH"]["timestamp"].tolist()

    for empty_t in empty_times:
        # Для каждого момента опустения находим самый ранний APPROACH после него
        future_approaches = [t for t in approach_times if t > empty_t]
        if future_approaches:
            # Задержка = время подхода − время ухода гостя
            delays.append(future_approaches[0] - empty_t)

    # Итоговая статистика
    if delays:
        avg_delay = np.mean(delays)
        min_delay = np.min(delays)
        max_delay = np.max(delays)
        print(f"\n[RESULT] Среднее время между уходом гостя и подходом следующего: {avg_delay:.2f} сек")
        print(f"[RESULT] Мин: {min_delay:.2f} сек  |  Макс: {max_delay:.2f} сек  |  Случаев: {len(delays)}")
    else:
        avg_delay = None
        print("\n[RESULT] Недостаточно данных для подсчёта статистики.")


    # Сохранение текстового отчёта

    with open(args.report, "w", encoding="utf-8") as f:
        f.write("=" * 50 + "\n")
        f.write("ОТЧЁТ: Детекция уборки столиков\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Видео:        {args.video}\n")
        f.write(f"ROI (столик): x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}\n")
        f.write(f"FPS:          {fps:.1f}\n")
        f.write(f"Кадров всего: {total}\n\n")
        f.write("Таблица событий:\n")
        f.write(df.to_string(index=False))
        f.write("\n\n")
        if avg_delay is not None:
            f.write(f"Среднее время задержки: {avg_delay:.2f} сек\n")
            f.write(f"Минимальное:            {min_delay:.2f} сек\n")
            f.write(f"Максимальное:           {max_delay:.2f} сек\n")
            f.write(f"Всего случаев:          {len(delays)}\n")
        else:
            f.write("Недостаточно данных для статистики.\n")

    print(f"[INFO] Отчёт сохранён: {args.report}")


if __name__ == "__main__":
    main()
