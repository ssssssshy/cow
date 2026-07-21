Архитектура системы строится на **модульном многоэтапном пайплайне (Modular Ensemble Pipeline)**. В качестве ключевого источника данных для анализа походки и хромоты используется датасет **CattleEye** (видеопоток сверху над прогоном), дополненный картами глубины и боковыми RGB-камерами.

---

---

### Архитектурный обзор пайплайна

```
[ Камеры (RGB + Depth) ]
         │
         ▼
[ Этап 1: Фильтрация & 3-Channel Depth ] (Deduplication, Blur, Distance Cut)
         │
         ▼
[ Этап 2: Smart Annotation ] (Grounding DINO + SAM 2 -> Qwen2.5-VL -> Cleanlab)
         │
         ▼
[ Stage 1: Detector & Crop ] (YOLOv11/YOLOE-26 + Focal Bi-Tempered Loss)
         │
         ├───> [ Crop Вымени/Тела ] ───> [ Stage 2: Classifiers ] (ResNet/EfficientNet)
         └───> [ Позвоночник/Ноги ] ───> [ Stage 2: Pose Estimation ] (YOLO-Pose)
                                                   │
                                                   ▼
[ Этап 5: Tracking & Window ] <───────── [ ByteTrack / BoT-SORT ]
         │
         ▼
[ Ансамбль-Агрегатор (XGBoost) ] ───> [ Ветеринарный Alert / Dashboard ]

```

---

### Этап 1: Подготовка данных и фильтрация (CattleEye + Depth)

Чтобы избавиться от дубликатов, шума и утечек данных (Data Leakage):

1. **Дедупликация (Deduplication):**
* **Perceptual Hashing (`imagehash` / pHash):** Быстрая первичная отбраковка статичных кадров, где корова стоит без движения.
* **CLIP-эмбеддинги (`OpenCLIP` / ViT-B/32):** Вычисление косинусного расстояния между векторами соседних кадров ($cos(\theta) > 0.98$) для удаления семантических дублей.


2. **Фильтрация брака (Blur & Occlusions):**
* **Motion Blur:** Дисперсия оператора Лапласа (`Laplacian Variance < 100.0` в OpenCV) удаляет размытые в движении кадры.
* **Перекрытие и пыль:** Анализ гистограммы яркости и контраста (выявление «слепых» кадров из-за брызг или грязи на линзе).


3. **Специфика карт глубины (по методологии Depth Filtering):**
* **Distance Limiting:** Отсечение фона по жесткому порогу расстояния (зануление всех пикселей карты глубины дальше 2.5–3.0 метров от камеры).
* **Генерация 3-канального входа (3-Channel Input):**
* *Канал 1:* Нормализованная карта глубины (Depth).
* *Канал 2:* Бинаризованный силуэт (Binarization) — четкий контур коровьего тела.
* *Канал 3:* Первая производная / Градиент (First Derivative) — подсвечивает выпирающие кости (маклаки, позвоночник) для оценки упитанности (BCS).


* **Sub-sampling аугментация:** Шаговое уменьшение кадра в 2 раза через пиксель для сэмплирования 4 под-кадров из одного (увеличение объема данных без потери деталей).


4. **Разделение датасета (Cow ID GroupKFold Split):**
* Строгая группировка по **ID коровы** (`GroupKFold` / `GroupShuffleSplit` из `scikit-learn`). Ни одна корова не должна попадать одновременно в Train и Val/Test.
* **Test Set:** Фиксированные 10–15% данных, где 100% разметки верифицированы экспертами-ветеринарами вручную.



---

### Этап 2: Умная разметка (Human-in-the-Loop)

Для сокращения затрат на ручную разметку на 90%:

1. **Автоматический вырез контуров (BBox & Masks):**
* **Grounding DINO:** Текстовый промпт (`"cow udder"`, `"cow spine"`, `"cow"`) локализует объекты и генерирует Bounding Boxes.
* **SAM 2 (Segment Anything Model 2):** Принимает BBox от DINO как промпт и строит точные попиксельные маски вымени и копыт.


2. **Автоматическая классификация аномалий (VLM Prompting):**
* **Qwen2.5-VL / Gemini 1.5 Pro:** Анализ кадров с системным промптом, строго обязывающим возвращать JSON-структуру:
```json
{
  "udder_asymmetry": true,
  "swelling_detected": false,
  "confidence": 0.72,
  "needs_review": true
}

```


* Если `confidence < 0.85` или флаг `needs_review == true`, кадр отправляется оператору.


3. **Платформа проверки (CVAT / Label Studio):**
* Оператор в интерфейсе правит лишь 5–10% спорных меток, сгенерированных AI.


4. **Аудит разметки (Cleanlab + FiftyOne):**
* **Cleanlab:** Поиск скрытых ошибок (Label Errors) с помощью анализа Out-of-Fold предсказаний.
* **FiftyOne:** Визуальный поиск аномалий в пространстве эмбеддингов, отбор нетипичных поз и выбросов.



---

### Этап 3: Обучение Stage 1 (Детекция и Кропы)

Задача первого этапа — быстро найти животное или его часть и вырезать фрагмент высокого разрешения.

* **Архитектура:** **YOLOv11x** или **YOLOE-26** (для детектирования редких патологий).
* **Backbone Freezing:** Заморозка первых 15 эпох (`freeze=10..20`), чтобы предобученные на COCO/ImageNet веса не разрушились о грязные начальные метки.
* **Gradient Clipping:** Защита от градиентного взрыва (`max_norm=10.0`).
* **Оптимизатор:** `AdamW` ($lr=1e-3$, $weight\_decay=1e-2$) с разогревом (`warmup_epochs=3.0`) и косинусным затуханием (`CosineAnnealingLR`).
* **Custom Loss:** **Focal Bi-Tempered Loss**:

$$\mathcal{L}_{BiTempered} = -\log_{t_1} \hat{p} + \text{Focal Term} (\gamma = 1.5–2.0)$$


* $t_1 = 0.7–0.8$: делает лосс ограниченным сверху, снижая влияние грубых ошибочных меток VLM.
* $t_2 = 1.2$: отвечает за тяжелые хвосты распределения вероятностей.
* $\gamma = 1.8$: фокусирует градиенты на сложных редких классах (больные коровы).



---

### Этап 4: Обучение Stage 2 (Специализированный анализ)

На вырезанных фрагментах высокое разрешение сохраняется для глубокого анализа.

1. **Классификация патологий (Вымя / Навоз):**
* **Модели:** `ConvNeXt-Base` или `EfficientNet-B4` (вход 512x512).
* Оценка асимметрии вымени, покраснений, консистенции навоза.


2. **Оценка спины и хромоты (Pose Estimation):**
* **Модель:** `YOLO11-Pose` (или `MMPose` Top-Down).
* **5 Ключевых точек позвоночника:**
1. $P_1$ — Холка (Withers)
2. $P_2$ — Грудной отдел (Thoracic)
3. $P_3$ — Поясница (Lumbar / Spine Arch)
4. $P_4$ — Крестец (Sacrum)
5. $P_5$ — Корень хвоста (Tailhead)


* **Вычисление геометрического угла ($\theta$):**
Математический угол искривления спины вычисляется по векторам $\vec{V}_{23} = P_3 - P_2$ и $\vec{V}_{34} = P_4 - P_3$:

$$\theta = \arccos\left(\frac{\vec{V}_{23} \cdot \vec{V}_{34}}{\Vert{}\vec{V}_{23}\Vert{} \Vert{}\vec{V}_{34}\Vert{}}\right)$$



Если $\theta < 165^\circ$ на протяжении шага — фиксируется выгнутая спина (признак хромоты).



---

### Этап 5: Временная агрегация и Трекинг

Корова находится в динамике, поэтому решения по единичному кадру недопустимы.

1. **Присвоение ID в потоке (Tracking):**
* **ByteTrack / BoT-SORT:** Трекинг объектов с использованием Motion Model (Фильтр Калмана) + Re-ID эмбеддингов для сохранения ID животного даже при временном перекрытии другими коровами.


2. **Алгоритм скользящего окна (Rolling Window):**
* Фиксация патологии производится только при выполнении условия:

$$\text{Pathology\_Detected} = \text{True} \quad \Longleftrightarrow \quad \frac{1}{N} \sum_{t=1}^{N} \mathbb{I}(\text{Score}_t > \text{Threshold}) \ge 0.8$$


* Патология считается валидной, если она удерживается минимум $N = 90$ кадров подряд (~3 секунды при 30 FPS). Это полностью исключает ложные срабатывания, когда корова просто чешется или встряхивается.


3. **Модульный Агрегатор (Modular Ensemble):**
```python
# Логика работы Агрегатора верхнего уровня (XGBoost / Rule Engine)
def evaluate_cow_health(cow_id, tracking_window_data):
    lameness_score = model_a_gait.predict(tracking_window_data['depth_gait'])  # 1.0 - 5.0
    bcs_score = model_b_bcs.predict(tracking_window_data['depth_3chan'])       # 1.0 - 5.0
    udder_anomaly_prob = model_c_udder.predict(tracking_window_data['crop_rgb']) # 0.0 - 1.0

    if lameness_score >= 3.0 or udder_anomaly_prob > 0.75:
        send_alert(
            cow_id=cow_id,
            status="NEEDS_INSPECTION",
            reason=f"Lameness: {lameness_score}, Udder Prob: {udder_anomaly_prob:.2f}"
        )

```



---

### Этап 6: Валидация, Active Learning и HPO

1. **Метрики:**
* Отдельные PR-кривые (Precision-Recall) и mAP50 для больных классов.
* Анализ матриц ошибок (**Confusion Matrix**) для отслеживания False Positives (тени, принятые за вымя) и False Negatives (пропущенная хромота в темноте).


2. **Петля Active Learning:**
* Отбор 200–300 сложных примеров с максимальной энтропией (Uncertainty Sampling):

$$H(X) = -\sum p(x) \log p(x)$$


* Кадры отправляются в CVAT на ручную переразметку ветеринаром и возвращаются в Train.


3. **Подбор гиперпараметров (HPO):**
* Автоматический поиск через `model.tune()` (Optuna / Ray Tune) для оптимизации $lr_0$, $weight\_decay$, параметров Focal Loss $\gamma$ и коэффициентов аугментации Mosaic/Mixup.



---

### Полный технологический стек (Master Tech Stack)

| Компонент / Задача | Технология / Библиотека |
| --- | --- |
| **Обработка Depth & Видео** | OpenCV, NumPy, SciPy (3-channel generation, Distance Limiting) |
| **Дедупликация & Фильтры** | `imagehash` (pHash), `OpenCLIP`, PyTorch |
| **Smart Labeling (Zero-Shot)** | Grounding DINO, SAM 2 (Segment Anything 2) |
| **VLM & Авто-классификация** | Qwen2.5-VL-7B-Instruct / Gemini 1.5 Pro |
| **Интерфейс разметки** | CVAT / Label Studio |
| **Аудит данных & Метрик** | Cleanlab, FiftyOne |
| **Stage 1 (Detection & Crop)** | YOLOv11x / YOLOE-26 (Ultralytics) |
| **Stage 2 (Classifiers)** | PyTorch, `timm` (ConvNeXt, EfficientNet-B4) |
| **Stage 2 (Pose Estimation)** | YOLO11-Pose / MMPose |
| **Tracking & Re-ID** | ByteTrack / BoT-SORT |
| **Ensemble Aggregator** | XGBoost / LightGBM / Python Rule Engine |
| **Hyperparameter Tuning** | Optuna / Ray Tune (`model.tune()`) |

---

### Аппаратная инфраструктура (Edge vs. Cloud Strategy)

1. **Локальный Edge-узел (Directly on Farm):**
* **Оборудование:** NVIDIA Jetson AGX Orin (64GB) или ПК с NVIDIA RTX 4060 Ti / RTX 4070 на каждую проходную секцию коровника.
* **Задачи на Edge:** Прием видеопотока 1080p@30fps, фильтрация расстояния, дедупликация, запуск Stage 1 (YOLOv11), ByteTrack и извлечение Pose-ключевых точек.


2. **Центральный сервер (On-Prem Server / Cloud):**
* **Оборудование:** Сервер с NVIDIA RTX 4090 / L40S.
* **Задачи на сервере:** Запуск тяжелых Stage 2 классификаторов (`ConvNeXt`), сведение данных в скользящем окне, выполнение логики Агрегатора, отправка пуш-уведомлений ветеринарам и дообучение моделей в рамках Active Learning.