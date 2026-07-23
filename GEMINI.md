# Cow Body Condition Scoring (BCS) Project

This project, named **cow**, is an image regression deep learning system designed to evaluate the **Body Condition Score (BCS)** of cows. BCS is an index ranging from 1.0 to 5.0 (with 0.25 steps) that measures the nutritional status, body fat, and muscle of dairy cows. The system uses top-down view (Top-View) camera imagery of cows to predict the continuous BCS.

---

## 🏗️ Project Architecture

The codebase is highly modular and written in Python using **PyTorch**, **timm** (PyTorch Image Models), and **Albumentations**.

```
/home/georgiy/projects/ml/cow/
├── pyproject.toml      # Project configuration & dependencies (using uv)
├── uv.lock             # Lockfile for reproducible python env
├── main.py             # Basic entrypoint / placeholder script
├── config/             # Folder for YAML configurations (currently empty)
├── checkpoints/        # Saved model weights (.pt files)
├── data/               # Dataset directory (expects raw/images and raw/labels)
├── notebook/           # Jupyter Notebooks for exploratory data analysis
│   └── eda.ipynb
├── src/                # Main source code
│   ├── config.py       # Configuration parser (currently placeholder)
│   ├── data.py         # PyTorch Dataset, YOLO labels loader & Albumentations pipelines
│   ├── losses.py       # Custom loss functions (Wing Loss, Weighted Smooth L1 Loss)
│   ├── metrics.py      # Evaluation metrics (MAE, RMSE, tolerances ±0.25, ±0.50)
│   ├── models.py       # PyTorch model using timm (ConvNeXt-Small backbone)
│   ├── trainer.py      # DDP Training & validation loop with AMP
│   └── utils.py        # Utilities (EarlyStopping, etc.)
└── test/               # Unit and integration tests (currently empty)
```

### Key Components

1. **Dataset & Preprocessing (`src/data.py`):**
   - **`CowBCSDataset`**: Loads YOLO-format labels (`class_id xc yc w h`). The dataset automatically crops the cow body using the bounding box with customizable padding (`bbox_padding=0.1`).
   - **BCS Mapping**: Translates categorical class IDs ($0$ to $16$) into continuous BCS values ($1.0$ to $5.0$, with a step of $0.25$):
     - `0` ➡️ `1.0`
     - `8` ➡️ `3.0`
     - `16` ➡️ `5.0`
   - **Augmentations**: Uses Albumentations optimized for a top-down view (Horizontal flips, Affine scaling/rotation, Color jitter, Random gamma, Motion blur, Gaussian noise).
   - **Weighted Sampler**: Implements a `WeightedRandomSampler` based on class frequencies to counter severe class imbalance.

2. **Model Definition (`src/models.py`):**
   - **`CowBCSModel`**: Based on `timm` backbones (default: `convnext_small.fb_in22k_ft_in1k_384`) with a custom **CBAM (Convolutional Block Attention Module)** for channel and spatial attention to focus on critical body features.
   - **Bias Initialization**: The final linear head's bias is initialized to the dataset's average BCS score (e.g. `2.88`) to stabilize early training.

3. **Custom Loss Functions (`src/losses.py`):**
   - **`WingLoss`**: Designed for high-accuracy continuous regression, configurable via `omega` and `epsilon` parameters for fine-tuning gradient focus on small errors.
   - **`WeightedSmoothL1Loss`**: Smooth L1 loss with optional sample weighting.
   - **`get_loss_function`**: A factory function supporting `"smooth_l1"`, `"l1"`, `"mse"`, `"wing"`, `"weighted_smooth_l1"`, `"huber"`, and `"ordinal"`.

4. **Metrics (`src/metrics.py`):**
   - **MAE & RMSE**: Mean Absolute Error and Root Mean Squared Error.
   - **`acc_exact`**: Exact matching accuracy after rounding prediction to the nearest 0.25 interval.
   - **`acc_tol_0.25`**: Percentage of predictions within $\pm0.25$ error.
   - **`acc_tol_0.50`**: Percentage of predictions within $\pm0.50$ error.

5. **Distributed Training Loop (`src/trainer.py`):**
   - Uses PyTorch **DistributedDataParallel (DDP)** to scale training across multiple GPUs.
   - Employs **Automatic Mixed Precision (AMP)** for fast, memory-efficient GPU computing.
   - Employs gradient clipping (`max_norm=10.0`) to shield learning from noisy labels.
   - Uses **AdamW** optimizer paired with a `SequentialLR` scheduler (linear warmup followed by Cosine Annealing decay).

---

## 🛠️ Building, Running & Testing

This project uses `uv` for lightning-fast python virtual environment and dependency management.

### Environment Setup

Install dependencies and create virtual environment:
```bash
# Sync dependencies from uv.lock
uv sync
```
Or install manually via pip:
```bash
pip install -e .
```

### Running Verification & Module Smoke Tests
Every core module in `src/` contains a `__main__` entrypoint that verifies its own basic functionality on dummy data. You can run them to ensure packages and CUDA interfaces are functional:

```bash
python src/data.py
python src/losses.py
python src/models.py
python src/metrics.py
```

### Running Distributed Training (DDP)
Since `src/trainer.py` uses `torch.distributed`, training must be launched via `torchrun`:

```bash
torchrun --nproc_per_node=<num_gpus> src/trainer.py
```

### Testing (TODO)
The `test/` directory is currently empty.
*To run any future test suites added to the project:*
```bash
pytest test/
```

---

## 📝 Development Conventions

- **Docstrings & Comments:** Core algorithms and mathematical/domain choices (such as BCS mapping or Top-View crop logic) are fully documented using Russian comments/docstrings to maintain contextual depth.
- **Config Management:** Currently, config variables are defined inside `src/trainer.py` as constants. Future iterations should load configs from YAML files via `src/config.py`.
- **DDP-Aware Code:** Ensure any logic modified in the training loop accounts for distributed environments (e.g., computing metrics or saving checkpoints only on `is_master` rank, setting epoch on `DistributedSampler`).
- **Data Splitting:** When splitting data, ensure the division is done on **cow ID** boundaries rather than individual images to prevent target leakage between the train and val sets.
- **Noisy Label Control:** Always keep gradient clipping enabled (`max_norm=10.0` or similar) to handle noisy/imperfect target labels in regression.
