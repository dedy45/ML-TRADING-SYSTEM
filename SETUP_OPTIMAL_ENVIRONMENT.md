# Setup Optimal Environment untuk Deep Learning Fibonacci Trading
# Panduan Lengkap - June 12, 2025

## 🎯 Tujuan
- Install Anaconda dengan Python 3.11 (optimal untuk TensorFlow)
- Setup environment khusus deep learning
- Install semua dependencies yang diperlukan
- Restart enhanced analysis dengan performa maksimal

## 📋 Step-by-Step Installation Guide

### Step 1: Download dan Install Anaconda

1. **Download Anaconda**
   - Buka: https://www.anaconda.com/download
   - Pilih: **Anaconda Individual Edition**
   - Download: **Windows 64-bit** (Python 3.11)
   - File size: ~900MB

2. **Install Anaconda**
   - Run installer sebagai Administrator
   - Pilih: "Add Anaconda to PATH" ✅
   - Pilih: "Register Anaconda as default Python" ✅
   - Install location: `C:\Anaconda3\` (recommended)

### Step 2: Verify Installation

Setelah install, buka **Anaconda Prompt** dan test:

```bash
# Check conda version
conda --version

# Check Python version
python --version

# List environments
conda env list
```

### Step 3: Create Dedicated Environment

```bash
# Create environment khusus untuk deep learning fibonacci
conda create -n fibonacci_dl python=3.11 -y

# Activate environment
conda activate fibonacci_dl

# Verify Python version
python --version  # Should show Python 3.11.x
```

### Step 4: Install Core Packages

```bash
# Install essential scientific packages
conda install numpy pandas scikit-learn matplotlib seaborn jupyter -y

# Install TensorFlow dengan GPU support (optional)
pip install tensorflow[and-cuda]

# Alternative: CPU-only TensorFlow
pip install tensorflow

# Install additional ML packages
pip install mlflow optuna hyperopt

# Install trading-specific packages
pip install yfinance ta-lib pandas-ta

# Install utilities
pip install tqdm pyyaml python-dotenv
```

### Step 5: Verify TensorFlow Installation

```python
# Test TensorFlow
python -c "
import tensorflow as tf
print('TensorFlow version:', tf.__version__)
print('GPU available:', tf.config.list_physical_devices('GPU'))
print('✅ TensorFlow ready!')
"
```

## 🏗️ Project Structure Optimal

```
fibonacci_deep_learning/
├── environment.yml           # Conda environment file
├── requirements.txt          # Pip requirements
├── setup_environment.py      # Automated setup script
├── data/
│   ├── raw/                 # Original CSV files
│   ├── processed/           # Cleaned data
│   └── features/            # ML features
├── models/
│   ├── tensorflow/          # TensorFlow models
│   ├── sklearn/             # Scikit-learn models
│   └── saved/               # Trained models
├── notebooks/               # Jupyter analysis
├── src/
│   ├── data/               # Data processing
│   ├── models/             # Model architectures
│   ├── training/           # Training pipelines
│   └── utils/              # Utilities
├── experiments/            # MLflow tracking
├── reports/                # Analysis reports
└── deployment/             # Production code
```

## 🎯 Next Steps After Installation

1. **Restart Computer** (to ensure PATH updates)
2. **Open Anaconda Prompt**
3. **Navigate to project**: `cd E:\aiml\MLFLOW`
4. **Activate environment**: `conda activate fibonacci_dl`
5. **Run setup script**: `python setup_optimal_environment.py`

## ⚠️ Important Notes

- **Uninstall old Python**: Windows Settings → Apps → Python 3.13.4 → Uninstall
- **Use Anaconda Prompt**: Always use Anaconda Prompt, bukan Command Prompt biasa
- **Environment isolation**: Selalu activate `fibonacci_dl` environment sebelum coding
- **GPU support**: Jika punya NVIDIA GPU, install CUDA toolkit untuk performa maksimal

## 🔧 Troubleshooting

### Jika TensorFlow gagal install:
```bash
pip install --upgrade pip
pip install tensorflow --no-cache-dir
```

### Jika import error:
```bash
conda update --all
pip install --upgrade tensorflow
```

### Jika environment conflict:
```bash
conda remove -n fibonacci_dl --all
# Repeat Step 3-4
```

---

**Ready?** Mari kita mulai installation! 🚀
