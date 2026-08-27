# Intel Image Classification

Image classification project for the CNN assignment using the [Intel Image Classification dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification).

## Dataset

- **Classes:** buildings, forest, glacier, mountain, sea, street
- **Images:** ~25,000
- **Splits:**
  - Training: 80% of `seg_train`
  - Validation: 20% of `seg_train`
  - Test: `seg_test`

## Model

Sequential CNN built with Keras:
- EfficientNetB0 pretrained base (frozen, then fine-tuned)
- GlobalAveragePooling2D
- BatchNormalization + Dense + Dropout head
- Softmax output for 6 classes

## Results

| Split      | Accuracy |
|------------|----------|
| Train      | 97.10%   |
| Validation | 93.76%   |
| Test       | 93.33%   |

The 85% minimum accuracy requirement is met. The 95% bonus target was not reached on this run.

## Notes

- GPU training was attempted but the RTX 5060 (compute capability 12.0) is not supported by TensorFlow 2.19, so training runs on CPU.
- The notebook disables GPU via `CUDA_VISIBLE_DEVICES=-1`.
- `tensorflowjs` 4.22.0 depends on `tensorflow-decision-forests`, which requires a newer protobuf than TF 2.19 allows. The notebook patches the protobuf runtime version check to allow TFJS export.

## Requirements

- Python 3.10+
- CPU training (GPU not usable with this TF/version combination)

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Kaggle credentials

The notebook downloads the dataset from Kaggle. Set credentials before running:

```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_key
```

Or place `kaggle.json` in `~/.kaggle/`:

```bash
mkdir -p ~/.kaggle
cp /path/to/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

## Run

```bash
jupyter notebook notebook.ipynb
```

Run all cells. The notebook will:
1. Download the dataset
2. Train the model
3. Plot accuracy and loss
4. Evaluate on train/validation/test sets
5. Export the model to SavedModel, TF-Lite, and TFJS formats
6. Run inference and display predictions