# Intel Image Classification

Image classification project for the CNN assignment using the [Intel Image Classification dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification).

## Dataset

- **Classes:** buildings, forest, glacier, mountain, sea, street
- **Images:** ~25,000 total, 14,034 in the source `seg_train` folder used for this project
- **Splits:**
  - Training: 80% of `seg_train` (11,227 images)
  - Validation: 10% of `seg_train` (1,403 images)
  - Test: 10% of `seg_train` (1,404 images)

  The `seg_train` folder was manually re-split using stratified sampling so that the test set is created from the same source distribution as the training data, not from the original `seg_test` folder.

## Model

Sequential CNN built with `tf.keras.Sequential`:

- MobileNetV2 pretrained base
- Explicit Conv2D + MaxPooling2D top layers
- GlobalAveragePooling2D
- BatchNormalization + Dense + Dropout head
- Softmax output for 6 classes

MobileNetV2 preprocessing (`preprocess_input`, scaling to [-1, 1]) is applied to all splits. Light data augmentation (random flip, rotation, zoom, contrast) is applied to the training set only.

## Training

Two-stage training on CPU:

1. **Feature extraction:** train the custom top layers while the MobileNetV2 base is frozen.
2. **Fine-tuning:** unfreeze the last 10 MobileNetV2 layers and continue training at a low learning rate (1e-5) with early stopping and learning-rate reduction.

Callbacks used: `ModelCheckpoint`, `EarlyStopping`, `ReduceLROnPlateau`.

## Results

| Split      | Accuracy |
|------------|----------|
| Train      | 93.32%   |
| Validation | 90.88%   |
| Test       | 90.67%   |

The 85% minimum accuracy requirement is met. The 95% bonus target was not reached on this run.

## Export Formats

The trained model is exported to:

- `saved_model/` — standard TensorFlow SavedModel
- `tflite/` — TF-Lite model + `label.txt`
- `tfjs_model/` — TensorFlow.js model

## Inference

The notebook loads the TF-Lite model, runs predictions on a sample test batch, and displays the images with true/predicted labels. Preprocessed images are de-normalized back to [0, 255] before display so they appear correctly.

## Submission Structure

```
submission/
├── tfjs_model/
│   ├── model.json
│   └── group1-shard*.bin
├── tflite/
│   ├── model.tflite
│   └── label.txt
├── saved_model/
│   ├── saved_model.pb
│   └── variables/
├── notebook.ipynb
├── README.md
└── requirements.txt
```

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
2. Split it into train/validation/test sets
3. Apply preprocessing and augmentation
4. Train the top layers
5. Fine-tune the last 10 MobileNetV2 layers
6. Plot accuracy and loss
7. Evaluate on train/validation/test sets
8. Export the model to SavedModel, TF-Lite, and TFJS formats
9. Run inference and display predictions
