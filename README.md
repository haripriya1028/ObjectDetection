# Object Detection using TensorFlow

A computer vision project that classifies hand signs and predicts their locations using a dual-head Convolutional Neural Network built with TensorFlow and Keras.

Developed as part of the **Artificial Intelligence with Python internship at Coincent**.

## Overview

This project adapts the Sign Language MNIST classification dataset into a single-object detection task.

Each 28 × 28 grayscale image is randomly placed on a larger 75 × 75 blank canvas. The placement coordinates generate the ground-truth bounding box, allowing the model to learn both sign classification and localization.

## Features

- Synthetic bounding-box generation without manual annotation
- Shared CNN feature extraction
- Separate classification and bounding-box regression heads
- Efficient input processing using `tf.data`
- Classification accuracy and Intersection over Union evaluation
- Visualization of predicted and ground-truth bounding boxes

## Technologies Used

- Python
- TensorFlow and Keras
- NumPy
- pandas
- Matplotlib
- Pillow
- Google Colab
- Kaggle API

## Dataset

The project uses **Sign Language MNIST**, available on Kaggle as `datamunge/sign-language-mnist`.

| Property | Details |
|----------|---------|
| Original image dimensions | 28 × 28 |
| Image format | Grayscale |
| Training samples | 27,455 |
| Model input dimensions | 75 × 75 × 1 |
| Pixel normalization | Division by 255 |
| Bounding-box format | `[xmin, ymin, xmax, ymax]` |

The dataset contains 24 static ASL letters, excluding J and Z because they involve motion. The implementation uses a 25-unit classification head to retain the original label indices, including an unused index.

## Methodology

### 1. Data Preparation

- Load training and test samples from CSV files.
- Reshape pixel values into 28 × 28 grayscale images.
- Normalize pixel values to the range `[0, 1]`.
- Convert class labels into one-hot vectors.

### 2. Synthetic Bounding-Box Generation

- Randomly position each sign image on a 75 × 75 canvas.
- Calculate the inserted image patch's bounding box.
- Normalize coordinates relative to the canvas dimensions.

### 3. Model Development

A shared CNN extracts image features before passing them to two output heads:

- **Classification head:** Predicts the hand-sign label.
- **Regression head:** Predicts four bounding-box coordinates.

### 4. Evaluation

- Measure classification accuracy.
- Calculate Intersection over Union for bounding boxes.
- Display predictions alongside ground-truth labels and boxes.

## Model Architecture

| Layer or Component | Configuration |
|--------------------|---------------|
| Input | 75 × 75 × 1 |
| Convolution block 1 | Conv2D with 16 filters, ReLU, AveragePooling2D |
| Convolution block 2 | Conv2D with 32 filters, ReLU, AveragePooling2D |
| Convolution block 3 | Conv2D with 64 filters, ReLU, AveragePooling2D |
| Shared dense layer | Flatten → Dense with 128 units and ReLU |
| Classification output | Dense with 25 units and softmax |
| Bounding-box output | Dense with 4 units and linear activation |
| Trainable parameters | 428,573 |

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Classification loss | Categorical cross-entropy |
| Bounding-box loss | Mean squared error |
| Batch size | 64 |
| Epochs | 10 |

## Results

| Metric | Result |
|--------|--------|
| Training classification accuracy | 82.4% |
| Validation classification accuracy | 76.98% |
| IoU in displayed examples | Predominantly 0.6–0.9 |

Intersection over Union measures the overlap between predicted and reference boxes:

**IoU = Intersection Area / Union Area**

The reported IoU range describes displayed prediction examples rather than a dataset-wide mean. Classification accuracy measures the classification head and should not be interpreted as a complete detection metric such as mAP.

The supplied test split was used as validation data during this experiment.

## How to Run

### Google Colab

1. Open the project notebook in Google Colab.
2. Add your Kaggle API token to Colab Secrets as `KAGGLE_API_TOKEN`.
3. Run the dependency installation and dataset download cells.
4. Run the remaining cells in order.
5. Review the training results and bounding-box visualizations.

### Local Setup

Install the required libraries:

    pip install tensorflow numpy pandas matplotlib pillow kaggle jupyter

Download and extract Sign Language MNIST. Ensure the dataset paths match the implementation:

    sign_language_mnist/sign_mnist_train.csv
    sign_language_mnist/sign_mnist_test.csv

Open the notebook in Jupyter and run the cells sequentially.

Replace Colab-specific authentication code with local Kaggle authentication or download the dataset manually. Do not commit API credentials.

## Limitations

- Each image contains a single sign patch on a simple background.
- Bounding boxes describe the inserted patch rather than a tightly annotated hand.
- Performance on synthetic images does not establish performance in natural scenes.
- Similar-looking signs can be misclassified.
- Random placement can produce different inputs between evaluation runs.

## Future Improvements

- Add image augmentation and varied backgrounds.
- Evaluate on real images with manually annotated bounding boxes.
- Report dataset-wide localization metrics.
- Explore deeper CNN architectures.
- Tune the balance between classification and regression losses.

