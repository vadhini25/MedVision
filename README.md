# COVID Detection Using MobileNet

## Introduction
This project is an implementation of a deep learning model using MobileNet for COVID-19 detection from X-ray images. It preprocesses X-ray images, applies data augmentation, trains a convolutional neural network, and evaluates the model using various accuracy metrics and visualizations.

## Features
- **Deep Learning Model:** Built using MobileNet architecture.
- **Image Preprocessing:** Uses image augmentation techniques.
- **Train/Test Split:** Splits dataset for training, validation, and testing.
- **Visualization:** Displays training loss, accuracy plots, and confusion matrix.
- **Accurate Predictions:** Predicts COVID status from X-ray images.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/covid-detection-mobilenet.git
   cd covid-detection-mobilenet
   ```
2. Install dependencies:
   ```bash
   pip install tensorflow keras numpy matplotlib seaborn scikit-learn opencv-python
   ```
3. Ensure you have access to the dataset in Google Drive (modify paths accordingly).

## Usage
1. Mount Google Drive and set up paths:
   ```python
   from google.colab import drive
   drive.mount('/gdrive')
   ```
2. Run the training script:
   ```python
   python train.py
   ```
3. Evaluate the model:
   ```python
   python evaluate.py
   ```
4. Run inference on new images:
   ```python
   python predict.py --image path/to/image.jpg
   ```

## Project Structure
```
Covid-detection-mobilenet
  - data  # Contains X-ray dataset
  -  models  # Saved trained models
  - notebooks  # Jupyter notebooks for experimentation
  - train.py  # Training script
  - evaluate.py  # Model evaluation script
  - predict.py  # Inference script
  - requirements.txt  # Dependencies
  - README.md  # Project documentation
```

## Results
- Achieved high accuracy using MobileNet.
- Confusion matrix visualization for better error analysis.

## Acknowledgements
- **Keras & TensorFlow:** Used for model implementation.
- **MediaPipe:** Assisted in pose detection (if applicable).
- **Open-source X-ray dataset.**

Feel free to contribute by improving the model, adding features, or optimizing performance. Fork the repository and submit a pull request!
