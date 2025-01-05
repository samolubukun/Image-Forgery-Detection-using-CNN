# Image Forgery Detection using CNN

This repository provides a deep learning-based solution for detecting image forgeries using Convolutional Neural Networks (CNNs). The system preprocesses images using Error Level Analysis (ELA) and predicts whether an image is real or forged with a trained CNN model.

![Screenshot (262)](https://github.com/user-attachments/assets/883982f1-7928-4839-a5ba-fc45f91f6cf3)


## Features
- **Streamlit Web App:** A user-friendly interface for uploading and analyzing images.
- **Trained CNN Model:** Achieves accurate predictions on real and forged images.

## Model Summary
The CNN model was trained on preprocessed ELA images and optimized for binary classification:
- **Architecture:** Sequential CNN with convolutional, pooling, and dense layers.
- **Input Shape:** 128x128 RGB images.
- **Metrics:** 
  - Accuracy: 92.5%
  - Precision: 91.8%
  - Recall: 93.0%
   **Access Training notebook: [Notebook](https://github.com/samolubukun/Image-Forgery-Detection-using-CNN/tree/main/Notebook)

Model download (download and place in the same folder as the streamlit app.py in order to run the system): [imageforgerydetection.h5](https://drive.google.com/file/d/1Z4IQ7ba1xIEzZxAoD3aH6Bdnb0gsVNLU/view?usp=sharing)

## How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/image-forgery-detection.git
   cd image-forgery-detection


2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Run the Streamlit app:
   ```bash
   streamlit run app.py

