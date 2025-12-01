# **✈️ AI Runway Detector (Dual-Model Architecture)**
A robust computer vision system for autonomous aircraft landing guidance. This project utilizes a Dual-Model Architecture to detect both the safe landing area (Segmentation) and the precise geometric runway lines (Regression) in real-time.

## 🌟 Features

Dual-Stream Inference:

branch A (U-Net): Performs Semantic Segmentation to identify the drivable runway area (Green Mask).

branch B (Regressor): Uses a ResNet18-based keypoint regressor to predict the exact coordinates of the Center, Left, and Right lines.

Robustness: Trained on synthetic data (FS2020) with heavy data augmentation (brightness, noise, contrast) to bridge the Sim-to-Real gap.

Visual Feedback: Provides a live overlay of the detected area and flight lines on cockpit video feeds.

Easy Deployment: Includes a Gradio web app for browser-based testing and visualization.

Live Demo: Check out the running app on Hugging Face Spaces: https://huggingface.co/spaces/rizzam/runway_detector/tree/main

## 🛠️ Installation

#### Clone the Repository:

git clone [https://github.com/YOUR_USERNAME/runway_detector.git](https://github.com/YOUR_USERNAME/runway_detector.git)
cd runway_detector



#### Install Dependencies:

pip install -r requirements.txt



#### Initialize Git LFS (Important for Models):

git lfs install
git lfs pull



## 🚀 Usage

1. Run the Web App (Gradio)

Launch the interactive web interface to test the model on your own videos.

python app.py



Open your browser to http://localhost:7860.

2. Training the Models

The project includes two separate training pipelines in Jupyter Notebooks:

runway-detector.ipynb: Trains the U-Net for Runway Area Segmentation.

runwaylinedetector.ipynb: Trains the ResNet18 Regressor for Line Coordinate Prediction.

To train:

Download the FS2020 Runway Dataset from Kaggle: FS2020 Runway Dataset

Update the TRAIN_IMG_DIR paths in the notebooks.

Run the cells to train and save best_unet.pth and best_regressor.pth.

## 🧠 Model Architecture

1. Area Detector (Segmentation)

Architecture: U-Net

Backbone: ResNet34 (Pre-trained on ImageNet)

Input: RGB Image (640x360)

Output: Binary Mask (Runway vs. Background)

Loss Function: Dice Loss + Binary Cross Entropy

2. Line Detector (Regression)

Architecture: ResNet18 + Custom Head

Input: RGB Image (640x360)

Output: 12 Coordinates (Start/End points for Left, Right, Center lines) + 3 Presence Probabilities.

Loss Function: Masked Wing Loss (for precision) + BCE (for presence classification).

## 📂 Directory Structure
```
runway_detector/
├── app.py                  # Gradio inference application
├── requirements.txt        # Python dependencies
├── best_unet.pth           # Trained weights for Area model
├── best_regressor.pth      # Trained weights for Line model
├── runway-detector.ipynb   # Training notebook for U-Net
├── runwaylinedetector.ipynb# Training notebook for Regressor
└── README.md               # Project documentation
```


## 📊 Dataset

The models were trained on the FS2020 Runway Dataset, which contains high-quality synthetic images from Microsoft Flight Simulator 2020.

Resolution: 1920x1080 (Downscaled to 640x360 for training)

Conditions: Day, Night, Fog, Rain, Clear.

## 🤝 Contributing

Contributions are welcome! Please fork the repository and submit a Pull Request.

## 📜 License

This project is licensed under the MIT License.

**Author: Rizzam**

