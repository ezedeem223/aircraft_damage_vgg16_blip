# Aircraft Damage Classification & Captioning (VGG16 + BLIP)

## 📌 Project Overview
This project aims to automate the inspection of aircraft surfaces by classifying damage types and generating descriptive captions. It combines computer vision and natural language processing techniques to assist in aircraft maintenance and safety checks.

The solution consists of two main parts:
1.  **Damage Classification:** Classifying aircraft surface images into two categories: **"Dent"** vs. **"Crack"** using a Transfer Learning approach with **VGG16**.
2.  **Image Captioning & Summarization:** Generating detailed textual descriptions and summaries of the aircraft images using the **BLIP (Bootstrapping Language-Image Pretraining)** Transformer model.

## 📂 Dataset
The dataset used in this project is the **Aircraft Damage Dataset**.
- **Source:** [Roboflow Aircraft Dataset](https://universe.roboflow.com/youssef-donia-fhktl/aircraft-damage-detection-1j9qk)
- **License:** CC BY 4.0
- **Structure:**
    - `train/`: Training images (Dent/Crack)
    - `valid/`: Validation images
    - `test/`: Testing images
- **Preprocessing:** Images are rescaled (1./255) and resized to **224x224** pixels.

## 🛠️ Tech Stack & Libraries
- **Language:** Python 3.12+
- **Deep Learning Frameworks:** TensorFlow/Keras, PyTorch
- **Libraries:**
    - `transformers` (Hugging Face)
    - `matplotlib`, `numpy`, `pandas`
    - `scikit-learn` (implied for metrics)

## 🧠 Model Architectures

### 1. Classification Model (VGG16)
We utilize **VGG16** pre-trained on ImageNet as a feature extractor.
- **Base Model:** VGG16 (layers frozen).
- **Custom Head:**
    - Flatten Layer
    - Dense (512, ReLU)
    - Dropout (0.3)
    - Dense (512, ReLU)
    - Dropout (0.3)
    - Output Dense (1, Sigmoid) for binary classification.
- **Optimizer:** Adam (`lr=0.0001`)
- **Loss Function:** Binary Crossentropy

### 2. Captioning Model (BLIP)
We use the **Salesforce/blip-image-captioning-base** model from Hugging Face.
- A custom Keras Layer (`BlipCaptionSummaryLayer`) was implemented to integrate the PyTorch-based BLIP model into the TensorFlow workflow.
- Capabilities: Generates captions (e.g., "a close up of a plane engine") and summaries.

## 🚀 Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd aircraft-damage-project

2. **Install dependencies: It is recommended to use a virtual environment**
    ```Bash
    pip install -r requirements.txt
    Note: Ensure you have tensorflow, torch, transformers, pillow, and matplotlib installed.
3. **Run the Notebook: Open notebooks/aircraft_damage_vgg16_blip.ipynb in Jupyter Notebook or VS Code and execute the cells sequentially.**
📊 Results & Performance
Classification (VGG16)
    The model was trained for 5 epochs.

    Metrics:

        Training Accuracy: ~88%

        Validation Accuracy: ~70% (Subject to dataset variability)

        Test Loss: 0.7326

        Test Accuracy: 0.6875

    Visualizations included in the notebook show the Loss and Accuracy curves over epochs.

Captioning (BLIP)
    The model successfully generates context-aware descriptions for the input images.

    Example Output:

        Input: Image of a Boeing nose.

        Generated Caption: "this is a detailed photo showing the engine of a boeing 747"

**Project Structure**
├── data/                   # Dataset directory
├── notebooks/              # Jupyter notebooks
│   └── aircraft_damage_vgg16_blip.ipynb
├── outputs/                # Saved plots
├── .gitignore
├── README.md
└── requirements.txt

**License**
This project uses data provided by a Roboflow user under the CC BY 4.0 license.