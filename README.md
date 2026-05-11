# 🧠 Alzheimer's Detection System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-3.10.0-red?logo=keras&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

A deep learning-powered web application that detects stages of Alzheimer's disease from MRI brain scans. Built with **Keras/TensorFlow** and deployed via **Streamlit**.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🖼️ MRI Image Upload | Upload brain MRI scans in JPG, JPEG, or PNG format |
| 🔬 Multi-class Classification | Classifies into 4 stages — Non Demented, Very Mild Dementia, Mild Dementia, Moderate Dementia |
| 📊 Confidence Scores | Displays prediction confidence and probability breakdown for all classes |
| 🧪 Sample Image Testing | Quickly test the model using a built-in sample MRI image |
| ⚕️ Medical Disclaimer | Clearly labeled for educational/research use only |

---

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Deep Learning**: [TensorFlow 2.15](https://www.tensorflow.org/) + [Keras 3.10](https://keras.io/)
- **Image Processing**: [Pillow (PIL)](https://pillow.readthedocs.io/)
- **Numerical Computing**: [NumPy](https://numpy.org/)

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/raghav-decoded/alzheimers_detection.git
   cd alzheimers_detection
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # macOS / Linux
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the App

```bash
streamlit run app.py
```

Then open your browser to **http://localhost:8501** and:
1. Click **"Use Sample Image"** to test with the built-in MRI scan, or upload your own.
2. View the predicted dementia stage and confidence scores.

---

## 🧬 Model Details

The application uses a pre-trained Keras model (`alzheimers_model.keras`) trained to classify brain MRI images into four categories:

| Class | Description |
|---|---|
| Non Demented | No signs of dementia |
| Very Mild Dementia | Early-stage cognitive decline |
| Mild Dementia | Noticeable cognitive impairment |
| Moderate Dementia | Significant cognitive decline |

**Input**: 224×224 grayscale MRI images (converted to 3-channel internally)  
**Output**: Probability distribution across the 4 classes

---

## 📁 Project Structure

```
alzheimers_detection/
├── app.py                    # Main Streamlit application
├── alzheimers_model.keras    # Pre-trained deep learning model
├── requirements.txt          # Python dependencies
├── .gitignore                # Git ignore rules
└── README.md                 # Project documentation
```

---

## ⚠️ Disclaimer

> **This tool is for educational and research purposes only.** It is NOT a substitute for professional medical diagnosis. Always consult a qualified healthcare provider for accurate assessment and treatment.

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -m "Add my feature"`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use it for educational purposes.
