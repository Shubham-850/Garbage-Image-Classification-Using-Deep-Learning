♻️ Garbage Classification using Deep Learning

A deep learning-based web application that classifies garbage images into multiple categories using TensorFlow (CNN + Transfer Learning) and provides real-time predictions through a Streamlit UI.

🚀 Features
🖼️ Upload garbage image
🤖 Predict waste category (12 classes)
📊 Show confidence score
🔝 Top-3 predictions
⚡ Fast and lightweight model (MobileNetV2)
🧠 Tech Stack
Python 🐍
TensorFlow / Keras
NumPy
Pillow (PIL)
Streamlit
📁 Project Structure
garbage_classifier/
│
├── garbage_classification/   # Dataset folder
│   ├── battery/
│   ├── biological/
│   ├── cardboard/
│   ├── clothes/
│   ├── glass/
│   ├── metal/
│   ├── paper/
│   ├── plastic/
│   ├── shoes/
│   ├── trash/
│   └── ...
│
├── models/
│   ├── model.h5
│   ├── classes.json
│
├── train.py
├── app.py
├── requirements.txt
└── README.md
⚙️ How It Works
Load dataset using flow_from_directory()
Preprocess images (resize, normalize)
Use MobileNetV2 (pretrained) for feature extraction
Train custom classification layers
Save model + class labels
Deploy via Streamlit
▶️ How to Run
1. Create Environment (Python 3.10 recommended)
py -3.10 -m venv dl_env
dl_env\Scripts\activate
2. Install Dependencies
pip install tensorflow streamlit pillow numpy
3. Train Model
python train.py
4. Run Streamlit App
python -m streamlit run app.py
📌 Input
Upload image (jpg/png/jpeg)
📈 Output
Predicted class
Confidence score
Top-3 predictions
🧠 Key Learnings
Transfer Learning using MobileNetV2
Image preprocessing and augmentation
Handling multi-class classification
Building end-to-end ML applications
Deploying models with Streamlit
🚀 Future Improvements
Improve accuracy with more training epochs
Add confusion matrix & training graphs
Deploy on cloud (Streamlit Cloud / Hugging Face)
Add real-time camera input
👨‍💻 Author

Shubham Rathore
