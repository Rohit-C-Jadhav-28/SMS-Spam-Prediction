# SMS Spam Prediction using Machine Learning 📱

## Overview
This project implements a machine learning solution for detecting spam SMS messages. Using various classification algorithms, we've developed a robust system that can effectively distinguish between spam and legitimate (ham) messages.

## 🎯 Project Highlights
- Multiple classifier implementations for comparison
- High accuracy rates across different models
- Web interface for real-time prediction
- Easy-to-use API for message classification

## 🤖 Machine Learning Models & Performance
We've implemented and compared several classification algorithms:

1. Logistic Regression: 98.03% accuracy
2. MultinomialNB: 97.04% accuracy
3. Support Vector Classification (SVC): 96.86% accuracy
4. LogisticRegressionCV: 95.52% accuracy
5. K-Neighbors Classifier: 91.84% accuracy
6. Random Forest Classifier: 91.84% accuracy

## 🛠️ Technologies Used
- Python 3.x
- Scikit-learn
- Flask (for web interface)
- NLTK
- Pandas
- NumPy

## 📦 Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/sms-spam-prediction.git

# Navigate to project directory
cd sms-spam-prediction

# Install required packages
pip install -r requirements.txt
```

## 🚀 Usage
1. Run the Flask application:
```bash
python app.py
```
2. Open your web browser and navigate to `http://localhost:5000`
3. Enter the SMS message you want to classify
4. Click "Predict" to see the results

## 📂 Project Structure
```
SMS-Spam-Prediction/
│
├── templates/
│   ├── index.html
│
├── models/
│   ├── LogisticRegression.pkl
│   ├── LogisticRegressionCV.pkl
│   ├── RandomForestClassifier.pkl
│   ├── Spam_SMS_Tfidf.pkl
│   └── MultinomialNB.pkl
│
├── data/
│   └── spam.csv
│
├── app.py
├── main.py
├── main.ipynb
├── requirements.txt
└── README.md
```

## 🔍 Model Training
The models were trained on a dataset containing both spam and ham messages. The preprocessing pipeline includes:
- Text cleaning
- Tokenization
- Stop words removal
- TF-IDF vectorization

## 🌟 Features
- Real-time SMS classification
- Multi-model comparison
- User-friendly web interface
- High accuracy rates
- Fast prediction response

## 📊 Performance Metrics
```python
Model Accuracies = {
    'Logistic Regression': 0.9704,
    'MultinomialNB': 0.9803,
    'SVC': 0.9686,
    'LogisticRegressionCV': 0.9552,
    'KNeighbors': 0.9184,
    'Random Forest': 0.9184
}
```

## 🤝 Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 👥 Authors
- Rohit Chandrakant Jadhav

## 🙏 Acknowledgments
- Dataset providers
- Open source community
- Contributors and testers

## 📞 Contact
- **Rohit Chandrakant Jadhav** - rcjadhav005@gmail.com
- Project Link: https://github.com/yourusername/sms-spam-prediction
