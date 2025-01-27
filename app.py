from flask import render_template, request, Flask
import pickle
import re

# Model Path...
model_path = r'C:\Users\rcjad\Desktop\Desktop Files\NLP\CodeSoft ML Internship\Spam SMS Detection\Model\MultinomialNB.pkl'
tfidf_path = r'C:\Users\rcjad\Desktop\Desktop Files\NLP\CodeSoft ML Internship\Spam SMS Detection\Model\Spam_SMS_Tfidf.pkl'

# Model import...
model = pickle.load(open(model_path, 'rb'))
tfidf = pickle.load(open(tfidf_path, 'rb'))

# Helper Function...
def clean_text(text):
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    # Remove mentions (@username)
    text = re.sub(r"@\w+", '', text)
    # Remove hashtags but keep the text
    text = re.sub(r"#(\w+)", r'\1', text)
    # Remove special characters and numbers
    text = re.sub(r"[^A-Za-z\s]", '', text)
    # Remove extra spaces
    text = re.sub(r"\s+", ' ', text).strip()
    return text.lower()

# App Creation...
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/pred', methods=['POST'])
def prediction():
    if request.method == 'POST':
        try:
            Raw_test = request.form['SMS-DESC']
            Cleaned_Text = clean_text(Raw_test)
            vectorize_text = tfidf.transform([Cleaned_Text])
            prediction = model.predict(vectorize_text)
            prediction = str(prediction[0])
            return render_template('index.html', Pred=prediction)
        except Exception as e:
            return f"Error : {e}"
        
    else:
        print('Error.....')
# Main Function...

if __name__ == "__main__":
    app.run(port=2000,debug=True)
