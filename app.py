from flask import Flask, request, render_template
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from googletrans import Translator

app = Flask(__name__)

# Load the dataset
df = pd.read_csv("Language Detection.csv")

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Clean text function
def clean_function(Text):
    Text = re.sub(r'[\([{})\]!@#$,"%^*?:;~`0-9]', ' ', Text)
    Text = Text.lower()
    Text = re.sub('http\S+\s*', ' ', Text)
    Text = re.sub('RT|cc', ' ', Text)
    Text = re.sub('#\S+', '', Text)
    Text = re.sub('@\S+', '  ', Text)
    Text = re.sub('\s+', ' ', Text)
    return Text

# Apply text cleaning
df['cleaned_Text'] = df['Text'].apply(clean_function)

# Feature selection
X = df["cleaned_Text"]
y = df["Language"]

# Label Encoding
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Bag of words
CV = CountVectorizer()
X = CV.fit_transform(X).toarray()

# Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = MultinomialNB()
model.fit(X_train, y_train)

# Prediction function
def predict_language(text):
    x = CV.transform([clean_function(text)]).toarray()
    lang = model.predict(x)
    lang = encoder.inverse_transform(lang)
    return lang[0]

# Translation function using googletrans
def translate(text, target_languages):
    translations = {}
    translator = Translator()
    for lang in target_languages:
        translated_text = translator.translate(text, dest=lang).text
        translations[lang] = translated_text
    return translations

# Route to render index.html
@app.route('/')
def index():
    return render_template('index.html')

# Route for language prediction
@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    target_languages = request.form.getlist('target_languages')
    
    prediction = predict_language(text)
    
    translations = {}
    if target_languages:
        translations = translate(text, target_languages)
    
    return render_template('index.html', language=prediction, text=text, translations=translations, selected_languages=target_languages)

if __name__ == '__main__':
    app.run(debug=True)
