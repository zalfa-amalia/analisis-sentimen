from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
from nltk.tokenize import word_tokenize
from sklearn.metrics import confusion_matrix, accuracy_score
from nltk.corpus import stopwords
import mysql.connector
import re

app = Flask(__name__, template_folder='template')

# Koneksi ke database MySQL
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="data_set"
)
cursor = conn.cursor()

# Preprocessing: Case Folding
cursor.execute("SELECT text, sentimen FROM training")
data = cursor.fetchall()
dataset = pd.DataFrame(data, columns=['text', 'sentimen'])
dataset['text'] = dataset['text'].str.lower()

def cleansing(text):
    text = text.strip(" ")
    text = re.sub(r'[?|$|.|!_:")(-+,]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r"\b[a-zA-Z]\b", "", text)
    text = re.sub('\s+',' ', text)
    return text
dataset['text'] = dataset['text'].apply(cleansing)

# Replace specific words
replacement_dict = {
    r'\b(ga)\b': 'gak',
    r'\b(ngga|nggak|engga|enggak)\b': 'gak',
    r'\b(gaada)\b': 'gak ada',
    r'\b(pdhl)\b': 'padahal',
    r'\b(lgsg)\b': 'langsung',
    r'\b(bgtt{1,3})\b': 'banget',
    r'\b(banget{1,10})\b': 'banget',
    r'\b(stlh)\b': 'setelah',
    r'\b(sukaa{1,10})\b': 'suka',
    r'\b(seneng{2,3})\b': 'seneng',
    r'\b(gila{2,10})\b': 'gila'
}

def replace_words(text):
    for pattern, replacement in replacement_dict.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text

dataset['text'] = dataset['text'].apply(replace_words)

#NLTK word tokenize
def word_tokenize_wrapper(text):
    return word_tokenize(text)
dataset['text'] = dataset['text'].apply(word_tokenize_wrapper)

# Preprocessing: Stopword Removal
custom_stopwords = ['adalah', 'adanya', 'adapun', 'agar', 'akan', 'akankah', 'akhir', 'akhiri', 'akhirnya', 'aku',
                    ]

def stopword_removal(Review):
    filtering = stopwords.words('indonesian','english')
    filtering.extend(custom_stopwords)
    x = []
    data = []
    def myFunc(x):
        if x in filtering:
            return False
        else:
            return True
    fit = filter(myFunc, Review)
    for x in fit:
        data.append(x)
    return data
dataset['text'] = dataset['text'].apply(stopword_removal)

def stemming(text):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    do = []
    for w in text:
        dt = stemmer.stem(w)
        do.append(dt)
    d_clean = " ".join(do)
    return d_clean


# Check if table exists in the database
cursor.execute("SHOW TABLES LIKE 'stemmed_data'")
table_exists = cursor.fetchone()

if table_exists:
    print("Table 'stemmed_data' already exists. Skipping data insertion.")
else:
    # Create a new table for stemmed data
    cursor.execute("CREATE TABLE stemmed_data (text VARCHAR(255), sentimen VARCHAR(255))")
    conn.commit()

    # Insert stemmed data into the database
    for text, sentiment in zip(dataset['text'], dataset['sentimen']):
        stemmed_text = stemming(text)  # Convert list to string
        insert_query = "INSERT INTO stemmed_data (text, sentimen) VALUES (%s, %s)"
        values = (stemmed_text, sentiment)
        cursor.execute(insert_query, values)
        conn.commit()



# Pembagian Dataset menjadi Data Training dan Testing
X = dataset['text']
y = dataset['sentimen']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Extraction menggunakan Bag of Words
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train.apply(' '.join))
X_test = vectorizer.transform(X_test.apply(' '.join))

# Pelatihan Model Naive Bayes
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

# Evaluasi Model
y_pred = nb_classifier.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['GET', 'POST'])
def classify():
    if request.method == 'POST':
        text = request.form['text']
        
        # Preprocessing: Case Folding
        text = text.lower()
        
        # Preprocessing: Cleansing
        text = cleansing(text)
        
        # Preprocessing: Replace specific words
        text = replace_words(text)
        
        # NLTK word tokenize
        text_tokens = word_tokenize_wrapper(text)
        
        # Preprocessing: Stopword Removal
        text_tokens = stopword_removal(text_tokens)
        
        # Preprocessing: Stemming
        stemmed_text = stemming(text_tokens)
        
        vectorized_text = vectorizer.transform([stemmed_text])
        prediction = nb_classifier.predict(vectorized_text)[0]
        
        if prediction == 'Negatif':
            result = 'Sentimen: Negatif'
        elif prediction == 'Positif':
            result = 'Sentimen: Positif'
        else:
            result = 'Sentimen: Netral'

        # Menambahkan data masukan ke dalam database
        insert_query = "INSERT INTO training_new (text, sentimen) VALUES (%s, %s)"
        values = (text.lower(), prediction)
        cursor.execute(insert_query, values)
        conn.commit()

        # Memperbarui dataset dengan data terbaru
        dataset.loc[len(dataset)] = [text.lower(), prediction]

        # Evaluasi Model Terbaru
        X = dataset['text']
        y = dataset['sentimen']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Feature Extraction menggunakan Bag of Words
        X_train = vectorizer.fit_transform(X_train.apply(' '.join))
        X_test = vectorizer.transform(X_test.apply(' '.join))
        
        nb_classifier.fit(X_train, y_train)
        y_pred = nb_classifier.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)

        return render_template('result.html', result=result, cm=cm, accuracy=accuracy, stemmed_text=stemmed_text)
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)
