from datetime import datetime
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
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from wordcloud import WordCloud

app = Flask(__name__, template_folder='template')
app.static_folder = 'static'

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
                    'akulah', 'amat', 'amatlah', 'anda', 'andalah', 'antar', 'antara', 'antaranya', 'apa', 'apaan',
                    'apabila', 'apakah', 'apatah', 'artinya', 'asal', 'asalkan', 'atas', 'atau', 'ataukah', 'ataupun',
                    'awal', 'awalnya', 'bagai', 'bagaikan', 'bagaimana', 'bagaimanakah', 'bagaimanapun', 'bagi', 'bagian',
                    'bahkan', 'bahwa', 'bahwasanya', 'baik', 'bakal', 'bakalan', 'balik', 'banyak', 'bapak', 'baru', 'bawah',
                    'beberapa', 'begini', 'beginian', 'beginikah', 'beginilah', 'begitu', 'begitukah', 'begitulah', 'begitupun',
                    'bekerja', 'belakang', 'belakangan', 'belumlah', 'benar', 'benarkah', 'benarlah', 'berada', 'berakhir',
                    'berakhirlah', 'berakhirnya', 'berapa', 'berapakah', 'berapalah', 'berapapun', 'berarti', 'berawal', 'berbagai',
                    'berdatangan', 'beri', 'berikan', 'berikut', 'berikutnya', 'berjumlah', 'berkali-kali', 'berkata', 'berkehendak',
                    'berkeinginan', 'berkenaan', 'berlainan', 'berlalu', 'berlangsung', 'berlebihan', 'bermacam', 'bermacam-macam',
                    'bermaksud', 'bermula', 'bersama', 'bersama-sama', 'bersiap', 'bersiap-siap', 'bertanya', 'bertanya-tanya',
                    'berturut', 'berturut-turut', 'bertutur', 'berujar', 'berupa', 'besar', 'betulkah', 'bila', 'bilakah', 'bisa',
                    'bisakah', 'bolehkah', 'buat', 'bukan', 'bukankah', 'bukanlah', 'bukannya', 'bulan', 'bung', 'cukupkah', 'dahulu',
                    'dalam', 'dan', 'dapat', 'dari', 'daripada', 'datang', 'dekat', 'demi', 'demikian', 'demikianlah', 'dengan',
                    'depan', 'di', 'dia', 'diakhiri', 'diakhirinya', 'dialah', 'diantara', 'diantaranya', 'diberi', 'diberikan',
                    'diberikannya', 'dibuat', 'dibuatnya', 'didapat', 'didatangkan', 'diibaratkan', 'diibaratkannya', 'diingat',
                    'diingatkan', 'diinginkan', 'dijawab', 'dijelaskan', 'dijelaskannya', 'dikarenakan', 'dikatakan', 'dikatakannya',
                    'dikerjakan', 'diketahui','diketahuinya','dilakukan','dilalui','dilihat','dimaksud','dimaksudkan','dimaksudkannya',
                    'dimaksudnya','diminta','dimintai','dimisalkan','dimulainya','dimungkinkan','dini','dipastikan','diperbuat',
                    'diperbuatnya', 'dipergunakan','diperkirakan','diperlihatkan','diperlukan','diperlukannya','dipersoalkan','dipunyai',
                    'diri','dirinya','disampaikan','disebut','disebutkan','disebutkannya','disini','disinilah','ditambahkan',
                    'ditandaskan','ditanya','ditanyai','ditanyakan','ditegaskan','ditujukan','ditunjuk','ditunjuki','ditunjukkan',
                    'ditunjukkannya','ditunjuknya','dituturkan', 'dituturkannya', 'diucapkan', 'diucapkannya', 'diungkapkan',
                    'dong', 'dulu', 'empat', 'gunakan', 'hal', 'hanya', 'hanyalah', 'hari', 'haruslah', 'hendak', 'hendaklah',
                    'hendaknya', 'hingga', 'ia', 'ialah', 'ibarat', 'ibaratkan', 'ibaratnya', 'ibu', 'ikut', 'ingat', 'ingat-ingat',
                    'ingin', 'inginkah', 'inginkan', 'ini', 'inikah', 'inilah', 'itu', 'itukah', 'itulah', 'jadi', 'jadilah',
                    'jadinya', 'jangan', 'jangankan', 'janganlah', 'jauh', 'jawab', 'jawaban', 'jawabnya', 'jelaskan', 'jelasnya',
                    'jika', 'jikalau', 'juga', 'jumlah', 'jumlahnya', 'kala', 'kalaulah', 'kalaupun', 'kalian', 'kami', 'kamilah',
                    'kamu', 'kamulah', 'kan', 'kapankah', 'kapanpun', 'karenanya', 'kasus', 'kata', 'katakan', 'katakanlah', 'katanya',
                    'ke', 'keadaan', 'kebetulan', 'kecil', 'kedua', 'keduanya', 'keinginan', 'kelima', 'keluar', 'kembali', 'kemudian',
                    'kepada', 'kepadanya', 'kesampaian', 'keseluruhan', 'keseluruhannya', 'keterlaluan', 'ketika', 'khususnya',
                    'kini', 'kinilah', 'kiranya', 'kitalah', 'kok', 'lah', 'lalu', 'lanjut', 'lanjutnya', 'lebih', 'lewat', 'lima',
                    'luar', 'macam', 'maka', 'makanya', 'mampukah', 'mana', 'manakala', 'manalagi', 'masa', 'masihkah', 'masing',
                    'masing-masing', 'maupun', 'melainkan', 'melakukan', 'melalui', 'melihat', 'melihatnya', 'memang', 'memastikan',
                    'memberi', 'memberikan', 'membuat', 'memerlukan', 'memihak', 'meminta', 'memintakan', 'memisalkan', 'memperbuat',
                    'mempergunakan', 'memperkirakan', 'memperlihatkan', 'mempersiapkan', 'mempersoalkan', 'mempertanyakan', 'mempunyai',
                    'memulai', 'memungkinkan', 'menaiki', 'menandaskan', 'menantikan', 'menanya', 'menanyai', 'menanyakan', 'mendapat',
                    'mendatang', 'mendatangi', 'mendatangkan', 'menegaskan', 'mengatakan', 'mengatakannya', 'mengenai', 'mengerjakan',
                    'mengetahui', 'menghendaki', 'mengibaratkan', 'mengibaratkannya', 'mengingat', 'mengingatkan', 'menginginkan',
                    'mengira', 'mengucapkan', 'mengucapkannya', 'mengungkapkan', 'menjawab', 'menjelaskan', 'menuju', 'menunjuk',
                    'menunjuki', 'menunjukkan', 'menunjuknya', 'menurut', 'menuturkan', 'menyampaikan', 'menyangkut', 'menyatakan',
                    'menyebutkan', 'menyeluruh', 'menyiapkan', 'mereka', 'merekalah', 'merupakan', 'meyakini', 'meyakinkan', 'minta',
                    'mirip', 'misal', 'misalkan', 'misalnya', 'mula', 'mulai', 'mulailah', 'mungkin', 'mungkinkah', 'nah', 'naik',
                    'nyaris', 'nyatanya', 'oleh', 'olehnya', 'pada', 'padanya', 'pak', 'paling', 'panjang', 'pantas', 'para', 'pasti',
                    'pastilah', 'penting', 'pentingnya', 'per', 'perlukah', 'perlunya', 'persoalan', 'pertama', 'pertanyaan',
                    'pertanyakan', 'pihak', 'pihaknya', 'pukul', 'pula', 'pun', 'punya', 'saatnya', 'saling', 'sama-sama', 'sambil',
                    'sampai', 'sampai-sampai', 'sampaikan', 'sana', 'saya', 'sayalah', 'se', 'sebab', 'sebabnya', 'sebagai',
                    'sebagaimana', 'sebagainya', 'sebagian', 'sebaik', 'sebaik-baiknya', 'sebaliknya', 'sebanyak', 'sebegini',
                    'sebegitu', 'sebelum', 'sebenarnya', 'seberapa', 'sebesar', 'sebisanya', 'sebuah', 'sebut', 'sebutlah',
                    'sebutnya', 'secara', 'sedemikian', 'seenaknya', 'segala', 'segalanya', 'seingat', 'sejauh', 'sejenak',
                    'sejumlah', 'sekadar', 'sekadarnya', 'sekali', 'sekalian', 'sekecil', 'seketika', 'sekiranya', 'sekitarnya',
                    'sekurang-kurangnya', 'sekurangnya', 'sela', 'selain', 'selaku', 'selanjutnya', 'seluruhnya', 'semacam', 'semampu',
                    'semampunya', 'semasa', 'semasih', 'semata', 'semata-mata', 'semaunya', 'sempat', 'sendirian', 'sendirinya', 'seolah',
                    'seolah-olah', 'seorang', 'sepanjang', 'sepantasnya', 'sepantasnyalah', 'seperlunya', 'seperti', 'sepertinya',
                    'sepihak', 'seringnya', 'serta', 'serupa', 'sesaat', 'sesama', 'sesampai', 'sesegera', 'seseorang', 'sesuatu',
                    'sesuatunya', 'sesudah', 'sesudahnya', 'setelah', 'setempat', 'setengah', 'seterusnya', 'setiap', 'setiba',
                    'setibanya', 'setidak-tidaknya', 'setidaknya', 'sela', 'selain', 'selaku', 'selanjutnya', 'seluruhnya', 'semacam',
                    'semampu', 'semampunya', 'semasa', 'semasih', 'semata', 'semata-mata', 'semaunya', 'sempat', 'sendirian',
                    'sendirinya', 'seolah', 'seolah-olah', 'seorang', 'sepanjang', 'sepantasnya', 'sepantasnyalah', 'seperlunya',
                    'seperti', 'sepertinya', 'sepihak', 'seringnya', 'serta', 'serupa', 'sesaat', 'sesama', 'sesampai', 'sesegera',
                    'seseorang', 'sesuatu', 'sesuatunya', 'sesudah', 'sesudahnya', 'setelah', 'setempat', 'setengah', 'seterusnya',
                    'setiap', 'setiba', 'setibanya', 'setidak-tidaknya', 'setidaknya', 'sila', 'silakan', 'sini', 'sinilah', 'soal',
                    'suatu', 'sudah', 'sudahkah', 'sudahlah', 'supaya', 'tadi', 'tadinya', 'tahu', 'tahun', 'tak', 'tambahnya',
                    'tampaknya', 'tandas', 'tandasnya', 'tanya', 'tanyakan', 'tanyanya', 'tegas', 'tegasnya', 'telah', 'tempat',
                    'tengah', 'tentang', 'tentu', 'tentulah', 'tentunya', 'tepat', 'terakhir', 'terasa', 'terbanyak', 'terdahulu',
                    'terdapat', 'terdiri', 'terhadap', 'terhadapnya', 'teringat', 'teringat-ingat', 'terjadilah', 'terjadinya',
                    'terkira', 'tersampaikan', 'tersebut', 'tersebutlah', 'tertentu', 'tertuju', 'tidakkah', 'tidaklah', 'tiga',
                    'tinggi', 'toh', 'tunjuk', 'turut', 'tutur', 'tuturnya', 'ucap', 'ucapnya', 'ujar', 'ujarnya', 'ungkap', 'ungkapnya',
                    'usai', 'waduh', 'wah', 'wahai', 'waktunya', 'walau', 'wong', 'yaitu', 'yakni', 'bener', 'beneran', 'pertama', 'SMA',
                    'aku', 'ini', 'buat', 'yang', 'dan', 'berbagai', 'setelah', 'guys', '10% niacinamide', '5% niacinamide', 'aja',
                    'pemakaian', 'kepo', 'penasaran', 'alhamdulillah', '1', '2', '3', 'kedua', 'ketiga', 'deh', 'besok', 'gue',
                    'produk', 'skincare', 'konsentrasi', 'member', 'adanya', 'karena', 'fd', 'female daily', 'somethinc', 'official',
                    'store', 'e-commerce', 'shopee', 'malam', 'niacin', 'mix', 'feeling', 'SMP', 'anak', 'to', 'jadiin', 'pol', 'vit',
                    'c', 'vitamin', 'pagi', 'paginya', 'online', 'gitu','banget','bgt','gak','tp','nya']

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


# Check if column exists in the table
cursor.execute("SHOW COLUMNS FROM stemmed_data LIKE 'sentimen'")
column_exists = cursor.fetchone()

if column_exists:
    print("Column 'sentimen' already exists. Skipping data insertion.")
else:
    # Alter the table and modify the data type of 'sentimen' column
    cursor.execute("ALTER TABLE stemmed_data MODIFY sentimen VARCHAR(255)")
    conn.commit()

    # Insert stemmed data into the database
    # for text, sentiment in zip(dataset['text'], dataset['sentimen']):
    #     stemmed_text = stemming(text)  # Convert list to string
    #     insert_query = "INSERT INTO training_new (text, sentimen) VALUES (%s, %s)"
    #     values = (text.lower().encode('utf-8'), prediction.encode('utf-8'))
    #     cursor.execute(insert_query, values)
    #     conn.commit()

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
        X_testing = dataset['text']
        y_testing= dataset['sentimen']
        X_train, X_test, y_train, y_test = train_test_split(X_testing, y_testing, test_size=0.2, random_state=42)

        # Feature Extraction menggunakan Bag of Words
        X_train = vectorizer.fit_transform(X_train.apply(' '.join))
        X_test = vectorizer.transform(X_test.apply(' '.join))

        nb_classifier.fit(X_train, y_train)
        y_pred = nb_classifier.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)

        # Visualisasi dengan Pie Chart (Testing Data)
        labels_train = ['Negatif', 'Positif', 'Netral']
        counts_train = y_train.value_counts()
        sizes_train = [counts_train['Negatif'], counts_train['Positif'], counts_train['Netral']]
        colors_train = ['red', 'green', 'blue']
        explode_train = (0.1, 0.1, 0.1)  # Memisahkan slice untuk sentimen negatif

        fig_train, ax_train = plt.subplots()
        ax_train.pie(sizes_train, explode=explode_train, labels=labels_train, colors=colors_train, autopct='%1.1f%%', startangle=90)
        ax_train.axis('equal')  # Memastikan pie chart berbentuk lingkaran
        ax_train.set_title('Distribusi Sentimen Data Training')
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        
        # Save the pie chart image with the timestamp
        file_name = f'static/pie_chart_{timestamp}.png'
        plt.savefig(file_name)


         # Generate word cloud
        word_cloud_text = ' '.join(X_testing.astype(str))
        if word_cloud_text:
            word_cloud = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(word_cloud_text)
            word_cloud_filename = f'static/word_cloud_{timestamp}.png'
            word_cloud.to_file(word_cloud_filename)
        else:
            word_cloud_filename = None
        # Render the template with the result and timestamp
        return render_template('result.html', result=result, accuracy=accuracy, word_cloud=word_cloud_filename,stemmed_text=stemmed_text, timestamp=timestamp)
    else:
        return render_template('index.html')


if __name__=="__main__":
    app.run(debug=True)
