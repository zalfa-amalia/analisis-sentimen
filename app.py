from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.metrics import confusion_matrix, accuracy_score

app = Flask(__name__)

# Load Dataset from Excel file
dataset = pd.read_excel('data_trainingnew.xlsx')

# Preprocessing: Case Folding
dataset['text'] = dataset['text'].str.lower()

# Preprocessing: Tokenizing
dataset['text'] = dataset['text'].apply(word_tokenize)

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
                    'c', 'vitamin', 'pagi', 'paginya', 'online']

def remove_stopwords(tokens):
    filtered_tokens = [word for word in tokens if word not in custom_stopwords]
    return filtered_tokens

dataset['text'] = dataset['text'].apply(remove_stopwords)

# Preprocessing: Stemming
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def stem_tokens(tokens):
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return stemmed_tokens

dataset['text'] = dataset['text'].apply(stem_tokens)

# Pembagian Dataset menjadi Data Training dan Testing
X = dataset['text']
y = dataset['sentiment']

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
    return render_template('AnalisisSentimen.html')

@app.route('/classify', methods=['POST'])
def classify():
    text = request.form['text']
    stemmed_text = [stemmer.stem(word) for word in word_tokenize(text.lower())]
    vectorized_text = vectorizer.transform([' '.join(stemmed_text)])
    prediction = nb_classifier.predict(vectorized_text)[0]
    return render_template('result.html', prediction=prediction)