from flask import Flask, request, render_template, flash, redirect
from werkzeug.utils import secure_filename
import os
from transformers import pipeline
import docx
import fitz  # PyMuPDF
from concurrent.futures import ThreadPoolExecutor
import torch
from wordcloud import WordCloud
from nltk import bigrams, trigrams, FreqDist
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import coo_matrix
import numpy as np
from flask_socketio import SocketIO
import os
import secrets

app = Flask(__name__)  # Create the app object here
socketio = SocketIO(app)

app.config['SECRET_KEY'] = secrets.token_hex(16)
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', secrets.token_hex(16))

nltk.download('punkt')

# Initialize the summarizer pipeline with the specified model
summarizer = pipeline('summarization', model='google/pegasus-xsum', device=0)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_txt(file_path):
    with open(file_path, 'r') as f:
        return f.read()

def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = []
    for page in doc:
        text.append(page.get_text())
    return "".join(text)

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

def summarize_text(text):
    input_length = len(text.split())
    max_length = min(input_length // 10, 150)
    min_length = max(max_length // 2, 20)
    if input_length < 10:
        max_length = input_length
        min_length = input_length // 2
    chunk_size = 2000   

    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    total_chunks = len(chunks)
    batch_size = 8
    summary = []

    def summarize_batch(batch):
        return summarizer(batch, max_length=max_length, min_length=min_length, do_sample=False)

    with ThreadPoolExecutor() as executor:
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i+batch_size]
            future = executor.submit(summarize_batch, batch)
            results = future.result()
            for result in results:
                summary.append(result['summary_text'])

            # Emit progress update
            progress = ((i+batch_size) / total_chunks) * 100
            socketio.emit('progress', {'progress': progress})

    summarized_text = " ".join(summary)
    summarized_length = len(summarized_text.split())

    # Emit 100% completion before returning
    socketio.emit('progress', {'progress': 100})
    
    return summarized_text, input_length, summarized_length

def process_text_for_analysis(text):
    # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove stop words
    text = ' '.join([word for word in text.split() if word.lower() not in ENGLISH_STOP_WORDS])
    
    return text

def generate_word_cloud(text):
    os.makedirs('static', exist_ok=True)  # Ensure the directory exists
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.savefig('static/word_cloud.png')

def generate_unique_word_cloud(text):
    os.makedirs('static', exist_ok=True)  # Ensure the directory exists
    unique_words = set(text.split())
    unique_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(unique_words))
    plt.figure()
    plt.imshow(unique_wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.savefig('static/unique_word_cloud.png')

def calculate_lexical_diversity(text):
    total_words = len(text.split())
    unique_words = len(set(text.split()))
    return unique_words / total_words

def calculate_bigram_frequency(text):
    bigram_freq = FreqDist(bigrams(text.split()))
    return dict(bigram_freq.most_common(20))

def calculate_trigram_frequency(text):
    trigram_freq = FreqDist(trigrams(text.split()))
    return dict(trigram_freq.most_common(20))

def calculate_collocations(text):
    bigram_collocation = BigramCollocationFinder.from_words(text.split())
    trigram_collocation = TrigramCollocationFinder.from_words(text.split())

    return bigram_collocation.ngram_fd.most_common(30), trigram_collocation.ngram_fd.most_common(30)


@app.route('/')
def upload_file():
    return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_file_post():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            file_extension = filename.rsplit('.', 1)[1].lower()
            if file_extension == 'txt':
                text = extract_text_from_txt(file_path)
            elif file_extension == 'pdf':
                text = extract_text_from_pdf(file_path)
            elif file_extension == 'docx':
                text = extract_text_from_docx(file_path)
            
            summary, original_word_count, summary_word_count = summarize_text(text)
            
            # Process the text after summarization
            processed_text = process_text_for_analysis(summary)

            generate_word_cloud(processed_text)
            generate_unique_word_cloud(processed_text)
            lexical_diversity = calculate_lexical_diversity(processed_text)
            bigram_frequency = calculate_bigram_frequency(processed_text)
            trigram_frequency = calculate_trigram_frequency(processed_text)
            bigram_collocation, trigram_collocation = calculate_collocations(processed_text)
        
        finally:
            os.remove(file_path)  # Clean up the uploaded file

        return render_template(
            'result.html', 
            summary=summary, 
            original_word_count=original_word_count, 
            summary_word_count=summary_word_count, 
            lexical_diversity=lexical_diversity, 
            bigram_frequency=bigram_frequency, 
            trigram_frequency=trigram_frequency, 
            bigram_collocation=bigram_collocation, 
            trigram_collocation=trigram_collocation, 
        )

if __name__ == '__main__':
    socketio.run(app)





