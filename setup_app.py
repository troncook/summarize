import os

# Create necessary directories
os.makedirs('uploads', exist_ok=True)
os.makedirs('templates', exist_ok=True)

# Create app.py
with open('app.py', 'w') as f:
    f.write("""\
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os
from transformers import pipeline
import docx
import fitz  # PyMuPDF

app = Flask(__name__)

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
    text = ""
    for page in doc:
        text += page.getText()
    return text

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\\n'.join(full_text)

def summarize_text(text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    max_length = len(text.split()) // 4
    min_length = max(30, max_length // 2)
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

@app.route('/')
def upload_file():
    return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_file_post():
    # Check if the post request has the file part
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

        # Extract text based on file extension
        file_extension = filename.rsplit('.', 1)[1].lower()
        if file_extension == 'txt':
            text = extract_text_from_txt(file_path)
        elif file_extension == 'pdf':
            text = extract_text_from_pdf(file_path)
        elif file_extension == 'docx':
            text = extract_text_from_docx(file_path)

        summary = summarize_text(text)
        os.remove(file_path)  # Remove the uploaded file after processing

        return render_template('result.html', summary=summary)

if __name__ == '__main__':
    app.run()
""")

# Create requirements.txt
with open('requirements.txt', 'w') as f:
    f.write("""\
Flask
werkzeug
python-docx
PyMuPDF
transformers
""")

# Create upload.html
with open('templates/upload.html', 'w') as f:
    f.write("""\
<!doctype html>
<title>File Upload</title>
<h1>Upload a File</h1>
<form method=post enctype=multipart/form-data>
  <input type=file name=file>
  <input type=submit value=Upload>
</form>
""")

# Create result.html
with open('templates/result.html', 'w') as f:
    f.write("""\
<!doctype html>
<title>Summary Result</title>
<h1>Summary</h1>
<p>{{ summary }}</p>
""")
    
print("Project setup completed.")
