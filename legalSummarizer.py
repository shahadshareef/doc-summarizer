from flask import Flask, request, render_template, Response
import os
import pdfplumber
import nltk
from nltk.tokenize import sent_tokenize
from transformers import pipeline
import torch
import re

# --- Setup ---
nltk.download('punkt')
device = 0 if torch.cuda.is_available() else -1

summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    device=device
)

UPLOAD_FOLDER = "uploads"
SUMMARY_FOLDER = "./dataset/summary"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SUMMARY_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# --- Helper Functions ---
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                page_text = re.sub(r'\n+', ' ', page_text)
                text += page_text + " "
    return text.strip()

def extract_text_from_txt(txt_path):
    try:
        with open(txt_path, "r", encoding="utf-8") as file:
            return file.read().strip()
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def split_into_chunks(text, max_words=500):
    sentences = sent_tokenize(text)
    chunks, current_chunk, word_count = [], [], 0
    for sentence in sentences:
        words = len(sentence.split())
        if word_count + words > max_words:
            chunks.append(" ".join(current_chunk))
            current_chunk, word_count = [], 0
        current_chunk.append(sentence)
        word_count += words
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def summarize_document(text):
    """Summarizes text by chunking and using BART"""
    chunks = split_into_chunks(text)
    summary_chunks = []
    for chunk in chunks:
        try:
            summary = summarizer(chunk, max_length=150, min_length=50, do_sample=False)[0]["summary_text"]
            summary_chunks.append(summary)
        except Exception as e:
            print(f"Summarization error: {e}")
            summary_chunks.append("[Error in chunk]")
    return summary_chunks

# --- Flask Route ---
@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        file = request.files.get("document")
        if not file or not file.filename:
            return render_template("upload.html", error="No file selected.")

        ext = file.filename.rsplit(".", 1)[-1].lower()
        if ext not in ["pdf", "txt"]:
            return render_template("upload.html", error="Only PDF and TXT files are allowed.")

        # Save temporarily
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        # Extract text
        text = extract_text_from_pdf(filepath) if ext == "pdf" else extract_text_from_txt(filepath)
        if not text:
            return render_template("upload.html", error="No text found in the file.")

        # Summarize and save
        summary_save_path = os.path.join(SUMMARY_FOLDER, file.filename)
        summary_chunks = summarize_document(text)
        full_summary = " ".join(summary_chunks)
        with open(summary_save_path, "w", encoding="utf-8") as f:
            f.write(full_summary)
        print(f"✅ Summary saved to {summary_save_path}")

        # SSE Generator for progress
        def generate():
            total = len(summary_chunks)
            for i, chunk_summary in enumerate(summary_chunks):
                progress = int(((i + 1) / total) * 100)
                yield f"data: {progress}\n\n"
            yield f"data: DONE\n\n"
            yield f"data: {full_summary}\n\n"

        return Response(generate(), mimetype="text/event-stream")

    return render_template("upload.html")


if __name__ == "__main__":
    app.run(debug=True)
