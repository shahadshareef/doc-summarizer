from flask import Flask, request, render_template, Response
import os
import pdfplumber
import nltk
from nltk.tokenize import sent_tokenize
from transformers import pipeline
import torch
import re

# Ensure punkt is downloaded
nltk.download('punkt_tab')

# Device setup (GPU if available)
device = 0 if torch.cuda.is_available() else -1

# Use a public summarization model
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    device=device
)

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# --- Text extraction ---
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                page_text = re.sub(r'\n+', ' ', page_text)  # Clean newlines
                text += page_text + " "
    return text.strip()

def extract_text_from_txt(txt_path):
    try:
        with open(txt_path, "r", encoding="utf-8") as file:
            text = file.read()
        return text.strip()
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

# --- Chunking for long docs ---
def split_into_chunks(text, max_words=500):
    sentences = sent_tokenize(text)
    chunks, current_chunk, word_count = [], [], 0

    for sentence in sentences:
        sentence_words = len(sentence.split())
        if word_count + sentence_words > max_words:
            chunks.append(" ".join(current_chunk))
            current_chunk, word_count = [], 0
        current_chunk.append(sentence)
        word_count += sentence_words

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# --- Summarization ---
def summarize_long_doc(text):
    chunks = split_into_chunks(text)
    summary_chunks = []

    for chunk in chunks:
        try:
            summary = summarizer(
                chunk, max_length=150, min_length=50, do_sample=False
            )[0]["summary_text"]
            summary_chunks.append(summary)
        except Exception as e:
            print(f"Summarization error: {e}")
            continue

    return " ".join(summary_chunks)

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        file = request.files.get("document")
        if not file or not file.filename:
            return render_template("upload.html", error="No file selected.")

        file_ext = file.filename.rsplit(".", 1)[-1].lower()
        if file_ext not in ["pdf", "txt"]:
            return render_template("upload.html", error="Only PDF and TXT files are allowed.")

        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)

        text = extract_text_from_pdf(file_path) if file_ext == "pdf" else extract_text_from_txt(file_path)
        if not text:
            return render_template("upload.html", error="No text found in the file.")

        chunks = split_into_chunks(text)
        total_chunks = len(chunks)
        summary_chunks = []

        def generate():
            for i, chunk in enumerate(chunks):
                try:
                    summary = summarizer(chunk, max_length=150, min_length=50, do_sample=False)[0]["summary_text"]
                    summary_chunks.append(summary)
                except Exception as e:
                    print(f"Chunk {i} summarization error: {e}")
                    summary_chunks.append("[Error in chunk]")
                progress = int(((i + 1) / total_chunks) * 100)
                yield f"data: {progress}\n\n"

            # Send the final summary once all chunks are done
            full_summary = " ".join(summary_chunks)
            yield f"data: DONE\n\n"
            yield f"data: {full_summary}\n\n"

        return Response(generate(), mimetype="text/event-stream")

    return render_template("upload.html")

if __name__ == "__main__":
    app.run(debug=True)

# # --- Flask routes ---
# @app.route("/", methods=["GET", "POST"])
# def upload_file():
#     if request.method == "POST":
#         file = request.files.get("document")
#         if not file or not file.filename:
#             return render_template("upload.html", error="No file selected.")

#         file_ext = file.filename.rsplit(".", 1)[-1].lower()
#         if file_ext not in ["pdf", "txt"]:
#             return render_template("upload.html", error="Only PDF and TXT files are allowed.")

#         file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
#         file.save(file_path)

#         text = extract_text_from_pdf(file_path) if file_ext == "pdf" else extract_text_from_txt(file_path)

#         if not text:
#             return render_template("upload.html", error="No text found in the file.")

#         summary = summarize_long_doc(text)
#         return render_template("result.html", summary=summary)

#     return render_template("upload.html")

