from flask import Flask, request, render_template, redirect, url_for, session
import os
import glob
import openai
import yt_dlp as youtube_dl
from yt_dlp import DownloadError
import docarray
import shutil
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.embeddings import OpenAIEmbeddings

app = Flask(__name__)
app.secret_key = os.getenv("OPENAI_API_KEY")  # replace with your secret key

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        youtube_url = request.form.get('youtube_url')

        # Your code to process the YouTube video and generate the transcript...
        output_dir = "./audios"
        ffprobe_path = shutil.which("ffprobe")
        ffmpeg_path = shutil.which("ffmpeg")

        ydl_config = {
            "format": "bestaudio/best",
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }
            ],
            "outtmpl": os.path.join(output_dir, "%(title)s.%(ext)s"),
            "verbose": True,
            "ffmpeg_location": ffmpeg_path
        }

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        try:
            with youtube_dl.YoutubeDL(ydl_config) as ydl:
                ydl.download([youtube_url])
        except DownloadError:
            with youtube_dl.YoutubeDL(ydl_config) as ydl:
                ydl.download([youtube_url])

        audio_files = glob.glob(os.path.join(output_dir, "*.mp3"))
        audio_filename = audio_files[0]

        audio_file = audio_filename
        output_file = "./transcripts/output.txt"
        model = "whisper-1"

        with open(audio_file, "rb") as audio:
            response = openai.Audio.transcribe(model, audio)

        transcript = (response["text"])

        if output_file is not None:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "w") as file:
                file.write(transcript)

        session['transcript'] = transcript  # store the transcript in the session
        return redirect(url_for('query'))
    return render_template('index.html')

@app.route('/query', methods=['GET', 'POST'])
def query():
    if request.method == 'POST':
        query = request.form['query']

        # Your code to run the query on the transcript...
        loader = TextLoader("./transcripts/output.txt")
        docs = loader.load()

        db = DocArrayInMemorySearch.from_documents(docs, OpenAIEmbeddings())
        retriever = db.as_retriever()

        llm = ChatOpenAI(temperature=0.0)
        qa_stuff = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, verbose=True)

        response = qa_stuff.run(query)

        return render_template('query.html', response=response)
    return render_template('query.html')

if __name__ == '__main__':
    app.run(debug=True)