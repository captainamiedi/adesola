from flask import Flask, jsonify, request, send_from_directory
import PyPDF2
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
import openai
from pydub import AudioSegment
import filetype
import os
import tempfile
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chains import RetrievalQAWithSourcesChain
from dotenv import dotenv_values
from flask_cors import CORS
import json
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from supabase import create_client, Client
import QAEmbedding


config = dotenv_values(".env") 
# print(config)
vectorDb = {}
UPLOAD_FOLDER = 'C:/Users/HP/lawEmbedding2/upload'
app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.environ["OPENAI_API_KEY"] = config['OPENAI_API_KEYS']

ALLOWED_EXTENSION = set(['pdf'])
ALLOWED_MEDIA_EXTENSION = set(['mp3', 'mp4', 'mpeg', 'mpga', 'm4a', 'wav', 'webm'])
llm = OpenAI(openai_api_key=config['OPENAI_API_KEYS'],temperature=0)
text_splitter = CharacterTextSplitter()
openai.api_key = config['OPENAI_API_KEYS']
supabase: Client = create_client(config['SUPABASE_PROJECT_URL'], config['SUPABASE_API_KEY'])
embeddingDoc = QAEmbedding.User()

# Load the Whisper model:
# model = whisper.load_model('base')

def authenticate():
    try:

        # Retrieve the access token from the request headers
        access_token = request.headers.get('Authorization')
        
        data = supabase.auth.get_user(access_token)
        request.user_id = data.user.id
        # Check if the access token is valid and corresponds to an authenticated user
        # Example: verify the access token against a database or JWT token

        if access_token is None or not data:
            # Return an error response if authentication fails
            return jsonify({'error': 'Unauthorized access'}), 401
    except Exception as exc:
        return jsonify({'error': exc})
    
    # request.user_id = data.user.id

def allowed_extension(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSION
def allowed_media_extension(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_MEDIA_EXTENSION

def download_file(name):
    return send_from_directory(app.config["UPLOAD_FOLDER"], name)

@app.route('/hello/', methods=['GET', 'POST'])
def welcome():
    return jsonify({'name': 'bright', 'address': 'here'})

@app.before_request
def before_request():
    if request.path.startswith('/api'):
        authenticate()
        
@app.route('/api/upload_file/', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        resp = jsonify({'message': 'No file found'})
        resp.status_code = 400
        return resp
    file = request.files['file']
    if file.filename == '':
        resp = jsonify({'message': 'No file selected'})
        resp.status_code = 400
        return resp
    if file and allowed_extension(file.filename):
        try:
            reader = PyPDF2.PdfReader(file)
            number_of_pages = len(reader.pages)
            # Extract text from all the pages
            text = ''
            for page_num in range(number_of_pages):
                page = reader.pages[page_num]
                text += page.extract_text()

            texts = text_splitter.split_text(text)
            # print(texts)
            docs = [Document(page_content=t) for t in texts]
            # print(docs)
            chain = load_summarize_chain(llm, chain_type="refine", return_intermediate_steps=True)
            result = chain({"input_documents": docs}, return_only_outputs=True)
            print(result)
            resp = jsonify({'message': 'Upload Successful', 'data': result})
            resp.status_code = 200
            return resp
        except Exception as exc:
            print(exc)
            return jsonify({'error': 'Failed to read the PDF file'})
            
    else:
        resp = jsonify({'message': 'Allowed file types are pfd, doc'})
        resp.status_code = 400
        return resp

@app.route('/api/upload_media/', methods=['POST'])
def upload_media():
    if 'file' not in request.files:
        resp = jsonify({'message': 'No file found'})
        resp.status_code = 400
        return resp
    file = request.files['file']
    if file.filename == '':
        resp = jsonify({'message': 'No file selected'})
        resp.status_code = 400
        return resp
    if file and allowed_media_extension(file.filename):
        try:
            kind = filetype.guess(file)
            if kind.extension == 'mp3':
                temp_audio = tempfile.NamedTemporaryFile(delete=False)
                file.save(temp_audio.name)
                temp_audio.close()
                audio = AudioSegment.from_file(temp_audio.name)
                temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                audio.export(temp_wav.name, format='wav')
                temp_wav.close()
                transcript = openai.Audio.transcribe('whisper-1', open(temp_wav.name, 'rb'))
                print(transcript)
                os.remove(temp_audio.name)
                os.remove(temp_wav.name)
                return jsonify({'data': transcript['text']})
            return jsonify({'extension': kind.extension, 'type': kind.mime})
        except Exception as exc:
            print('exception: %s' % exc)

@app.route('/api/question_answer/', methods=['POST'])
def question_answer():
    # print(request.files, 'init')
    if 'file' not in request.files:
        resp = jsonify({'message': 'No file found'})
        resp.status_code = 400
        return resp
    file = request.files['file']
    # print(file, 'upload file')
    if file.filename == '':
        resp = jsonify({'message': 'No file selected'})
        resp.status_code = 400
        return resp
    if file and allowed_extension(file.filename):
        try:
            reader = PyPDF2.PdfReader(file)
            number_of_pages = len(reader.pages)
            text = ''
            for page_num in range(number_of_pages):
                page = reader.pages[page_num]
                text += page.extract_text()
            
            texts = text_splitter.split_text(text)
            text_splitter1 = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            texts1 = text_splitter1.split_text(text)
            # docs = [Document(page_content=t) for t in texts1]
            # print(docs, 'docs')
            # vectorDb['userText'] = docs
            # print(texts1, 'text')
            embeddings = OpenAIEmbeddings(openai_api_key=config['OPENAI_API_KEYS'])
            docsearch = Chroma.from_texts(texts1, embeddings, metadatas=[{"source": str(i)} for i in range(len(texts1))])
            print(docsearch, 'search')
            if request.form['type'] == 'question':
            # responseText = ''
                for text in texts:
                    prompt = text + '\nQUESTIONS:'
                    # print(prompt)
                    response = openai.Completion.create(
                        engine='text-davinci-003',
                        prompt=prompt,
                        temperature=0.3,
                        max_tokens=700,
                        # n=,
                        stop=None,
                        logit_bias={'101': 10.0} 
                    )
                return jsonify({'message': 'Generation Completed', 'data': response['choices'][0]['text']})
            chain = RetrievalQAWithSourcesChain.from_chain_type(OpenAI(temperature=0), chain_type="stuff", retriever=docsearch.as_retriever())
            # data, count = supabase
            # .table('countries')
            # .insert({"id": 1, "name": "Denmark"})
            # .execute()
            # data, count = supabase.table('QuestionAndAnswerChain').insert({"content": vars(chain), 'userId': request.user_id, "document_name": file.name}).execute()
            # print(chain)
            # print(vars(chain), 'compare')
            # print(type(chain))
            result = chain({"question": request.form['question']}, return_only_outputs=True)
            merged_data = {
                "merged": result["answer"] + " Sources: " + result["sources"]
            }
            return jsonify({'data': merged_data})
        except Exception as exc:
            print('exception', exc)

@app.route('/api/answer/', methods=['POST'])
def answer():
    try:
        user_id = request.user_id
        response = supabase.table('QuestionAndAnswerChain').select('*').eq('userId', user_id).execute()
        print(response, 'user id')
        data = response.get('data')
        # Process the data as needed
        print(data)
        result = vectorDb['userText']({"question": request.form['question']}, return_only_outputs=True)
        merged_data = {
            "merged": result["answer"] + " Sources: " + result["sources"]
        }
        return jsonify({'data': merged_data})
    except Exception as exc:
        print(exc)
        return jsonify({'error': exc})
    
@app.route('/ap/embedding/doc', methods=['POST'])
def embeddingDocUpload():
    try:
        files = request.files['file']
        user_id = request.user_id
        QAEmbedding.create_user_api(user_id)
        resp = QAEmbedding.upload_and_embed_documents(user_id, files)
        return jsonify({'data': resp})
    except Exception as exc:
        print(exc)

@app.route('/ap/query/doc', methods=['POST'])
def queryEmbeddingDoc():
    try:
        question = request.form['question']
        user_id = request.user_id
        resp = QAEmbedding.query_chain(user_id, question)
        return jsonify({'data': resp})
    except Exception as exc:
        print(exc)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)


# TODO
# Prevent second embedding if first has been done
# store doc search to supabase, set a field in supabase to embedding done.
# determine the type of data in doc search