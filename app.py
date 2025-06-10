from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
import os
from dotenv import load_dotenv

from utils import ask_question, process_all_courses

load_dotenv()

app = Flask(__name__)
app.config['PDFS_DIR'] = 'pdfs'

# Processar todos os cursos ao iniciar
GLOBAL_INDEX = process_all_courses(app.config['PDFS_DIR'])

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message']

    try:
        response = ask_question(
            question=user_message,
            global_index=GLOBAL_INDEX
        )

        return jsonify({"response": response})
    except Exception as e:
        print(f"Erro: {e}")
        return jsonify({'response': "Desculpe, estou com dificuldades. Por favor, tente novamente."})



if __name__ == '__main__':
    app.run(debug=True)