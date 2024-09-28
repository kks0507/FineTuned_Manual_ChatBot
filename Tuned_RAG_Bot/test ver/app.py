from flask import Flask, request, jsonify, render_template
import PyPDF2
from sentence_transformers import SentenceTransformer
import chromadb
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

app = Flask(__name__)

# ChromaDB 클라이언트 설정
client = chromadb.Client()
collection = client.create_collection("security_manual")

# 파인튜닝된 모델 및 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("kingkim/kodialogpt_v3.0_SecurityManual")
model = AutoModelForCausalLM.from_pretrained("kingkim/kodialogpt_v3.0_SecurityManual")
gpt2_pipeline = pipeline('text-generation', model=model, tokenizer=tokenizer)

# SentenceTransformer 모델 로드 (문장 임베딩)
sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# 1. PDF 업로드 및 처리
def process_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    sentences = text.split('.')
    return sentences

# 2. 문장 임베딩 후 ChromaDB에 저장
def store_in_chroma(sentences):
    embeddings = sentence_model.encode(sentences)
    ids = [f"id_{i}" for i in range(len(sentences))]  # 각 문장에 대한 고유 ID 생성
    for i, sentence in enumerate(sentences):
        collection.add(ids=[ids[i]], embeddings=[embeddings[i]], metadatas={"sentence": sentence})

# 3. 질문 인코딩 및 유사 문장 찾기
def find_similar_sentence(question):
    question_embedding = sentence_model.encode([question])
    results = collection.query(query_embeddings=question_embedding, n_results=1)
    
    print(f"ChromaDB query 결과: {results}")  # 결과 구조 확인
    
    if "metadatas" in results and len(results["metadatas"]) > 0:
        # 첫 번째 결과의 첫 번째 리스트 아이템에 접근하고, 해당 딕셔너리의 "sentence" 값 반환
        return results["metadatas"][0][0].get("sentence", "유사한 문장을 찾을 수 없습니다.")
    else:
        return "유사한 문장을 찾을 수 없습니다."

# 4. GPT-2 모델로 답변 생성
# GPT-2 모델로 답변 생성
def generate_response(similar_sentence):
    generation_args = {
        "max_new_tokens": 80,        # 생성할 텍스트의 최대 길이
        "do_sample": True,           # 샘플링 사용
        "top_p": 0.9,                # 상위 확률 토큰 샘플링
        "temperature": 0.7,          # 창의성 제어
        "repetition_penalty": 1.2,   # 반복 억제
        "no_repeat_ngram_size": 2,   # 반복 단어 방지
        "early_stopping": True       # 의미가 없는 길이의 문장을 막기 위해 조기 종료
    }
    response = gpt2_pipeline(similar_sentence, **generation_args)
    return response[0]['generated_text']


# 5. Flask 라우팅 설정
@app.route('/')
def home():
    return render_template('tuned.html')

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    file = request.files['pdf']
    if file:
        file.save('uploaded_pdf.pdf')
        print(f"파일 저장 완료: uploaded_pdf.pdf")  # 디버깅 메시지 추가
        sentences = process_pdf('uploaded_pdf.pdf')
        print(f"추출된 문장: {sentences}")  # 추출된 문장 확인
        store_in_chroma(sentences)
        print("ChromaDB에 문장 저장 완료")  # ChromaDB 저장 확인
        return jsonify({"message": "PDF 업로드 및 처리 완료"}), 200
    return jsonify({"message": "파일 업로드 실패"}), 400

@app.route('/ask_question', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data.get('question', '')
    if question:
        print(f"사용자 질문: {question}")  # 질문 로그
        similar_sentence = find_similar_sentence(question)
        print(f"유사한 문장: {similar_sentence}")  # 유사 문장 확인
        answer = generate_response(similar_sentence)
        print(f"생성된 답변: {answer}")  # 생성된 답변 확인
        return jsonify({"answer": answer}), 200
    return jsonify({"answer": "질문을 입력해주세요"}), 400

if __name__ == '__main__':
    app.run(debug=True)

