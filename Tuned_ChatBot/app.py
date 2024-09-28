from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

app = Flask(__name__)

# Hugging Face 모델 불러오기
model_name = "kingkim/kodialogpt_v3.0_SecurityManual"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 텍스트 생성 파이프라인
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# 텍스트 생성 옵션
generation_args = {
    "repetition_penalty": 1.2,  # GPT-2 모델에 적합하게 약간 낮춤
    "no_repeat_ngram_size": 3,  # 4에서 3으로 줄여 더 다양한 답변을 유도
    "eos_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 50,  # 응답의 길이를 약간 줄여 빠른 응답을 유도
    "do_sample": True,
    "top_p": 0.65,  # 다소 자유로운 응답을 허용
    "temperature": 0.6,  # 온도를 약간 낮춰 일관된 답변을 유도
    "early_stopping": True
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask_question', methods=['POST'])
def ask_question():
    data = request.json
    user_input = data.get("question", "")
    
    # GPT-2 모델에 맞는 간단하고 명확한 프롬프트 작성
    input_text = f"질문: {user_input}\n답변:"
    
    # 질문에 대해 GPT-2 기반 응답 생성
    response = generator(input_text, **generation_args)
    
    # 결과 반환
    return jsonify({"answer": response[0]["generated_text"].strip()})

if __name__ == '__main__':
    app.run(debug=True)

