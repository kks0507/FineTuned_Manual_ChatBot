# FineTuned_Manual_ChatBot 🚀

## 프로젝트 설명
**FineTuned_Manual_ChatBot**은 GPT-2 기반의 모델에 직접 파인튜닝을 통해 빌딩 보안 매뉴얼을 학습시키고, 이를 바탕으로 챗봇을 개발한 프로젝트입니다. 
이 챗봇은 보안 요원을 위한 매뉴얼에서 발생 가능한 다양한 질문에 대한 답변을 제공합니다. 
이번 프로젝트의 주요 목적은 보안 매뉴얼 데이터를 효율적으로 모델에 학습시키고, 로컬 CPU 환경에서도 실행 가능한 챗봇을 구축하는 것이었습니다. 
또한 파인튜닝된 모델을 활용해 사용자의 질문에 적합한 답변을 제공하는 시스템을 구현했습니다.

<br>

## 설치 및 실행 방법
### 1. 필수 패키지 설치
다음 명령어로 필요한 라이브러리를 설치합니다:
```bash
pip install -r requirements.txt
```

### 2. Flask 애플리케이션 실행
```bash
python app.py
```

### 3. PDF 업로드 및 처리
웹 브라우저에서 Flask 서버가 실행된 후, PDF 파일을 업로드하여 해당 파일의 내용을 분석하고 ChromaDB에 저장할 수 있습니다.

### 4. 질문에 대한 답변 생성
업로드된 매뉴얼을 기반으로 질문을 입력하면 유사한 문장들을 찾아 GPT-2 모델을 통해 답변을 생성합니다.

<br>

## 데이터셋 정보
이번 프로젝트에서는 보안 매뉴얼을 JSON 형식으로 변환한 후, Hugging Face에 `kingkim/DS_Building_SecurityManual_V5` (https://huggingface.co/datasets/kingkim/DS_Building_SecurityManual_V5)로 업로드한 데이터셋을 사용했습니다. 
해당 데이터셋은 다음과 같은 과정을 통해 생성되었습니다:
- PDF 파일을 랭체인과 ChatGPT API를 사용하여 질문-답변 형식의 JSON 데이터로 변환.
- 데이터 증강 기법을 활용하여 200개의 질문-응답 구조로 구성된 데이터셋으로 확장.
- Hugging Face에서 파인튜닝 과정에 활용.

<br>

## 모델 정보
프로젝트에서 사용된 모델은 GPT-2 기반의 `kingkim/kodialogpt_v3.0_SecurityManual` (https://huggingface.co/kingkim/kodialogpt_v3.0_SecurityManual)입니다. 이 모델은 한국어 대화 데이터로 학습된 `heegyu/kodialogpt-v1` 모델을 기반으로 보안 매뉴얼에 맞춰 파인튜닝되었습니다. 
모델의 학습 과정은 구글 코랩에서 진행되었으며, 주요 학습 파라미터는 다음과 같습니다:
- 학습률: 0.0002
- 배치 크기: 8
- Adam 옵티마이저 사용
- 에폭 수: 5
최종적으로 0.9083의 검증 손실 값을 기록하였습니다. 이 모델은 로컬 CPU 환경에서도 원활하게 작동하도록 최적화되었습니다.

<br>

## 기술 스택
- **Transformers**: 사전 학습된 언어 모델을 불러오고 파인튜닝하기 위한 Hugging Face의 라이브러리.
- **Datasets**: 대규모 데이터셋을 효율적으로 처리하고 활용하는 라이브러리.
- **Accelerate**: GPU나 TPU 같은 하드웨어 가속기를 활용하여 모델 훈련을 가속화하는 라이브러리.
- **Huggingface_hub**: Hugging Face와 통신하여 모델 및 데이터셋을 공유하고 관리하는 라이브러리.
- **Sentence-Transformers**: 문장 단위로 임베딩을 생성하여 벡터화하는 라이브러리.
- **PyPDF2**: PDF 파일에서 텍스트를 추출하고 처리하는 파이썬 라이브러리.
- **ChromaDB**: 벡터 데이터를 저장하고 검색하는 벡터 데이터베이스.
- **Flask**: 간단한 웹 애플리케이션을 구축하기 위한 파이썬 마이크로 웹 프레임워크.

<br>

## 주요 기능 및 개선 사항
### 임베딩 품질 개선
현재 `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` 모델을 사용하여 보안 매뉴얼 데이터를 임베딩하고 있으며, 문맥을 고려한 더 나은 결과를 얻기 위해 문장 단위가 아닌 **여러 문장 묶음**으로 임베딩하는 방식을 적용했습니다. 
이를 통해 더 많은 정보를 포함하고 질문에 대해 더 정확한 답변을 생성할 수 있게 했습니다.

### GPT-2 답변 생성 품질 개선
GPT-2 모델이 생성하는 답변의 품질을 높이기 위해 프롬프트를 보다 명확하게 제공하며, 유사 문장을 결합하여 모델이 적합한 답변을 생성하도록 했습니다. 
또한 파라미터 (`top_p`, `temperature`, `repetition_penalty`)를 조정하여 중복 답변을 최소화하고 답변의 품질을 향상시켰습니다.

<br>

## 중요 함수 및 코드 설명

### `process_pdf`
```python
def process_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    sentences = text.split('.')
    grouped_sentences = [' '.join(sentences[i:i+3]) for i in range(0, len(sentences), 3)]
    return grouped_sentences
```
- PDF 파일을 읽고, 각 페이지에서 텍스트를 추출한 후 문장을 3개씩 묶어 반환하는 함수입니다.

### `generate_response`
```python
def generate_response(similar_sentence, question):
    prompt = f"질문: {question}\n유사한 문장: {similar_sentence}\n답변:"
    response = gpt2_pipeline(prompt, **generation_args)
    return response[0]['generated_text']
```
- 질문과 유사한 문장을 GPT-2 모델에 전달하여 적절한 답변을 생성하는 함수입니다.

## 프로젝트 참여자
- kks0507

## 코드
프로젝트 전체 코드는 GitHub에서 확인하실 수 있습니다: https://github.com/kks0507/FineTuned_Manual_ChatBot.git
