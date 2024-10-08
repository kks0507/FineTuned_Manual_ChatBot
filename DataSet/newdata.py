# -*- coding: utf-8 -*-
"""NewData.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/13jR0m-Wv2PRPJt5SHLW5d4v4vUauyxY8
"""

pip install datasets huggingface_hub

from huggingface_hub import login

login()  # API 토큰을 입력하면 로그인 완료

from google.colab import drive
import pandas as pd

# Google Drive 마운트
drive.mount('/content/drive')

# Google Drive에서 파일 경로 설정
file_path = '/content/drive/MyDrive/Colab Notebooks/NEW/dataset/scraped.csv'

# 파일 읽기
df = pd.read_csv(file_path)

# 파일 확인
print(df.head())

from datasets import Dataset

# 데이터셋을 Hugging Face의 Dataset 형식으로 변환
dataset = Dataset.from_pandas(df)

# 데이터셋 확인
print(dataset)

from huggingface_hub import HfApi

# HfApi 객체 생성
api = HfApi()

# Hugging Face에서 새로운 데이터셋 repository 생성 (repo_name은 원하는 이름으로 설정)
repo_url = api.create_repo(repo_id="kingkim/DS_Building_SecurityManual_V4", repo_type="dataset")

# 데이터셋을 Hugging Face Hub에 푸시
dataset.push_to_hub("kingkim/DS_Building_SecurityManual_V4")