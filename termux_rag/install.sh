#!/bin/bash
# Termux RAG 서버 설치 스크립트

echo "=========================================="
echo "Termux RAG 서버 설치 시작"
echo "=========================================="

# 패키지 업데이트
echo "[1/5] 패키지 업데이트 중..."
pkg update -y && pkg upgrade -y

# 기본 패키지 설치
echo "[2/5] 기본 패키지 설치 중..."
pkg install -y python python-pip git build-essential rust

# Python 패키지 설치
echo "[3/5] Flask 설치 중..."
pip install flask

echo "[4/5] ChromaDB 설치 중... (시간이 좀 걸립니다)"
pip install chromadb

echo "[5/5] Sentence Transformers 설치 중... (시간이 좀 걸립니다)"
pip install sentence-transformers

echo "=========================================="
echo "설치 완료!"
echo ""
echo "서버 실행 방법:"
echo "  python rag_server.py"
echo ""
echo "테스트:"
echo "  curl http://localhost:5000/"
echo "=========================================="
