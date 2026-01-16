#!/bin/bash
# Voice RAG System 설치 스크립트 (Termux용)

echo "=========================================="
echo "🎤 Voice RAG System 설치"
echo "=========================================="

# Termux 패키지 업데이트
echo "[1/4] 패키지 업데이트..."
pkg update -y && pkg upgrade -y

# 필수 패키지 설치
echo "[2/4] Python 및 빌드 도구 설치..."
pkg install -y python python-pip git build-essential

# Python 패키지 설치
echo "[3/4] Flask 설치..."
pip install flask

echo "[4/4] Hnswlib 및 Numpy 설치..."
pip install numpy hnswlib

echo ""
echo "=========================================="
echo "✅ 설치 완료!"
echo "=========================================="
echo ""
echo "서버 실행 방법:"
echo "  python voice_rag.py"
echo ""
echo "웹 UI 접속:"
echo "  http://localhost:5000/ui"
echo ""
echo "=========================================="
