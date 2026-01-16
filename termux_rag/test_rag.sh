#!/bin/bash
# RAG 서버 테스트 스크립트

BASE_URL="http://localhost:5000"

echo "=========================================="
echo "RAG 서버 테스트"
echo "=========================================="

# 1. 서버 상태 확인
echo ""
echo "[1] 서버 상태 확인"
curl -s $BASE_URL/ | python -m json.tool

# 2. 문서 추가
echo ""
echo "[2] 문서 추가"
curl -s -X POST $BASE_URL/add \
  -H "Content-Type: application/json" \
  -d '{"text": "파이썬은 간단하고 배우기 쉬운 프로그래밍 언어입니다.", "id": "python1"}' | python -m json.tool

curl -s -X POST $BASE_URL/add \
  -H "Content-Type: application/json" \
  -d '{"text": "자바스크립트는 웹 개발에 많이 사용되는 언어입니다.", "id": "js1"}' | python -m json.tool

curl -s -X POST $BASE_URL/add \
  -H "Content-Type: application/json" \
  -d '{"text": "머신러닝은 인공지능의 한 분야로 데이터에서 패턴을 학습합니다.", "id": "ml1"}' | python -m json.tool

# 3. 검색
echo ""
echo "[3] 검색: '프로그래밍 언어'"
curl -s -X POST $BASE_URL/search \
  -H "Content-Type: application/json" \
  -d '{"query": "프로그래밍 언어", "n": 2}' | python -m json.tool

echo ""
echo "[4] 검색: '인공지능'"
curl -s -X POST $BASE_URL/search \
  -H "Content-Type: application/json" \
  -d '{"query": "인공지능", "n": 2}' | python -m json.tool

# 4. 문서 목록
echo ""
echo "[5] 문서 목록"
curl -s $BASE_URL/list | python -m json.tool

echo ""
echo "=========================================="
echo "테스트 완료!"
echo "=========================================="
