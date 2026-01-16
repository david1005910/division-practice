#!/usr/bin/env python3
"""
경량 RAG 서버 - 순수 Python 버전
외부 의존성: flask만 필요 (chromadb, sentence-transformers 불필요)
TF-IDF 기반 검색
"""

from flask import Flask, request, jsonify
import json
import os
import re
import math
from collections import Counter

app = Flask(__name__)

# 데이터 저장 경로
DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "documents.json")

# 문서 저장소
documents = {}

def load_documents():
    """저장된 문서 로드"""
    global documents
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, 'r', encoding='utf-8') as f:
                documents = json.load(f)
        except:
            documents = {}

def save_documents():
    """문서 저장"""
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)

def tokenize(text):
    """간단한 토크나이저 (한국어/영어 지원)"""
    # 소문자 변환 및 특수문자 제거
    text = text.lower()
    # 한글, 영문, 숫자만 남기고 공백으로 분리
    text = re.sub(r'[^\w\s가-힣]', ' ', text)
    tokens = text.split()
    # 불용어 제거 (간단한 버전)
    stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                 '이', '가', '은', '는', '을', '를', '의', '에', '에서', '으로', '로'}
    return [t for t in tokens if t not in stopwords and len(t) > 1]

def compute_tf(tokens):
    """Term Frequency 계산"""
    tf = Counter(tokens)
    total = len(tokens)
    return {word: count / total for word, count in tf.items()} if total > 0 else {}

def compute_idf(all_docs_tokens):
    """Inverse Document Frequency 계산"""
    idf = {}
    n_docs = len(all_docs_tokens)

    # 모든 단어 수집
    all_words = set()
    for tokens in all_docs_tokens:
        all_words.update(tokens)

    # IDF 계산
    for word in all_words:
        doc_count = sum(1 for tokens in all_docs_tokens if word in tokens)
        idf[word] = math.log(n_docs / (1 + doc_count)) + 1

    return idf

def compute_tfidf(tokens, idf):
    """TF-IDF 벡터 계산"""
    tf = compute_tf(tokens)
    return {word: tf_val * idf.get(word, 1) for word, tf_val in tf.items()}

def cosine_similarity(vec1, vec2):
    """코사인 유사도 계산"""
    # 공통 키 찾기
    common_keys = set(vec1.keys()) & set(vec2.keys())

    if not common_keys:
        return 0.0

    # 내적
    dot_product = sum(vec1[k] * vec2[k] for k in common_keys)

    # 크기
    mag1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
    mag2 = math.sqrt(sum(v ** 2 for v in vec2.values()))

    if mag1 == 0 or mag2 == 0:
        return 0.0

    return dot_product / (mag1 * mag2)

def search_documents(query, n_results=3):
    """문서 검색"""
    if not documents:
        return []

    # 모든 문서 토큰화
    doc_tokens = {doc_id: tokenize(doc['text']) for doc_id, doc in documents.items()}

    # IDF 계산
    idf = compute_idf(list(doc_tokens.values()))

    # 쿼리 TF-IDF
    query_tokens = tokenize(query)
    query_tfidf = compute_tfidf(query_tokens, idf)

    # 각 문서와 유사도 계산
    similarities = []
    for doc_id, tokens in doc_tokens.items():
        doc_tfidf = compute_tfidf(tokens, idf)
        sim = cosine_similarity(query_tfidf, doc_tfidf)
        similarities.append((doc_id, sim))

    # 유사도 순 정렬
    similarities.sort(key=lambda x: x[1], reverse=True)

    # 상위 n개 반환
    results = []
    for doc_id, sim in similarities[:n_results]:
        results.append({
            "id": doc_id,
            "text": documents[doc_id]['text'],
            "similarity": round(sim, 4),
            "metadata": documents[doc_id].get('metadata', {})
        })

    return results

# 시작 시 문서 로드
load_documents()

@app.route('/')
def home():
    """서버 상태"""
    return jsonify({
        "status": "running",
        "version": "simple (TF-IDF)",
        "documents": len(documents),
        "endpoints": {
            "POST /add": "문서 추가",
            "POST /search": "문서 검색",
            "GET /list": "문서 목록",
            "DELETE /delete": "문서 삭제"
        }
    })

@app.route('/add', methods=['POST'])
def add_document():
    """문서 추가"""
    data = request.json

    if not data or 'text' not in data:
        return jsonify({"error": "text 필드가 필요합니다"}), 400

    doc_id = data.get('id', f"doc_{len(documents) + 1}")

    documents[doc_id] = {
        "text": data['text'],
        "metadata": data.get('metadata', {})
    }

    save_documents()

    return jsonify({
        "status": "success",
        "id": doc_id,
        "total": len(documents)
    })

@app.route('/add_batch', methods=['POST'])
def add_batch():
    """여러 문서 추가"""
    data = request.json

    if not data or 'documents' not in data:
        return jsonify({"error": "documents 배열이 필요합니다"}), 400

    added = 0
    for i, doc in enumerate(data['documents']):
        if 'text' in doc:
            doc_id = doc.get('id', f"doc_{len(documents) + 1}")
            documents[doc_id] = {
                "text": doc['text'],
                "metadata": doc.get('metadata', {})
            }
            added += 1

    save_documents()

    return jsonify({
        "status": "success",
        "added": added,
        "total": len(documents)
    })

@app.route('/search', methods=['POST'])
def search():
    """문서 검색"""
    data = request.json

    if not data or 'query' not in data:
        return jsonify({"error": "query 필드가 필요합니다"}), 400

    query = data['query']
    n = data.get('n', 3)

    results = search_documents(query, n)

    return jsonify({
        "query": query,
        "results": results,
        "count": len(results)
    })

@app.route('/list', methods=['GET'])
def list_documents():
    """문서 목록"""
    limit = request.args.get('limit', 100, type=int)

    doc_list = []
    for doc_id, doc in list(documents.items())[:limit]:
        doc_list.append({
            "id": doc_id,
            "text": doc['text'][:100] + "..." if len(doc['text']) > 100 else doc['text'],
            "metadata": doc.get('metadata', {})
        })

    return jsonify({
        "total": len(documents),
        "showing": len(doc_list),
        "documents": doc_list
    })

@app.route('/delete', methods=['DELETE'])
def delete_document():
    """문서 삭제"""
    data = request.json

    if not data or 'id' not in data:
        return jsonify({"error": "id 필드가 필요합니다"}), 400

    doc_id = data['id']

    if doc_id in documents:
        del documents[doc_id]
        save_documents()
        return jsonify({
            "status": "success",
            "deleted": doc_id,
            "remaining": len(documents)
        })
    else:
        return jsonify({"error": "문서를 찾을 수 없습니다"}), 404

@app.route('/clear', methods=['DELETE'])
def clear_all():
    """모든 문서 삭제"""
    global documents
    documents = {}
    save_documents()
    return jsonify({
        "status": "success",
        "message": "모든 문서가 삭제되었습니다"
    })

if __name__ == '__main__':
    print("\n" + "="*50)
    print("경량 RAG 서버 (TF-IDF 기반)")
    print("="*50)
    print(f"문서 수: {len(documents)}")
    print("="*50)
    print("\n서버 시작: http://0.0.0.0:5000\n")

    app.run(host='0.0.0.0', port=5000, debug=False)
