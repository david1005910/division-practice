#!/usr/bin/env python3
"""
Hnswlib 기반 RAG 서버
경량 벡터 검색 - Termux/모바일 호환
"""

from flask import Flask, request, jsonify
import hnswlib
import numpy as np
import json
import os
import re
import hashlib
from collections import Counter
import math

app = Flask(__name__)

# 설정
EMBEDDING_DIM = 100  # 임베딩 차원
MAX_ELEMENTS = 10000  # 최대 문서 수
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_FILE = os.path.join(DATA_DIR, "hnswlib_index.bin")
DOCS_FILE = os.path.join(DATA_DIR, "hnswlib_docs.json")

# 전역 변수
index = None
documents = {}  # {doc_id: {"text": ..., "metadata": ..., "idx": ...}}
idx_to_doc_id = {}  # {idx: doc_id}
current_idx = 0
vocab = {}  # 단어 -> 인덱스 매핑
idf_values = {}  # IDF 값 저장


def tokenize(text):
    """한국어/영어 토크나이저"""
    text = text.lower()
    text = re.sub(r'[^\w\s가-힣]', ' ', text)
    tokens = text.split()
    stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                 'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
                 'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
                 'from', 'as', 'into', 'through', 'during', 'before', 'after',
                 'above', 'below', 'between', 'under', 'again', 'further', 'then',
                 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
                 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
                 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
                 '이', '가', '은', '는', '을', '를', '의', '에', '에서', '으로', '로',
                 '와', '과', '도', '만', '까지', '부터', '이다', '있다', '하다', '되다'}
    return [t for t in tokens if t not in stopwords and len(t) > 1]


def build_vocab(all_texts):
    """어휘 사전 구축"""
    global vocab, idf_values
    word_doc_count = Counter()
    all_words = set()

    tokenized_docs = []
    for text in all_texts:
        tokens = set(tokenize(text))
        tokenized_docs.append(tokens)
        all_words.update(tokens)
        for word in tokens:
            word_doc_count[word] += 1

    # 상위 빈출 단어로 vocab 구축
    vocab = {word: idx for idx, word in enumerate(sorted(all_words)[:EMBEDDING_DIM])}

    # IDF 계산
    n_docs = len(all_texts) + 1
    idf_values = {word: math.log(n_docs / (count + 1)) + 1
                  for word, count in word_doc_count.items()}


def text_to_embedding(text):
    """텍스트를 임베딩 벡터로 변환 (TF-IDF 기반)"""
    tokens = tokenize(text)

    if not tokens:
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    # TF 계산
    tf = Counter(tokens)
    total = len(tokens)

    # TF-IDF 벡터 생성
    embedding = np.zeros(EMBEDDING_DIM, dtype=np.float32)

    for word, count in tf.items():
        if word in vocab:
            tf_val = count / total
            idf_val = idf_values.get(word, 1.0)
            embedding[vocab[word]] = tf_val * idf_val

    # L2 정규화
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return embedding


def init_index():
    """Hnswlib 인덱스 초기화"""
    global index
    index = hnswlib.Index(space='cosine', dim=EMBEDDING_DIM)
    index.init_index(max_elements=MAX_ELEMENTS, ef_construction=200, M=16)
    index.set_ef(50)  # 검색 시 정확도


def save_data():
    """인덱스와 문서 저장"""
    if index and index.get_current_count() > 0:
        index.save_index(INDEX_FILE)

    save_obj = {
        "documents": documents,
        "idx_to_doc_id": {str(k): v for k, v in idx_to_doc_id.items()},
        "current_idx": current_idx,
        "vocab": vocab,
        "idf_values": idf_values
    }
    with open(DOCS_FILE, 'w', encoding='utf-8') as f:
        json.dump(save_obj, f, ensure_ascii=False, indent=2)


def load_data():
    """저장된 데이터 로드"""
    global documents, idx_to_doc_id, current_idx, index, vocab, idf_values

    init_index()

    if os.path.exists(DOCS_FILE):
        try:
            with open(DOCS_FILE, 'r', encoding='utf-8') as f:
                save_obj = json.load(f)

            documents = save_obj.get("documents", {})
            idx_to_doc_id = {int(k): v for k, v in save_obj.get("idx_to_doc_id", {}).items()}
            current_idx = save_obj.get("current_idx", 0)
            vocab = save_obj.get("vocab", {})
            idf_values = save_obj.get("idf_values", {})

            if os.path.exists(INDEX_FILE) and documents:
                index.load_index(INDEX_FILE, max_elements=MAX_ELEMENTS)
                print(f"인덱스 로드 완료: {index.get_current_count()}개 문서")
        except Exception as e:
            print(f"데이터 로드 실패: {e}")
            documents = {}
            idx_to_doc_id = {}
            current_idx = 0


def rebuild_index():
    """인덱스 재구축 (vocab 업데이트 시)"""
    global index, current_idx, idx_to_doc_id

    if not documents:
        return

    # vocab 재구축
    all_texts = [doc["text"] for doc in documents.values()]
    build_vocab(all_texts)

    # 인덱스 재생성
    init_index()
    current_idx = 0
    idx_to_doc_id = {}

    embeddings = []
    ids = []

    for doc_id, doc in documents.items():
        emb = text_to_embedding(doc["text"])
        embeddings.append(emb)
        ids.append(current_idx)
        idx_to_doc_id[current_idx] = doc_id
        doc["idx"] = current_idx
        current_idx += 1

    if embeddings:
        index.add_items(np.array(embeddings), ids)

    save_data()


# 시작 시 데이터 로드
load_data()


@app.route('/')
def home():
    """서버 상태"""
    return jsonify({
        "status": "running",
        "version": "hnswlib",
        "documents": len(documents),
        "index_size": index.get_current_count() if index else 0,
        "embedding_dim": EMBEDDING_DIM,
        "endpoints": {
            "POST /add": "문서 추가",
            "POST /add_batch": "여러 문서 추가",
            "POST /search": "문서 검색",
            "GET /list": "문서 목록",
            "DELETE /delete": "문서 삭제",
            "DELETE /clear": "전체 삭제",
            "POST /rebuild": "인덱스 재구축"
        }
    })


@app.route('/add', methods=['POST'])
def add_document():
    """문서 추가"""
    global current_idx

    data = request.json
    if not data or 'text' not in data:
        return jsonify({"error": "text 필드가 필요합니다"}), 400

    text = data['text']
    doc_id = data.get('id', f"doc_{len(documents) + 1}")
    metadata = data.get('metadata', {})

    # 이미 존재하는 문서면 업데이트
    if doc_id in documents:
        # 기존 인덱스에서 제거는 hnswlib에서 지원 안 함
        # 재구축 필요
        documents[doc_id] = {"text": text, "metadata": metadata}
        rebuild_index()
        return jsonify({
            "status": "updated",
            "id": doc_id,
            "total": len(documents)
        })

    # vocab에 새 단어 추가 (주기적으로 재구축 권장)
    tokens = tokenize(text)
    for word in tokens:
        if word not in vocab and len(vocab) < EMBEDDING_DIM:
            vocab[word] = len(vocab)
        if word not in idf_values:
            idf_values[word] = 1.0

    # 임베딩 생성
    embedding = text_to_embedding(text)

    # 인덱스에 추가
    index.add_items(np.array([embedding]), [current_idx])

    # 문서 저장
    documents[doc_id] = {
        "text": text,
        "metadata": metadata,
        "idx": current_idx
    }
    idx_to_doc_id[current_idx] = doc_id
    current_idx += 1

    save_data()

    return jsonify({
        "status": "success",
        "id": doc_id,
        "total": len(documents)
    })


@app.route('/add_batch', methods=['POST'])
def add_batch():
    """여러 문서 한번에 추가"""
    global current_idx

    data = request.json
    if not data or 'documents' not in data:
        return jsonify({"error": "documents 배열이 필요합니다"}), 400

    docs_to_add = data['documents']

    # vocab 구축을 위해 모든 텍스트 수집
    all_texts = [doc['text'] for doc in docs_to_add if 'text' in doc]
    existing_texts = [d['text'] for d in documents.values()]
    build_vocab(existing_texts + all_texts)

    # 문서 추가
    embeddings = []
    ids = []
    added = 0

    for doc in docs_to_add:
        if 'text' not in doc:
            continue

        doc_id = doc.get('id', f"doc_{len(documents) + added + 1}")

        if doc_id in documents:
            continue  # 중복 스킵

        embedding = text_to_embedding(doc['text'])
        embeddings.append(embedding)
        ids.append(current_idx)

        documents[doc_id] = {
            "text": doc['text'],
            "metadata": doc.get('metadata', {}),
            "idx": current_idx
        }
        idx_to_doc_id[current_idx] = doc_id
        current_idx += 1
        added += 1

    if embeddings:
        index.add_items(np.array(embeddings), ids)

    save_data()

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

    if not documents:
        return jsonify({
            "query": data['query'],
            "results": [],
            "count": 0
        })

    query = data['query']
    n_results = min(data.get('n', 3), len(documents))

    # 쿼리 임베딩
    query_embedding = text_to_embedding(query)

    # 검색
    labels, distances = index.knn_query(np.array([query_embedding]), k=n_results)

    # 결과 정리
    results = []
    for i, (label, distance) in enumerate(zip(labels[0], distances[0])):
        doc_id = idx_to_doc_id.get(label)
        if doc_id and doc_id in documents:
            doc = documents[doc_id]
            results.append({
                "id": doc_id,
                "text": doc['text'],
                "similarity": round(1 - distance, 4),  # cosine distance -> similarity
                "metadata": doc.get('metadata', {})
            })

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
    for i, (doc_id, doc) in enumerate(documents.items()):
        if i >= limit:
            break
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
    """문서 삭제 (재구축 필요)"""
    data = request.json
    if not data or 'id' not in data:
        return jsonify({"error": "id 필드가 필요합니다"}), 400

    doc_id = data['id']

    if doc_id not in documents:
        return jsonify({"error": "문서를 찾을 수 없습니다"}), 404

    del documents[doc_id]
    rebuild_index()

    return jsonify({
        "status": "success",
        "deleted": doc_id,
        "remaining": len(documents)
    })


@app.route('/clear', methods=['DELETE'])
def clear_all():
    """모든 문서 삭제"""
    global documents, idx_to_doc_id, current_idx, vocab, idf_values

    documents = {}
    idx_to_doc_id = {}
    current_idx = 0
    vocab = {}
    idf_values = {}

    init_index()

    # 파일 삭제
    if os.path.exists(INDEX_FILE):
        os.remove(INDEX_FILE)
    if os.path.exists(DOCS_FILE):
        os.remove(DOCS_FILE)

    return jsonify({
        "status": "success",
        "message": "모든 문서가 삭제되었습니다"
    })


@app.route('/rebuild', methods=['POST'])
def rebuild():
    """인덱스 재구축"""
    rebuild_index()
    return jsonify({
        "status": "success",
        "documents": len(documents),
        "index_size": index.get_current_count()
    })


if __name__ == '__main__':
    print("\n" + "="*50)
    print("Hnswlib RAG 서버")
    print("="*50)
    print(f"문서 수: {len(documents)}")
    print(f"인덱스 크기: {index.get_current_count() if index else 0}")
    print(f"임베딩 차원: {EMBEDDING_DIM}")
    print("="*50)
    print("\n서버 시작: http://0.0.0.0:5000\n")

    app.run(host='0.0.0.0', port=5000, debug=False)
