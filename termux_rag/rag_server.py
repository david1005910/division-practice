#!/usr/bin/env python3
"""
Termux용 경량 RAG 서버
ChromaDB + Sentence Transformers 기반
"""

from flask import Flask, request, jsonify
import chromadb
from sentence_transformers import SentenceTransformer
import os

app = Flask(__name__)

# 데이터 저장 경로
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")

# 경량 모델 (약 90MB, 핸드폰에서 돌릴 수 있음)
print("모델 로딩 중... (처음에는 다운로드가 필요합니다)")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("모델 로딩 완료!")

# ChromaDB 클라이언트 (영구 저장)
client = chromadb.PersistentClient(path=DATA_DIR)
collection = client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}
)

@app.route('/')
def home():
    """서버 상태 확인"""
    doc_count = collection.count()
    return jsonify({
        "status": "running",
        "documents": doc_count,
        "endpoints": {
            "POST /add": "문서 추가",
            "POST /search": "문서 검색",
            "GET /list": "전체 문서 목록",
            "DELETE /delete": "문서 삭제"
        }
    })

@app.route('/add', methods=['POST'])
def add_document():
    """문서 추가"""
    data = request.json

    if not data or 'text' not in data:
        return jsonify({"error": "text 필드가 필요합니다"}), 400

    text = data['text']
    doc_id = data.get('id', f"doc_{collection.count() + 1}")
    metadata = data.get('metadata', {})

    # 임베딩 생성
    embedding = model.encode(text).tolist()

    # ChromaDB에 저장
    collection.add(
        documents=[text],
        embeddings=[embedding],
        ids=[doc_id],
        metadatas=[metadata] if metadata else None
    )

    return jsonify({
        "status": "success",
        "id": doc_id,
        "message": f"문서가 추가되었습니다. (총 {collection.count()}개)"
    })

@app.route('/add_batch', methods=['POST'])
def add_batch():
    """여러 문서 한번에 추가"""
    data = request.json

    if not data or 'documents' not in data:
        return jsonify({"error": "documents 배열이 필요합니다"}), 400

    documents = data['documents']
    texts = [doc['text'] for doc in documents]
    ids = [doc.get('id', f"doc_{collection.count() + i + 1}") for i, doc in enumerate(documents)]
    metadatas = [doc.get('metadata', {}) for doc in documents]

    # 임베딩 생성
    embeddings = model.encode(texts).tolist()

    # ChromaDB에 저장
    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )

    return jsonify({
        "status": "success",
        "added": len(documents),
        "total": collection.count()
    })

@app.route('/search', methods=['POST'])
def search():
    """문서 검색"""
    data = request.json

    if not data or 'query' not in data:
        return jsonify({"error": "query 필드가 필요합니다"}), 400

    query = data['query']
    n_results = data.get('n', 3)

    # 쿼리 임베딩 생성
    query_embedding = model.encode(query).tolist()

    # 검색
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "distances", "metadatas"]
    )

    # 결과 정리
    formatted_results = []
    if results['ids'] and results['ids'][0]:
        for i, doc_id in enumerate(results['ids'][0]):
            formatted_results.append({
                "id": doc_id,
                "text": results['documents'][0][i] if results['documents'] else "",
                "distance": results['distances'][0][i] if results['distances'] else 0,
                "similarity": 1 - results['distances'][0][i] if results['distances'] else 1,
                "metadata": results['metadatas'][0][i] if results['metadatas'] else {}
            })

    return jsonify({
        "query": query,
        "results": formatted_results,
        "count": len(formatted_results)
    })

@app.route('/list', methods=['GET'])
def list_documents():
    """전체 문서 목록"""
    limit = request.args.get('limit', 100, type=int)

    results = collection.get(
        limit=limit,
        include=["documents", "metadatas"]
    )

    documents = []
    if results['ids']:
        for i, doc_id in enumerate(results['ids']):
            documents.append({
                "id": doc_id,
                "text": results['documents'][i][:100] + "..." if len(results['documents'][i]) > 100 else results['documents'][i],
                "metadata": results['metadatas'][i] if results['metadatas'] else {}
            })

    return jsonify({
        "total": collection.count(),
        "showing": len(documents),
        "documents": documents
    })

@app.route('/delete', methods=['DELETE'])
def delete_document():
    """문서 삭제"""
    data = request.json

    if not data or 'id' not in data:
        return jsonify({"error": "id 필드가 필요합니다"}), 400

    doc_id = data['id']

    try:
        collection.delete(ids=[doc_id])
        return jsonify({
            "status": "success",
            "deleted": doc_id,
            "remaining": collection.count()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/clear', methods=['DELETE'])
def clear_all():
    """모든 문서 삭제"""
    global collection
    client.delete_collection("documents")
    collection = client.get_or_create_collection(
        name="documents",
        metadata={"hnsw:space": "cosine"}
    )
    return jsonify({
        "status": "success",
        "message": "모든 문서가 삭제되었습니다"
    })

if __name__ == '__main__':
    print("\n" + "="*50)
    print("RAG 서버 시작!")
    print("="*50)
    print(f"문서 수: {collection.count()}")
    print(f"데이터 저장 위치: {DATA_DIR}")
    print("="*50)
    print("\nAPI 엔드포인트:")
    print("  POST /add      - 문서 추가")
    print("  POST /search   - 문서 검색")
    print("  GET  /list     - 문서 목록")
    print("  DELETE /delete - 문서 삭제")
    print("="*50 + "\n")

    app.run(host='0.0.0.0', port=5000, debug=False)
