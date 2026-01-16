#!/usr/bin/env python3
"""
Voice RAG System - Termux/모바일용
Hnswlib 기반 벡터 검색 + 음성 입출력
"""

from flask import Flask, request, jsonify, render_template_string
import hnswlib
import numpy as np
import json
import os
import re
from collections import Counter
import math

app = Flask(__name__)

# ===== 설정 =====
EMBEDDING_DIM = 100
MAX_ELEMENTS = 10000
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_FILE = os.path.join(DATA_DIR, "voice_rag_index.bin")
DOCS_FILE = os.path.join(DATA_DIR, "voice_rag_docs.json")

# ===== 전역 변수 =====
index = None
documents = {}
idx_to_doc_id = {}
current_idx = 0
vocab = {}
idf_values = {}


# ===== 텍스트 처리 함수 =====
def tokenize(text):
    """한국어/영어 토크나이저"""
    text = text.lower()
    text = re.sub(r'[^\w\s가-힣]', ' ', text)
    tokens = text.split()
    stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                 '이', '가', '은', '는', '을', '를', '의', '에', '에서', '으로',
                 '로', '와', '과', '도', '만', '까지', '부터'}
    return [t for t in tokens if t not in stopwords and len(t) > 1]


def build_vocab(all_texts):
    """어휘 사전 구축"""
    global vocab, idf_values
    word_doc_count = Counter()
    all_words = set()

    for text in all_texts:
        tokens = set(tokenize(text))
        all_words.update(tokens)
        for word in tokens:
            word_doc_count[word] += 1

    vocab = {word: idx for idx, word in enumerate(sorted(all_words)[:EMBEDDING_DIM])}
    n_docs = len(all_texts) + 1
    idf_values = {word: math.log(n_docs / (count + 1)) + 1
                  for word, count in word_doc_count.items()}


def text_to_embedding(text):
    """텍스트를 임베딩 벡터로 변환"""
    tokens = tokenize(text)
    if not tokens:
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    tf = Counter(tokens)
    total = len(tokens)
    embedding = np.zeros(EMBEDDING_DIM, dtype=np.float32)

    for word, count in tf.items():
        if word in vocab:
            tf_val = count / total
            idf_val = idf_values.get(word, 1.0)
            embedding[vocab[word]] = tf_val * idf_val

    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding


# ===== 인덱스 관리 =====
def init_index():
    global index
    index = hnswlib.Index(space='cosine', dim=EMBEDDING_DIM)
    index.init_index(max_elements=MAX_ELEMENTS, ef_construction=200, M=16)
    index.set_ef(50)


def save_data():
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
        except Exception as e:
            print(f"데이터 로드 실패: {e}")


def rebuild_index():
    global index, current_idx, idx_to_doc_id
    if not documents:
        return
    all_texts = [doc["text"] for doc in documents.values()]
    build_vocab(all_texts)
    init_index()
    current_idx = 0
    idx_to_doc_id = {}
    embeddings, ids = [], []
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


# 시작 시 로드
load_data()


# ===== 웹 UI (음성 지원) =====
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎤 Voice RAG System</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 600px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            padding: 25px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 20px;
            font-size: 1.5rem;
        }
        .tabs {
            display: flex;
            margin-bottom: 20px;
            border-radius: 10px;
            overflow: hidden;
            border: 2px solid #667eea;
        }
        .tab {
            flex: 1;
            padding: 12px;
            text-align: center;
            cursor: pointer;
            background: white;
            color: #667eea;
            font-weight: bold;
            transition: all 0.3s;
        }
        .tab.active {
            background: #667eea;
            color: white;
        }
        .section { display: none; }
        .section.active { display: block; }
        .voice-btn {
            width: 120px;
            height: 120px;
            border-radius: 50%;
            border: none;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            font-size: 3rem;
            cursor: pointer;
            margin: 20px auto;
            display: block;
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .voice-btn:hover { transform: scale(1.05); }
        .voice-btn:active { transform: scale(0.95); }
        .voice-btn.recording {
            background: linear-gradient(135deg, #f5576c, #f093fb);
            animation: pulse 1s infinite;
        }
        @keyframes pulse {
            0%, 100% { box-shadow: 0 0 0 0 rgba(245, 87, 108, 0.4); }
            50% { box-shadow: 0 0 0 20px rgba(245, 87, 108, 0); }
        }
        .status {
            text-align: center;
            color: #666;
            margin: 15px 0;
            min-height: 24px;
        }
        .input-group {
            margin: 15px 0;
        }
        .input-group label {
            display: block;
            margin-bottom: 8px;
            color: #333;
            font-weight: 500;
        }
        textarea, input[type="text"] {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 1rem;
            transition: border-color 0.3s;
        }
        textarea:focus, input:focus {
            outline: none;
            border-color: #667eea;
        }
        textarea { min-height: 100px; resize: vertical; }
        .btn {
            width: 100%;
            padding: 15px;
            border: none;
            border-radius: 10px;
            font-size: 1rem;
            font-weight: bold;
            cursor: pointer;
            transition: transform 0.2s;
            margin: 10px 0;
        }
        .btn:hover { transform: translateY(-2px); }
        .btn-primary {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
        }
        .btn-success {
            background: linear-gradient(135deg, #11998e, #38ef7d);
            color: white;
        }
        .btn-danger {
            background: linear-gradient(135deg, #f5576c, #f093fb);
            color: white;
        }
        .results {
            margin-top: 20px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 10px;
            max-height: 300px;
            overflow-y: auto;
        }
        .result-item {
            background: white;
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 10px;
            border-left: 4px solid #667eea;
        }
        .result-item .score {
            font-size: 0.8rem;
            color: #667eea;
            font-weight: bold;
        }
        .result-item .text {
            margin-top: 5px;
            color: #333;
        }
        .doc-count {
            text-align: center;
            padding: 10px;
            background: #e8f4f8;
            border-radius: 10px;
            margin-bottom: 15px;
            color: #333;
        }
        .speak-btn {
            background: #667eea;
            color: white;
            border: none;
            padding: 5px 10px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 0.8rem;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎤 Voice RAG System</h1>

        <div class="doc-count" id="docCount">📚 문서: 0개</div>

        <div class="tabs">
            <div class="tab active" onclick="showTab('search')">🔍 검색</div>
            <div class="tab" onclick="showTab('add')">➕ 추가</div>
            <div class="tab" onclick="showTab('list')">📋 목록</div>
        </div>

        <!-- 검색 탭 -->
        <div id="search" class="section active">
            <button class="voice-btn" id="searchVoiceBtn" onclick="toggleVoice('search')">🎤</button>
            <p class="status" id="searchStatus">마이크 버튼을 눌러 검색어를 말하세요</p>

            <div class="input-group">
                <label>또는 텍스트로 검색:</label>
                <input type="text" id="searchInput" placeholder="검색어 입력...">
            </div>

            <button class="btn btn-primary" onclick="searchDocs()">🔍 검색</button>

            <div class="results" id="searchResults"></div>
        </div>

        <!-- 추가 탭 -->
        <div id="add" class="section">
            <button class="voice-btn" id="addVoiceBtn" onclick="toggleVoice('add')">🎤</button>
            <p class="status" id="addStatus">마이크 버튼을 눌러 문서를 말하세요</p>

            <div class="input-group">
                <label>문서 ID (선택):</label>
                <input type="text" id="docId" placeholder="자동 생성됨">
            </div>

            <div class="input-group">
                <label>문서 내용:</label>
                <textarea id="docText" placeholder="문서 내용을 입력하세요..."></textarea>
            </div>

            <button class="btn btn-success" onclick="addDoc()">➕ 문서 추가</button>
        </div>

        <!-- 목록 탭 -->
        <div id="list" class="section">
            <button class="btn btn-primary" onclick="loadDocs()">🔄 새로고침</button>
            <div class="results" id="docList"></div>
            <button class="btn btn-danger" onclick="clearAll()">🗑️ 전체 삭제</button>
        </div>
    </div>

    <script>
        let recognition = null;
        let isRecording = false;
        let currentMode = '';

        // Web Speech API 초기화
        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            recognition = new SpeechRecognition();
            recognition.continuous = false;
            recognition.interimResults = true;
            recognition.lang = 'ko-KR';

            recognition.onresult = (event) => {
                const transcript = event.results[0][0].transcript;
                if (currentMode === 'search') {
                    document.getElementById('searchInput').value = transcript;
                    document.getElementById('searchStatus').textContent = '인식됨: ' + transcript;
                } else if (currentMode === 'add') {
                    document.getElementById('docText').value = transcript;
                    document.getElementById('addStatus').textContent = '인식됨: ' + transcript.substring(0, 30) + '...';
                }
            };

            recognition.onend = () => {
                isRecording = false;
                document.getElementById(currentMode + 'VoiceBtn').classList.remove('recording');
                if (currentMode === 'search' && document.getElementById('searchInput').value) {
                    searchDocs();
                }
            };

            recognition.onerror = (event) => {
                console.error('음성 인식 오류:', event.error);
                document.getElementById(currentMode + 'Status').textContent = '오류: ' + event.error;
                isRecording = false;
                document.getElementById(currentMode + 'VoiceBtn').classList.remove('recording');
            };
        }

        function toggleVoice(mode) {
            currentMode = mode;
            if (!recognition) {
                alert('이 브라우저는 음성 인식을 지원하지 않습니다.');
                return;
            }

            if (isRecording) {
                recognition.stop();
                isRecording = false;
                document.getElementById(mode + 'VoiceBtn').classList.remove('recording');
            } else {
                recognition.start();
                isRecording = true;
                document.getElementById(mode + 'VoiceBtn').classList.add('recording');
                document.getElementById(mode + 'Status').textContent = '듣고 있습니다...';
            }
        }

        function speak(text) {
            if ('speechSynthesis' in window) {
                const utterance = new SpeechSynthesisUtterance(text);
                utterance.lang = 'ko-KR';
                utterance.rate = 1.0;
                speechSynthesis.speak(utterance);
            }
        }

        function showTab(tabName) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
            document.querySelector(`[onclick="showTab('${tabName}')"]`).classList.add('active');
            document.getElementById(tabName).classList.add('active');
            if (tabName === 'list') loadDocs();
        }

        async function updateDocCount() {
            try {
                const res = await fetch('/');
                const data = await res.json();
                document.getElementById('docCount').textContent = '📚 문서: ' + data.documents + '개';
            } catch(e) {}
        }

        async function searchDocs() {
            const query = document.getElementById('searchInput').value.trim();
            if (!query) {
                alert('검색어를 입력하세요');
                return;
            }

            try {
                const res = await fetch('/search', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({query: query, n: 5})
                });
                const data = await res.json();

                let html = '';
                if (data.results && data.results.length > 0) {
                    data.results.forEach((r, i) => {
                        html += `<div class="result-item">
                            <div class="score">📊 유사도: ${(r.similarity * 100).toFixed(1)}%</div>
                            <div class="text">${r.text}</div>
                            <button class="speak-btn" onclick="speak('${r.text.replace(/'/g, "\\'")}')">🔊 읽기</button>
                        </div>`;
                    });
                    // 첫 번째 결과 음성으로 읽기
                    speak(data.results[0].text);
                } else {
                    html = '<p>검색 결과가 없습니다.</p>';
                    speak('검색 결과가 없습니다.');
                }
                document.getElementById('searchResults').innerHTML = html;
            } catch(e) {
                alert('검색 실패: ' + e.message);
            }
        }

        async function addDoc() {
            const text = document.getElementById('docText').value.trim();
            if (!text) {
                alert('문서 내용을 입력하세요');
                return;
            }

            const docId = document.getElementById('docId').value.trim() || undefined;

            try {
                const res = await fetch('/add', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({text: text, id: docId})
                });
                const data = await res.json();

                if (data.status === 'success') {
                    alert('문서가 추가되었습니다! ID: ' + data.id);
                    speak('문서가 추가되었습니다.');
                    document.getElementById('docText').value = '';
                    document.getElementById('docId').value = '';
                    document.getElementById('addStatus').textContent = '마이크 버튼을 눌러 문서를 말하세요';
                    updateDocCount();
                }
            } catch(e) {
                alert('추가 실패: ' + e.message);
            }
        }

        async function loadDocs() {
            try {
                const res = await fetch('/list?limit=50');
                const data = await res.json();

                let html = '';
                if (data.documents && data.documents.length > 0) {
                    data.documents.forEach(doc => {
                        html += `<div class="result-item">
                            <div class="score">🏷️ ${doc.id}</div>
                            <div class="text">${doc.text}</div>
                            <button class="speak-btn" onclick="speak('${doc.text.replace(/'/g, "\\'")}')">🔊 읽기</button>
                            <button class="speak-btn" style="background:#f5576c" onclick="deleteDoc('${doc.id}')">🗑️ 삭제</button>
                        </div>`;
                    });
                } else {
                    html = '<p>저장된 문서가 없습니다.</p>';
                }
                document.getElementById('docList').innerHTML = html;
                updateDocCount();
            } catch(e) {
                alert('목록 로드 실패: ' + e.message);
            }
        }

        async function deleteDoc(docId) {
            if (!confirm('정말 삭제하시겠습니까?')) return;

            try {
                const res = await fetch('/delete', {
                    method: 'DELETE',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({id: docId})
                });
                const data = await res.json();
                if (data.status === 'success') {
                    loadDocs();
                    speak('삭제되었습니다.');
                }
            } catch(e) {
                alert('삭제 실패: ' + e.message);
            }
        }

        async function clearAll() {
            if (!confirm('모든 문서를 삭제하시겠습니까?')) return;

            try {
                const res = await fetch('/clear', {method: 'DELETE'});
                const data = await res.json();
                if (data.status === 'success') {
                    loadDocs();
                    speak('모든 문서가 삭제되었습니다.');
                }
            } catch(e) {
                alert('삭제 실패: ' + e.message);
            }
        }

        // 초기화
        updateDocCount();

        // Enter 키로 검색
        document.getElementById('searchInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') searchDocs();
        });
    </script>
</body>
</html>
'''


# ===== API 엔드포인트 =====
@app.route('/')
def home():
    return jsonify({
        "status": "running",
        "version": "voice-rag-hnswlib",
        "documents": len(documents),
        "index_size": index.get_current_count() if index else 0
    })


@app.route('/ui')
def ui():
    return render_template_string(HTML_TEMPLATE)


@app.route('/add', methods=['POST'])
def add_document():
    global current_idx
    data = request.json
    if not data or 'text' not in data:
        return jsonify({"error": "text 필드가 필요합니다"}), 400

    text = data['text']
    doc_id = data.get('id', f"doc_{len(documents) + 1}")
    metadata = data.get('metadata', {})

    if doc_id in documents:
        documents[doc_id] = {"text": text, "metadata": metadata}
        rebuild_index()
        return jsonify({"status": "updated", "id": doc_id, "total": len(documents)})

    tokens = tokenize(text)
    for word in tokens:
        if word not in vocab and len(vocab) < EMBEDDING_DIM:
            vocab[word] = len(vocab)
        if word not in idf_values:
            idf_values[word] = 1.0

    embedding = text_to_embedding(text)
    index.add_items(np.array([embedding]), [current_idx])

    documents[doc_id] = {"text": text, "metadata": metadata, "idx": current_idx}
    idx_to_doc_id[current_idx] = doc_id
    current_idx += 1
    save_data()

    return jsonify({"status": "success", "id": doc_id, "total": len(documents)})


@app.route('/search', methods=['POST'])
def search():
    data = request.json
    if not data or 'query' not in data:
        return jsonify({"error": "query 필드가 필요합니다"}), 400

    if not documents:
        return jsonify({"query": data['query'], "results": [], "count": 0})

    query = data['query']
    n_results = min(data.get('n', 3), len(documents))

    query_embedding = text_to_embedding(query)
    labels, distances = index.knn_query(np.array([query_embedding]), k=n_results)

    results = []
    for label, distance in zip(labels[0], distances[0]):
        doc_id = idx_to_doc_id.get(label)
        if doc_id and doc_id in documents:
            doc = documents[doc_id]
            results.append({
                "id": doc_id,
                "text": doc['text'],
                "similarity": round(1 - distance, 4),
                "metadata": doc.get('metadata', {})
            })

    return jsonify({"query": query, "results": results, "count": len(results)})


@app.route('/list', methods=['GET'])
def list_documents():
    limit = request.args.get('limit', 100, type=int)
    doc_list = []
    for i, (doc_id, doc) in enumerate(documents.items()):
        if i >= limit:
            break
        doc_list.append({
            "id": doc_id,
            "text": doc['text'][:200] + "..." if len(doc['text']) > 200 else doc['text'],
            "metadata": doc.get('metadata', {})
        })
    return jsonify({"total": len(documents), "showing": len(doc_list), "documents": doc_list})


@app.route('/delete', methods=['DELETE'])
def delete_document():
    data = request.json
    if not data or 'id' not in data:
        return jsonify({"error": "id 필드가 필요합니다"}), 400

    doc_id = data['id']
    if doc_id not in documents:
        return jsonify({"error": "문서를 찾을 수 없습니다"}), 404

    del documents[doc_id]
    rebuild_index()
    return jsonify({"status": "success", "deleted": doc_id, "remaining": len(documents)})


@app.route('/clear', methods=['DELETE'])
def clear_all():
    global documents, idx_to_doc_id, current_idx, vocab, idf_values
    documents = {}
    idx_to_doc_id = {}
    current_idx = 0
    vocab = {}
    idf_values = {}
    init_index()
    if os.path.exists(INDEX_FILE):
        os.remove(INDEX_FILE)
    if os.path.exists(DOCS_FILE):
        os.remove(DOCS_FILE)
    return jsonify({"status": "success", "message": "모든 문서가 삭제되었습니다"})


if __name__ == '__main__':
    print("\n" + "="*50)
    print("🎤 Voice RAG System")
    print("="*50)
    print(f"문서 수: {len(documents)}")
    print(f"웹 UI: http://localhost:5000/ui")
    print(f"API: http://localhost:5000/")
    print("="*50 + "\n")

    app.run(host='0.0.0.0', port=5000, debug=False)
