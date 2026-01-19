#!/usr/bin/env python3
"""
Voice RAG + GPT 통합 시스템 (All-in-One)
하나의 파일로 RAG + GPT + 음성 채팅 모두 실행

사용법:
    python app.py

브라우저:
    http://localhost:5001
"""

from flask import Flask, request, jsonify, render_template_string
import numpy as np
import json
import os
import re
from collections import Counter
import math
import requests
import hnswlib

app = Flask(__name__)

# ===== 설정 =====
# .env 파일에서 API 키 자동 로드
def load_env():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ.setdefault(key.strip(), value.strip())

load_env()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

# RAG 설정
EMBEDDING_DIM = 100
MAX_ELEMENTS = 10000
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_FILE = os.path.join(DATA_DIR, "rag_index.bin")
DOCS_FILE = os.path.join(DATA_DIR, "rag_docs.json")

# ===== RAG 전역 변수 =====
index = None
documents = {}
idx_to_doc_id = {}
current_idx = 0
vocab = {}
idf_values = {}


# ===== RAG 텍스트 처리 =====
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


# ===== RAG 인덱스 관리 =====
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


# ===== RAG 검색 =====
def rag_search(query, n=3):
    """RAG에서 관련 문서 검색"""
    if not documents:
        return []
    
    n_results = min(n, len(documents))
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
    return results


# ===== OpenAI GPT =====
def ask_openai(question, context_docs):
    """OpenAI API 호출"""
    if not OPENAI_API_KEY:
        return "⚠️ OpenAI API 키가 설정되지 않았습니다.\n\n.env 파일에 OPENAI_API_KEY를 설정해주세요."
    
    # 프롬프트 생성
    if context_docs:
        context = "\n\n".join([
            f"[문서 {i+1}] (유사도: {doc['similarity']*100:.1f}%)\n{doc['text']}"
            for i, doc in enumerate(context_docs)
        ])
        
        system_prompt = """당신은 RAG 기반 AI 어시스턴트입니다.
사용자의 질문에 대해 제공된 문서를 참고하여 답변하세요.
문서에 없는 내용은 모른다고 말하세요.
답변은 친절하고 자연스럽게 한국어로 해주세요."""

        user_prompt = f"""=== 관련 문서 ===
{context}

=== 사용자 질문 ===
{question}

위 문서들을 참고하여 질문에 답변해주세요."""
    else:
        system_prompt = "당신은 친절한 AI 어시스턴트입니다. 한국어로 답변해주세요."
        user_prompt = f"질문: {question}\n\n저장된 문서가 없습니다. 일반적인 답변을 해주세요."

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": OPENAI_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "max_tokens": 1024,
                "temperature": 0.7
            },
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        elif response.status_code == 401:
            return "⚠️ OpenAI API 키가 유효하지 않습니다."
        elif response.status_code == 429:
            return "⚠️ API 호출 한도 초과. 잠시 후 다시 시도해주세요."
        else:
            return f"⚠️ API 오류: {response.status_code}"
            
    except Exception as e:
        return f"⚠️ 오류: {str(e)}"


# ===== 웹 UI =====
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>🤖 Voice RAG + GPT</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            padding: 15px;
        }
        .container {
            max-width: 500px;
            margin: 0 auto;
            background: #0f0f23;
            border-radius: 20px;
            padding: 20px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.5);
            border: 1px solid #333;
        }
        h1 {
            text-align: center;
            color: #10a37f;
            margin-bottom: 8px;
            font-size: 1.4rem;
        }
        .subtitle {
            text-align: center;
            color: #888;
            margin-bottom: 15px;
            font-size: 0.85rem;
        }
        .status-bar {
            display: flex;
            justify-content: center;
            gap: 15px;
            margin-bottom: 15px;
            font-size: 0.8rem;
        }
        .status-item {
            display: flex;
            align-items: center;
            gap: 5px;
            color: #888;
        }
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #ff4757;
        }
        .status-dot.ok { background: #2ed573; }
        .chat-box {
            background: #1a1a2e;
            border-radius: 15px;
            padding: 12px;
            height: 45vh;
            min-height: 280px;
            overflow-y: auto;
            margin-bottom: 15px;
            border: 1px solid #333;
        }
        .message {
            margin-bottom: 12px;
            padding: 10px 14px;
            border-radius: 15px;
            max-width: 88%;
            line-height: 1.5;
            font-size: 0.95rem;
            word-wrap: break-word;
        }
        .user-msg {
            background: linear-gradient(135deg, #10a37f, #1a7f5a);
            color: white;
            margin-left: auto;
        }
        .bot-msg {
            background: #2a2a4a;
            color: #e0e0e0;
            border: 1px solid #444;
        }
        .bot-msg .sources {
            margin-top: 10px;
            padding-top: 8px;
            border-top: 1px solid #444;
            font-size: 0.75rem;
            color: #888;
        }
        .bot-msg .actions {
            margin-top: 8px;
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }
        .action-btn {
            background: none;
            border: 1px solid #10a37f;
            color: #10a37f;
            padding: 4px 10px;
            border-radius: 12px;
            cursor: pointer;
            font-size: 0.75rem;
        }
        .action-btn:hover {
            background: #10a37f22;
        }
        .input-area {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        .input-row {
            display: flex;
            gap: 8px;
            align-items: center;
        }
        .voice-btn {
            width: 50px;
            height: 50px;
            border-radius: 50%;
            border: none;
            background: linear-gradient(135deg, #10a37f, #1a7f5a);
            color: white;
            font-size: 1.3rem;
            cursor: pointer;
            flex-shrink: 0;
            transition: all 0.3s;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .voice-btn:hover { transform: scale(1.05); }
        .voice-btn:active { transform: scale(0.95); }
        .voice-btn.recording {
            background: linear-gradient(135deg, #ff4757, #ff6b81);
            animation: pulse 1s infinite;
        }
        @keyframes pulse {
            0%, 100% { box-shadow: 0 0 0 0 rgba(255, 71, 87, 0.4); }
            50% { box-shadow: 0 0 0 12px rgba(255, 71, 87, 0); }
        }
        input[type="text"] {
            flex: 1;
            padding: 12px 14px;
            border: 2px solid #333;
            border-radius: 25px;
            background: #1a1a2e;
            color: white;
            font-size: 1rem;
            min-width: 0;
        }
        input:focus {
            outline: none;
            border-color: #10a37f;
        }
        .send-btn {
            width: 100%;
            padding: 14px 20px;
            border: none;
            border-radius: 25px;
            background: linear-gradient(135deg, #10a37f, #1a7f5a);
            color: white;
            font-weight: bold;
            cursor: pointer;
            font-size: 1rem;
        }
        .send-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .status-text {
            text-align: center;
            color: #10a37f;
            margin: 10px 0;
            min-height: 20px;
            font-size: 0.85rem;
        }
        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 2px solid #10a37f;
            border-radius: 50%;
            border-top-color: transparent;
            animation: spin 1s linear infinite;
            margin-right: 8px;
            vertical-align: middle;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        .tabs {
            display: flex;
            margin-bottom: 15px;
            border-radius: 10px;
            overflow: hidden;
            border: 2px solid #10a37f;
        }
        .tab {
            flex: 1;
            padding: 10px;
            text-align: center;
            cursor: pointer;
            background: transparent;
            color: #10a37f;
            font-weight: bold;
            font-size: 0.85rem;
            border: none;
            transition: all 0.3s;
        }
        .tab.active {
            background: #10a37f;
            color: white;
        }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        .doc-input {
            width: 100%;
            padding: 12px;
            border: 1px solid #333;
            border-radius: 10px;
            background: #1a1a2e;
            color: white;
            margin-bottom: 10px;
            font-size: 0.9rem;
        }
        textarea.doc-input {
            min-height: 100px;
            resize: vertical;
        }
        .btn {
            padding: 12px 20px;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            font-weight: bold;
            font-size: 0.9rem;
            margin-right: 8px;
            margin-bottom: 8px;
        }
        .btn-primary { background: #10a37f; color: white; }
        .btn-danger { background: #ff4757; color: white; }
        .btn-secondary { background: #333; color: white; }
        .doc-list {
            max-height: 250px;
            overflow-y: auto;
            margin-top: 15px;
        }
        .doc-item {
            background: #1a1a2e;
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 8px;
            border-left: 3px solid #10a37f;
        }
        .doc-item .doc-id { color: #10a37f; font-size: 0.8rem; font-weight: bold; }
        .doc-item .doc-text { color: #ccc; font-size: 0.85rem; margin-top: 5px; }
        .doc-item .doc-actions { margin-top: 8px; }
        .doc-item .btn { padding: 5px 10px; font-size: 0.75rem; }
        details.settings {
            margin-top: 15px;
            padding: 12px;
            background: #1a1a2e;
            border-radius: 10px;
            border: 1px solid #333;
        }
        details.settings summary {
            color: #888;
            cursor: pointer;
            font-size: 0.85rem;
        }
        details.settings label {
            display: block;
            color: #888;
            margin: 10px 0 5px;
            font-size: 0.8rem;
        }
        details.settings select {
            width: 100%;
            padding: 8px;
            border-radius: 8px;
            border: 1px solid #333;
            background: #0f0f23;
            color: white;
            font-size: 0.9rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Voice RAG + GPT</h1>
        <p class="subtitle">음성으로 질문하면 AI가 문서를 검색해서 답변해요</p>
        
        <div class="status-bar">
            <div class="status-item">
                <div class="status-dot ok"></div>
                <span>서버</span>
            </div>
            <div class="status-item">
                <div class="status-dot" id="llmDot"></div>
                <span>GPT</span>
            </div>
            <div class="status-item">
                📚 <span id="docCount">0</span>개 문서
            </div>
        </div>
        
        <div class="tabs">
            <button class="tab active" onclick="showTab('chat')">💬 채팅</button>
            <button class="tab" onclick="showTab('docs')">📄 문서관리</button>
        </div>
        
        <!-- 채팅 탭 -->
        <div id="chatTab" class="tab-content active">
            <div class="chat-box" id="chatBox">
                <div class="message bot-msg">
                    안녕하세요! 저는 RAG 기반 GPT 어시스턴트예요. 🤖<br><br>
                    📄 문서관리 탭에서 문서를 추가하고,<br>
                    🎤 버튼을 눌러 음성으로 질문하세요!
                </div>
            </div>
            
            <p class="status-text" id="status"></p>
            
            <div class="input-area">
                <div class="input-row">
                    <button class="voice-btn" id="voiceBtn" onclick="toggleVoice()">🎤</button>
                    <input type="text" id="userInput" placeholder="질문을 입력하세요...">
                </div>
                <button class="send-btn" id="sendBtn" onclick="sendMessage()">전송</button>
            </div>
        </div>
        
        <!-- 문서관리 탭 -->
        <div id="docsTab" class="tab-content">
            <input type="text" class="doc-input" id="docId" placeholder="문서 ID (선택사항)">
            <textarea class="doc-input" id="docText" placeholder="문서 내용을 입력하세요..."></textarea>
            <button class="btn btn-primary" onclick="addDoc()">➕ 문서 추가</button>
            <button class="btn btn-secondary" onclick="loadDocs()">🔄 새로고침</button>
            <button class="btn btn-danger" onclick="clearDocs()">🗑️ 전체삭제</button>
            
            <div class="doc-list" id="docList"></div>
        </div>
        
        <details class="settings">
            <summary>⚙️ 설정</summary>
            <label>검색 결과 수:</label>
            <select id="numResults">
                <option value="2">2개</option>
                <option value="3" selected>3개</option>
                <option value="5">5개</option>
            </select>
            <label>음성 자동 읽기:</label>
            <select id="autoSpeak">
                <option value="true" selected>켜기</option>
                <option value="false">끄기</option>
            </select>
            <label>음성 속도:</label>
            <select id="speechRate">
                <option value="0.8">느리게</option>
                <option value="1.0" selected>보통</option>
                <option value="1.2">빠르게</option>
            </select>
        </details>
    </div>

    <script>
        let recognition = null;
        let isRecording = false;
        let isProcessing = false;
        
        // 초기화
        checkHealth();
        loadDocs();
        
        // 음성 인식 초기화
        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            recognition = new SpeechRecognition();
            recognition.continuous = false;
            recognition.interimResults = true;
            recognition.lang = 'ko-KR';
            
            recognition.onresult = (event) => {
                const transcript = event.results[0][0].transcript;
                document.getElementById('userInput').value = transcript;
                if (event.results[0].isFinal) {
                    document.getElementById('status').textContent = '✅ 인식 완료';
                } else {
                    document.getElementById('status').textContent = '🎤 ' + transcript;
                }
            };
            
            recognition.onend = () => {
                isRecording = false;
                document.getElementById('voiceBtn').classList.remove('recording');
                const input = document.getElementById('userInput').value.trim();
                if (input && !isProcessing) {
                    sendMessage();
                }
            };
            
            recognition.onerror = (event) => {
                let errorMsg = '음성 인식 오류';
                if (event.error === 'not-allowed') errorMsg = '마이크 권한을 허용해주세요';
                else if (event.error === 'no-speech') errorMsg = '음성이 감지되지 않았어요';
                document.getElementById('status').textContent = '❌ ' + errorMsg;
                isRecording = false;
                document.getElementById('voiceBtn').classList.remove('recording');
            };
        }
        
        function showTab(tabName) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            event.target.classList.add('active');
            document.getElementById(tabName + 'Tab').classList.add('active');
            if (tabName === 'docs') loadDocs();
        }
        
        async function checkHealth() {
            try {
                const res = await fetch('/health');
                const data = await res.json();
                document.getElementById('llmDot').classList.toggle('ok', data.llm_available);
                document.getElementById('docCount').textContent = data.documents || 0;
            } catch (e) {}
        }
        
        function toggleVoice() {
            if (isProcessing) return;
            if (!recognition) {
                alert('이 브라우저는 음성 인식을 지원하지 않습니다.\\nChrome 브라우저를 사용해주세요.');
                return;
            }
            
            if (isRecording) {
                recognition.stop();
            } else {
                recognition.start();
                isRecording = true;
                document.getElementById('voiceBtn').classList.add('recording');
                document.getElementById('status').innerHTML = '🎤 듣고 있어요...';
            }
        }
        
        function speak(text) {
            if (!('speechSynthesis' in window)) return;
            if (document.getElementById('autoSpeak').value !== 'true') return;
            speechSynthesis.cancel();
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.lang = 'ko-KR';
            utterance.rate = parseFloat(document.getElementById('speechRate').value);
            speechSynthesis.speak(utterance);
        }
        
        function stopSpeaking() {
            if ('speechSynthesis' in window) speechSynthesis.cancel();
        }
        
        function addMessage(text, isUser, sources = null) {
            const chatBox = document.getElementById('chatBox');
            const msgDiv = document.createElement('div');
            msgDiv.className = 'message ' + (isUser ? 'user-msg' : 'bot-msg');
            
            let html = text.replace(/\\n/g, '<br>');
            
            if (!isUser) {
                if (sources && sources.length > 0) {
                    html += '<div class="sources">📚 참고: ' + sources.map(s => s.id).join(', ') + '</div>';
                }
                const safeText = text.replace(/`/g, "'").replace(/\\/g, "\\\\");
                html += '<div class="actions">';
                html += '<button class="action-btn" onclick="speak(`' + safeText + '`)">🔊 읽기</button>';
                html += '<button class="action-btn" onclick="stopSpeaking()">⏹️ 중지</button>';
                html += '</div>';
            }
            
            msgDiv.innerHTML = html;
            chatBox.appendChild(msgDiv);
            chatBox.scrollTop = chatBox.scrollHeight;
        }
        
        async function sendMessage() {
            const input = document.getElementById('userInput');
            const question = input.value.trim();
            
            console.log('sendMessage 호출됨, 질문:', question);
            
            if (!question || isProcessing) {
                console.log('질문 없음 또는 처리 중');
                return;
            }
            
            isProcessing = true;
            document.getElementById('sendBtn').disabled = true;
            document.getElementById('voiceBtn').disabled = true;
            
            addMessage(question, true);
            input.value = '';
            document.getElementById('status').innerHTML = '<span class="loading"></span>답변 생성 중...';
            
            try {
                console.log('API 호출 시작');
                const numResults = document.getElementById('numResults').value;
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ question: question, n_results: parseInt(numResults) })
                });
                
                console.log('API 응답:', response.status);
                
                const data = await response.json();
                console.log('응답 데이터:', data);
                
                document.getElementById('status').textContent = '';
                addMessage(data.answer, false, data.sources);
                speak(data.answer);
            } catch (error) {
                console.log('오류 발생:', error);
                document.getElementById('status').textContent = '';
                addMessage('오류가 발생했어요: ' + error.message, false);
            } finally {
                isProcessing = false;
                document.getElementById('sendBtn').disabled = false;
                document.getElementById('voiceBtn').disabled = false;
            }
        }
        
        async function addDoc() {
            const text = document.getElementById('docText').value.trim();
            if (!text) { alert('문서 내용을 입력하세요'); return; }
            const id = document.getElementById('docId').value.trim() || undefined;
            
            const res = await fetch('/add', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ text, id })
            });
            const data = await res.json();
            alert('문서 추가됨: ' + data.id);
            document.getElementById('docText').value = '';
            document.getElementById('docId').value = '';
            loadDocs();
            checkHealth();
        }
        
        async function loadDocs() {
            const res = await fetch('/list?limit=50');
            const data = await res.json();
            document.getElementById('docCount').textContent = data.total;
            
            let html = '';
            if (data.documents && data.documents.length > 0) {
                data.documents.forEach(doc => {
                    html += '<div class="doc-item">';
                    html += '<div class="doc-id">🏷️ ' + doc.id + '</div>';
                    html += '<div class="doc-text">' + doc.text + '</div>';
                    html += '<div class="doc-actions">';
                    html += '<button class="btn btn-danger" onclick="deleteDoc(\\'' + doc.id + '\\')">삭제</button>';
                    html += '</div></div>';
                });
            } else {
                html = '<p style="color:#888;text-align:center;padding:20px;">저장된 문서가 없습니다</p>';
            }
            document.getElementById('docList').innerHTML = html;
        }
        
        async function deleteDoc(id) {
            if (!confirm('삭제할까요?')) return;
            await fetch('/delete', {
                method: 'DELETE',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ id })
            });
            loadDocs();
            checkHealth();
        }
        
        async function clearDocs() {
            if (!confirm('모든 문서를 삭제할까요?')) return;
            await fetch('/clear', { method: 'DELETE' });
            loadDocs();
            checkHealth();
        }
        
        document.getElementById('userInput').addEventListener('keypress', function(e) {
            console.log('키 입력:', e.key);
            if (e.key === 'Enter') {
                e.preventDefault();
                sendMessage();
            }
        });
        
        document.getElementById('sendBtn').addEventListener('click', function() {
            console.log('전송 버튼 클릭됨');
            sendMessage();
        });
    </script>
</body>
</html>
'''


# ===== API 엔드포인트 =====
@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)


@app.route('/chat', methods=['POST'])
def chat():
    """채팅 API"""
    data = request.json
    question = data.get('question', '')
    n_results = data.get('n_results', 3)
    
    if not question:
        return jsonify({"error": "질문을 입력해주세요"}), 400
    
    sources = rag_search(question, n=n_results)
    answer = ask_openai(question, sources)
    
    return jsonify({
        "question": question,
        "answer": answer,
        "sources": sources
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
            "text": doc['text'][:200] + "..." if len(doc['text']) > 200 else doc['text'],
            "metadata": doc.get('metadata', {})
        })
    return jsonify({"total": len(documents), "showing": len(doc_list), "documents": doc_list})


@app.route('/delete', methods=['DELETE'])
def delete_document():
    """문서 삭제"""
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
    """전체 삭제"""
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


@app.route('/health')
def health():
    """서버 상태"""
    return jsonify({
        "status": "running",
        "documents": len(documents),
        "llm_available": bool(OPENAI_API_KEY),
        "model": OPENAI_MODEL
    })


# ===== 시작 =====
load_data()

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🤖 Voice RAG + GPT (All-in-One)")
    print("="*50)
    print(f"🌐 웹 UI: http://localhost:5001")
    print(f"📚 문서 수: {len(documents)}")
    print(f"🤖 모델: {OPENAI_MODEL}")
    
    if OPENAI_API_KEY:
        print("✅ OpenAI API 키 설정됨")
    else:
        print("⚠️  OpenAI API 키가 없습니다!")
        print("   .env 파일에 OPENAI_API_KEY를 설정하세요")
    
    print("="*50)
    print("\n🚀 서버 시작! 브라우저에서 http://localhost:5001 열기\n")
    
    app.run(host='0.0.0.0', port=5001, debug=False)
