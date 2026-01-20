#!/usr/bin/env python3
"""
Voice RAG + Claude 통합 시스템 (웹 검색 기능 포함)
하나의 파일로 RAG + Claude + 웹검색 + 음성 채팅 모두 실행

주요 기능:
    1. RAG: 저장된 문서에서 관련 정보 검색
    2. 웹 검색: 인터넷에서 최신 정보 검색 (2024년 이후 정보도 가능!)
    3. Claude AI: 자연어 답변 생성

사용법:
    python app_claude_websearch.py

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

# Claude API 설정
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-20250514")

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


# ===== 웹 검색이 필요한지 판단 =====
def needs_web_search(question):
    """
    질문이 웹 검색이 필요한지 판단합니다.
    
    웹 검색이 필요한 경우:
    - 최신 정보를 묻는 경우 (오늘, 현재, 최근, 2024년, 2025년 등)
    - 뉴스, 날씨, 주가 등 실시간 정보
    - "검색해줘", "찾아줘" 등의 명시적 요청
    """
    # 웹 검색을 명시적으로 요청하는 키워드
    explicit_keywords = [
        '검색', '찾아', '알아봐', '조사해', '인터넷',
        'search', 'find', 'look up', 'google'
    ]
    
    # 최신 정보가 필요한 키워드
    time_keywords = [
        '오늘', '현재', '지금', '최근', '요즘', '올해',
        '2024', '2025', '2026',
        '뉴스', '날씨', '주가', '환율', '주식',
        'today', 'now', 'current', 'recent', 'latest', 'news'
    ]
    
    question_lower = question.lower()
    
    # 명시적 요청 확인
    for keyword in explicit_keywords:
        if keyword in question_lower:
            return True
    
    # 최신 정보 키워드 확인
    for keyword in time_keywords:
        if keyword in question_lower:
            return True
    
    return False


# ===== Claude API (웹 검색 포함) =====
def ask_claude_with_web_search(question, context_docs, use_web_search=False):
    """
    Claude API 호출 (웹 검색 도구 지원)
    
    핵심 변경점:
    - tools 파라미터에 web_search 도구 추가
    - Claude가 필요시 자동으로 웹 검색 수행
    """
    if not ANTHROPIC_API_KEY:
        return "⚠️ Anthropic API 키가 설정되지 않았습니다.\n\n.env 파일에 ANTHROPIC_API_KEY를 설정해주세요.", []
    
    # 시스템 프롬프트 설정
    if context_docs:
        context = "\n\n".join([
            f"[문서 {i+1}] (유사도: {doc['similarity']*100:.1f}%)\n{doc['text']}"
            for i, doc in enumerate(context_docs)
        ])
        
        system_prompt = """당신은 RAG 기반 AI 어시스턴트입니다.
사용자의 질문에 대해 제공된 문서를 참고하여 답변하세요.

중요: 문서에 없는 최신 정보나 실시간 정보가 필요한 경우, 
web_search 도구를 사용하여 인터넷에서 검색하세요.

답변은 친절하고 자연스럽게 한국어로 해주세요.
웹 검색을 사용한 경우, 출처를 간략히 언급해주세요."""

        user_prompt = f"""=== 관련 문서 ===
{context}

=== 사용자 질문 ===
{question}

위 문서들을 참고하여 질문에 답변해주세요.
문서에 없는 최신 정보가 필요하면 웹 검색을 활용하세요."""
    else:
        system_prompt = """당신은 친절한 AI 어시스턴트입니다. 
한국어로 답변해주세요.

최신 정보나 실시간 정보가 필요한 경우, 
web_search 도구를 사용하여 인터넷에서 검색하세요.
웹 검색을 사용한 경우, 출처를 간략히 언급해주세요."""
        
        user_prompt = f"질문: {question}"

    # 웹 검색 도구 정의
    # 이것이 핵심입니다! Anthropic API의 web_search 도구를 사용합니다.
    tools = [
        {
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": 5  # 한 번의 대화에서 최대 5번까지 검색 가능
        }
    ]
    
    try:
        # API 요청 구성
        request_body = {
            "model": CLAUDE_MODEL,
            "max_tokens": 4096,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": user_prompt}
            ]
        }
        
        # 웹 검색 사용 시 tools 추가
        if use_web_search:
            request_body["tools"] = tools
        
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json"
            },
            json=request_body,
            timeout=120  # 웹 검색은 시간이 더 걸릴 수 있음
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # 응답에서 텍스트와 웹 검색 결과 추출
            answer_text = ""
            web_sources = []
            
            for block in data.get("content", []):
                if block.get("type") == "text":
                    answer_text += block.get("text", "")
                elif block.get("type") == "web_search_tool_result":
                    # 웹 검색 결과 처리
                    search_results = block.get("content", [])
                    for result in search_results:
                        if result.get("type") == "web_search_result":
                            web_sources.append({
                                "title": result.get("title", ""),
                                "url": result.get("url", ""),
                                "snippet": result.get("encrypted_content", "")[:200] if result.get("encrypted_content") else ""
                            })
            
            return answer_text, web_sources
            
        elif response.status_code == 401:
            return "⚠️ Anthropic API 키가 유효하지 않습니다.", []
        elif response.status_code == 429:
            return "⚠️ API 호출 한도 초과. 잠시 후 다시 시도해주세요.", []
        else:
            error_msg = response.json().get("error", {}).get("message", "알 수 없는 오류")
            return f"⚠️ API 오류: {error_msg}", []
            
    except requests.exceptions.Timeout:
        return "⚠️ API 응답 시간 초과. 다시 시도해주세요.", []
    except Exception as e:
        return f"⚠️ 오류 발생: {str(e)}", []


# ===== 기존 Claude API (웹 검색 없이) =====
def ask_claude(question, context_docs):
    """기존 방식의 Claude API 호출 (웹 검색 없음)"""
    answer, _ = ask_claude_with_web_search(question, context_docs, use_web_search=False)
    return answer


# ===== HTML 템플릿 (웹 검색 UI 추가) =====
MOBILE_APP_HTML = '''
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>🔍 RAG + 웹검색 Claude</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 10px;
        }
        .container { max-width: 800px; margin: 0 auto; }
        .header { 
            text-align: center; 
            padding: 15px; 
            color: white;
            margin-bottom: 10px;
        }
        .header h1 { font-size: 1.5rem; margin-bottom: 5px; }
        .header p { font-size: 0.85rem; opacity: 0.9; }
        .status-bar {
            display: flex;
            justify-content: center;
            gap: 15px;
            margin-top: 10px;
            flex-wrap: wrap;
        }
        .status-item {
            background: rgba(255,255,255,0.2);
            padding: 5px 12px;
            border-radius: 15px;
            font-size: 0.75rem;
        }
        .tabs {
            display: flex;
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
            padding: 5px;
            margin-bottom: 10px;
        }
        .tab {
            flex: 1;
            padding: 10px;
            text-align: center;
            color: white;
            border-radius: 8px;
            cursor: pointer;
            font-size: 0.9rem;
        }
        .tab.active { background: white; color: #667eea; }
        .panel { 
            display: none; 
            background: white; 
            border-radius: 15px; 
            padding: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }
        .panel.active { display: block; }
        
        /* 채팅 패널 */
        #chatContainer {
            height: 350px;
            overflow-y: auto;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 10px;
            margin-bottom: 10px;
        }
        .message {
            margin: 8px 0;
            padding: 10px 14px;
            border-radius: 18px;
            max-width: 85%;
            font-size: 0.9rem;
            line-height: 1.4;
        }
        .message.user {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            margin-left: auto;
            border-bottom-right-radius: 5px;
        }
        .message.bot {
            background: white;
            border: 1px solid #e0e0e0;
            border-bottom-left-radius: 5px;
        }
        .message.typing span {
            display: inline-block;
            width: 8px;
            height: 8px;
            background: #667eea;
            border-radius: 50%;
            margin: 0 2px;
            animation: bounce 1.4s infinite;
        }
        .message.typing span:nth-child(2) { animation-delay: 0.2s; }
        .message.typing span:nth-child(3) { animation-delay: 0.4s; }
        @keyframes bounce {
            0%, 60%, 100% { transform: translateY(0); }
            30% { transform: translateY(-8px); }
        }
        .sources {
            margin-top: 8px;
            padding-top: 8px;
            border-top: 1px dashed #ddd;
            font-size: 0.75rem;
            color: #666;
        }
        .sources a {
            color: #667eea;
            text-decoration: none;
        }
        .sources a:hover {
            text-decoration: underline;
        }
        .web-source {
            background: #e8f4fd;
            padding: 5px 8px;
            border-radius: 5px;
            margin: 3px 0;
        }
        .input-area {
            display: flex;
            gap: 8px;
            align-items: center;
        }
        #userInput {
            flex: 1;
            padding: 12px 15px;
            border: 2px solid #e0e0e0;
            border-radius: 25px;
            font-size: 0.95rem;
            outline: none;
        }
        #userInput:focus { border-color: #667eea; }
        .btn {
            padding: 12px 20px;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-size: 0.9rem;
            font-weight: 600;
        }
        .btn-primary {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
        }
        .btn-primary:disabled { opacity: 0.5; }
        
        /* 검색 옵션 */
        .search-options {
            display: flex;
            gap: 10px;
            margin-bottom: 10px;
            padding: 10px;
            background: #f0f4ff;
            border-radius: 10px;
            align-items: center;
            flex-wrap: wrap;
        }
        .search-options label {
            font-size: 0.85rem;
            color: #555;
        }
        .search-options select, .search-options input[type="checkbox"] {
            padding: 5px;
            border-radius: 5px;
            border: 1px solid #ddd;
        }
        .toggle-switch {
            position: relative;
            width: 50px;
            height: 26px;
        }
        .toggle-switch input {
            opacity: 0;
            width: 0;
            height: 0;
        }
        .toggle-slider {
            position: absolute;
            cursor: pointer;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: #ccc;
            transition: 0.4s;
            border-radius: 26px;
        }
        .toggle-slider:before {
            position: absolute;
            content: "";
            height: 20px;
            width: 20px;
            left: 3px;
            bottom: 3px;
            background-color: white;
            transition: 0.4s;
            border-radius: 50%;
        }
        input:checked + .toggle-slider {
            background: linear-gradient(135deg, #667eea, #764ba2);
        }
        input:checked + .toggle-slider:before {
            transform: translateX(24px);
        }
        
        /* 문서 관리 패널 */
        .doc-form { margin-bottom: 15px; }
        .doc-form textarea {
            width: 100%;
            padding: 10px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 0.9rem;
            resize: vertical;
            min-height: 80px;
        }
        .doc-form input {
            width: 100%;
            padding: 10px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            margin-top: 8px;
            font-size: 0.9rem;
        }
        .btn-group {
            display: flex;
            gap: 8px;
            margin-top: 10px;
        }
        .btn-secondary {
            background: #f0f0f0;
            color: #333;
        }
        .btn-danger {
            background: #ff4757;
            color: white;
        }
        #docList {
            max-height: 300px;
            overflow-y: auto;
        }
        .doc-item {
            padding: 10px;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            margin-bottom: 8px;
            background: #fafafa;
        }
        .doc-item-id {
            font-weight: 600;
            color: #667eea;
            font-size: 0.8rem;
        }
        .doc-item-text {
            font-size: 0.85rem;
            color: #666;
            margin: 5px 0;
        }
        .doc-item button {
            padding: 5px 10px;
            font-size: 0.75rem;
            border: none;
            background: #ff4757;
            color: white;
            border-radius: 5px;
            cursor: pointer;
        }
        .empty-state {
            text-align: center;
            padding: 30px;
            color: #999;
        }
        .empty-state .icon { font-size: 3rem; margin-bottom: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 RAG + 웹검색 Claude</h1>
            <p>저장된 문서 + 인터넷 검색으로 최신 정보까지!</p>
            <div class="status-bar">
                <span class="status-item">📚 문서: <span id="docCount">0</span>개</span>
                <span class="status-item">🧠 <span id="modelName">Claude</span></span>
                <span class="status-item" id="apiStatus">⏳ 확인중</span>
            </div>
        </div>
        
        <div class="tabs">
            <div class="tab active" onclick="showTab('chat')">💬 채팅</div>
            <div class="tab" onclick="showTab('docs')">📚 문서관리</div>
        </div>
        
        <div id="chatPanel" class="panel active">
            <div class="search-options">
                <label>📄 RAG 결과:</label>
                <select id="numResults">
                    <option value="0">사용안함</option>
                    <option value="1">1개</option>
                    <option value="3" selected>3개</option>
                    <option value="5">5개</option>
                </select>
                
                <label style="margin-left: 15px;">🌐 웹검색:</label>
                <label class="toggle-switch">
                    <input type="checkbox" id="webSearchToggle" checked>
                    <span class="toggle-slider"></span>
                </label>
            </div>
            
            <div id="chatContainer">
                <div class="message bot">
                    안녕하세요! 🔍 RAG와 웹검색을 지원하는 Claude입니다.<br><br>
                    💡 <b>웹검색 ON</b>: 최신 뉴스, 날씨, 주가 등 실시간 정보 검색<br>
                    📚 <b>RAG</b>: 저장된 문서에서 관련 정보 검색<br><br>
                    무엇이든 물어보세요!
                </div>
            </div>
            <div class="input-area">
                <input type="text" id="userInput" placeholder="메시지를 입력하세요...">
                <button id="sendBtn" class="btn btn-primary" onclick="sendMessage()">전송</button>
            </div>
        </div>
        
        <div id="docsPanel" class="panel">
            <div class="doc-form">
                <textarea id="docText" placeholder="저장할 문서 내용을 입력하세요..."></textarea>
                <input type="text" id="docId" placeholder="문서 ID (선택사항)">
                <div class="btn-group">
                    <button class="btn btn-primary" onclick="addDoc()">➕ 문서 추가</button>
                    <button class="btn btn-secondary" onclick="loadDocs()">🔄 새로고침</button>
                    <button class="btn btn-danger" onclick="clearDocs()">🗑️ 전체삭제</button>
                </div>
            </div>
            <div id="docList">
                <div class="empty-state">
                    <div class="icon">📄</div>
                    <p>문서가 없습니다</p>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let isProcessing = false;
        let speechSynth = window.speechSynthesis;
        
        function showTab(tab) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
            document.querySelector(`.tab:nth-child(${tab === 'chat' ? 1 : 2})`).classList.add('active');
            document.getElementById(tab + 'Panel').classList.add('active');
            if (tab === 'docs') loadDocs();
        }
        
        async function checkHealth() {
            try {
                const res = await fetch('/health');
                const data = await res.json();
                document.getElementById('docCount').textContent = data.documents;
                document.getElementById('modelName').textContent = data.model.split('-').slice(0,2).join('-');
                document.getElementById('apiStatus').textContent = data.llm_available ? '✅ API 연결됨' : '❌ API 키 없음';
            } catch(e) {
                document.getElementById('apiStatus').textContent = '❌ 서버 오류';
            }
        }
        checkHealth();
        
        function speak(text) {
            // TTS 기능 (선택적)
            if (speechSynth && document.getElementById('ttsToggle')?.checked) {
                const clean = text.replace(/[*#_`]/g, '').replace(/\\n/g, ' ');
                const utter = new SpeechSynthesisUtterance(clean);
                utter.lang = 'ko-KR';
                utter.rate = 1.0;
                speechSynth.speak(utter);
            }
        }
        
        function formatMessage(text) {
            return text
                .replace(/\\n/g, '<br>')
                .replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>')
                .replace(/\\*(.+?)\\*/g, '<em>$1</em>')
                .replace(/`(.+?)`/g, '<code>$1</code>');
        }
        
        function addMsg(text, isUser, sources = [], webSources = []) {
            const c = document.getElementById('chatContainer');
            const d = document.createElement('div');
            d.className = 'message ' + (isUser ? 'user' : 'bot');
            
            let h = formatMessage(text);
            
            // RAG 소스 표시
            if (sources && sources.length > 0) {
                h += '<div class="sources">📚 <b>참고 문서:</b><br>';
                sources.forEach(s => {
                    h += `<span>• ${s.id} (유사도: ${(s.similarity*100).toFixed(1)}%)</span><br>`;
                });
                h += '</div>';
            }
            
            // 웹 검색 소스 표시
            if (webSources && webSources.length > 0) {
                h += '<div class="sources">🌐 <b>웹 검색 결과:</b><br>';
                webSources.forEach(s => {
                    h += `<div class="web-source">`;
                    h += `<a href="${s.url}" target="_blank">${s.title || s.url}</a>`;
                    h += `</div>`;
                });
                h += '</div>';
            }
            
            d.innerHTML = h;
            c.appendChild(d);
            c.scrollTop = c.scrollHeight;
        }
        
        function showTyping() {
            const c = document.getElementById('chatContainer');
            const d = document.createElement('div');
            d.className = 'message bot typing';
            d.id = 'typing';
            d.innerHTML = '<span></span><span></span><span></span>';
            c.appendChild(d);
            c.scrollTop = c.scrollHeight;
        }
        
        function hideTyping() {
            const t = document.getElementById('typing');
            if (t) t.remove();
        }
        
        async function sendMessage() {
            const input = document.getElementById('userInput');
            const q = input.value.trim();
            if (!q || isProcessing) return;
            
            isProcessing = true;
            document.getElementById('sendBtn').disabled = true;
            
            addMsg(q, true);
            input.value = '';
            showTyping();
            
            try {
                const webSearchEnabled = document.getElementById('webSearchToggle').checked;
                const numResults = parseInt(document.getElementById('numResults').value);
                
                const res = await fetch('/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        question: q, 
                        n_results: numResults,
                        use_web_search: webSearchEnabled
                    })
                });
                hideTyping();
                const data = await res.json();
                addMsg(data.answer, false, data.sources, data.web_sources);
                speak(data.answer);
                checkHealth();
            } catch(e) {
                hideTyping();
                addMsg('오류: ' + e.message, false);
            } finally {
                isProcessing = false;
                document.getElementById('sendBtn').disabled = false;
            }
        }
        
        async function loadDocs() {
            try {
                const res = await fetch('/list?limit=50');
                const data = await res.json();
                document.getElementById('docCount').textContent = data.total;
                const list = document.getElementById('docList');
                if (data.documents && data.documents.length) {
                    list.innerHTML = data.documents.map(d => 
                        '<div class="doc-item"><div class="doc-item-id">🏷️ '+d.id+'</div><div class="doc-item-text">'+d.text+'</div><button onclick="delDoc(\\''+d.id+'\\')">🗑️ 삭제</button></div>'
                    ).join('');
                } else {
                    list.innerHTML = '<div class="empty-state"><div class="icon">📄</div><p>문서가 없습니다</p></div>';
                }
            } catch(e) {
                document.getElementById('docList').innerHTML = '<div class="empty-state"><div class="icon">❌</div><p>로드 실패</p></div>';
            }
        }
        
        async function addDoc() {
            const text = document.getElementById('docText').value.trim();
            if (!text) { alert('내용을 입력하세요'); return; }
            const id = document.getElementById('docId').value.trim() || undefined;
            try {
                const res = await fetch('/add', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({text, id})
                });
                const data = await res.json();
                alert('추가됨: ' + data.id);
                document.getElementById('docText').value = '';
                document.getElementById('docId').value = '';
                loadDocs();
                checkHealth();
            } catch(e) { alert('추가 실패'); }
        }
        
        async function delDoc(id) {
            if (!confirm('삭제?')) return;
            await fetch('/delete', {method:'DELETE', headers:{'Content-Type':'application/json'}, body:JSON.stringify({id})});
            loadDocs();
            checkHealth();
        }
        
        async function clearDocs() {
            if (!confirm('전체 삭제?')) return;
            await fetch('/clear', {method:'DELETE'});
            loadDocs();
            checkHealth();
        }
        
        document.getElementById('userInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') { e.preventDefault(); sendMessage(); }
        });
    </script>
</body>
</html>
'''


# ===== API 엔드포인트 =====
@app.route('/')
def home():
    return render_template_string(MOBILE_APP_HTML)


@app.route('/chat', methods=['POST'])
def chat():
    """채팅 API (웹 검색 지원)"""
    data = request.json
    question = data.get('question', '')
    n_results = data.get('n_results', 3)
    use_web_search = data.get('use_web_search', True)  # 기본값: 웹검색 활성화
    
    if not question:
        return jsonify({"error": "질문을 입력해주세요"}), 400
    
    # RAG 검색
    sources = []
    if n_results > 0:
        sources = rag_search(question, n=n_results)
    
    # 웹 검색 자동 판단 (선택사항)
    # 사용자가 명시적으로 끈 경우가 아니면, 질문 내용에 따라 자동 결정
    should_use_web = use_web_search
    
    # Claude API 호출 (웹 검색 포함/미포함)
    answer, web_sources = ask_claude_with_web_search(question, sources, use_web_search=should_use_web)
    
    return jsonify({
        "question": question,
        "answer": answer,
        "sources": sources,
        "web_sources": web_sources,
        "web_search_used": should_use_web
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
        "llm_available": bool(ANTHROPIC_API_KEY),
        "model": CLAUDE_MODEL,
        "llm_type": "claude",
        "web_search_available": True  # 웹 검색 가능 표시
    })


# ===== 시작 =====
load_data()

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🔍 Voice RAG + Claude + 웹검색 (All-in-One)")
    print("="*60)
    print(f"🌐 웹 UI: http://localhost:5001")
    print(f"📚 저장된 문서 수: {len(documents)}")
    print(f"🧠 모델: {CLAUDE_MODEL}")
    print(f"🔍 웹 검색: 활성화됨 (Anthropic web_search tool)")
    
    if ANTHROPIC_API_KEY:
        print("✅ Anthropic API 키 설정됨")
    else:
        print("⚠️  Anthropic API 키가 없습니다!")
        print("   .env 파일에 ANTHROPIC_API_KEY를 설정하세요")
    
    print("="*60)
    print("\n📌 주요 기능:")
    print("   • RAG: 저장된 문서에서 관련 정보 검색")
    print("   • 웹검색: 인터넷에서 최신 정보 검색 (2024년 이후 정보 가능!)")
    print("   • Claude AI: 자연어로 친절하게 답변")
    print("\n🚀 서버 시작! 브라우저에서 http://localhost:5001 열기\n")
    
    app.run(host='0.0.0.0', port=5001, debug=False)
