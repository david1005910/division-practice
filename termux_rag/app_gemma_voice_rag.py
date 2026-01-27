#!/usr/bin/env python3
"""
Voice RAG + Gemma (llama.cpp) + 웹검색 통합 시스템 (All-in-One)
하나의 파일로 RAG + Gemma AI + 웹검색 + 음성 채팅 모두 실행

주요 기능:
    1. RAG: 저장된 문서에서 관련 정보 검색
    2. 웹 검색: DuckDuckGo를 통한 무료 웹 검색 (API 키 불필요!)
    3. Gemma AI: llama.cpp 서버를 통한 로컬 AI (API 키 불필요!)
    4. Voice: 음성 인식(STT) + 음성 출력(TTS)

사전 준비:
    1. llama.cpp 설치 및 빌드
    2. Gemma 모델 다운로드 (gemma-2b.gguf)
    3. llama.cpp 서버 실행:
       cd ~/llama.cpp
       ./build/bin/llama-server -m models/gemma-2b.gguf --host 0.0.0.0 --port 8080

사용법:
    pip install flask hnswlib duckduckgo-search requests
    python app_gemma_voice_rag.py

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

# DuckDuckGo 웹 검색 라이브러리 import (무료!)
try:
    from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False
    print("⚠️ duckduckgo-search 라이브러리가 없습니다. 'pip install duckduckgo-search' 로 설치하세요.")

app = Flask(__name__)

# ===== 설정 =====
# .env 파일에서 설정 자동 로드
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

# ===== Gemma (llama.cpp) 설정 =====
# llama.cpp 서버 주소 (Termux에서 실행 중인 서버)
LLAMA_SERVER_URL = os.environ.get("LLAMA_SERVER_URL", "http://localhost:8080")
LLAMA_MODEL_NAME = os.environ.get("LLAMA_MODEL_NAME", "Gemma-2B")

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


# ===== llama.cpp 서버 연결 확인 =====
def check_llama_server():
    """llama.cpp 서버가 실행 중인지 확인"""
    try:
        response = requests.get(f"{LLAMA_SERVER_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


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


# ===== 웹 검색 (DuckDuckGo - 무료!) =====
def web_search(query, max_results=5):
    """
    DuckDuckGo를 사용한 무료 웹 검색
    """
    if not DDGS_AVAILABLE:
        return []
    
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, region='kr-kr', max_results=max_results))
            
        web_sources = []
        for r in results:
            web_sources.append({
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", "")[:300]
            })
        return web_sources
    except Exception as e:
        print(f"웹 검색 오류: {e}")
        return []


# ===== Gemma AI (llama.cpp 서버 사용) =====
def ask_gemma_with_web_search(question, context_docs, use_web_search=False):
    """
    Gemma AI (llama.cpp 서버)를 사용한 질문 답변
    
    핵심 변경점 (OpenAI → Gemma):
    - OpenAI API → llama.cpp 로컬 서버
    - API 키 불필요!
    - 완전 무료!
    """
    
    # llama.cpp 서버 확인
    if not check_llama_server():
        return """⚠️ llama.cpp 서버가 실행되지 않았습니다.

Termux에서 다음 명령어를 실행하세요:

cd ~/llama.cpp
./build/bin/llama-server -m models/gemma-2b.gguf --host 0.0.0.0 --port 8080

서버가 실행되면 다시 시도해주세요!""", []
    
    # 웹 검색 수행
    web_sources = []
    web_context = ""
    if use_web_search:
        web_sources = web_search(question)
        if web_sources:
            web_context = "\n\nWeb Search Results:\n"
            for i, source in enumerate(web_sources, 1):
                web_context += f"[{i}] {source['title']}\n{source['snippet']}\n\n"
    
    # 프롬프트 구성
    if context_docs:
        context = "\n\n".join([
            f"[Document {i+1}] (Relevance: {doc['similarity']*100:.1f}%)\n{doc['text']}"
            for i, doc in enumerate(context_docs)
        ])
        
        prompt = f"""You are a helpful AI assistant. Answer the question based on the provided documents and web search results.
Please answer in Korean (한국어로 답변해주세요).

=== Related Documents ===
{context}
{web_context}
=== User Question ===
{question}

Answer:"""
    else:
        prompt = f"""You are a helpful AI assistant. Please answer in Korean (한국어로 답변해주세요).
{web_context}
Question: {question}

Answer:"""

    try:
        # llama.cpp 서버 API 호출 (OpenAI 호환 형식)
        response = requests.post(
            f"{LLAMA_SERVER_URL}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": "gemma",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 1024,
                "temperature": 0.7,
                "stream": False
            },
            timeout=120
        )
        
        if response.status_code == 200:
            data = response.json()
            answer_text = data["choices"][0]["message"]["content"]
            return answer_text.strip(), web_sources
        else:
            # 다른 API 형식 시도 (llama.cpp 기본 형식)
            response = requests.post(
                f"{LLAMA_SERVER_URL}/completion",
                headers={"Content-Type": "application/json"},
                json={
                    "prompt": prompt,
                    "n_predict": 1024,
                    "temperature": 0.7,
                    "stop": ["User:", "Question:", "\n\n\n"]
                },
                timeout=120
            )
            
            if response.status_code == 200:
                data = response.json()
                answer_text = data.get("content", "")
                return answer_text.strip(), web_sources
            else:
                return f"⚠️ 서버 오류: {response.status_code}", []
            
    except requests.exceptions.Timeout:
        return "⚠️ 응답 시간 초과. 모델이 로딩 중이거나 질문이 너무 복잡합니다.", []
    except requests.exceptions.ConnectionError:
        return "⚠️ llama.cpp 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.", []
    except Exception as e:
        return f"⚠️ 오류 발생: {str(e)}", []


# ===== HTML 템플릿 (Gemma용 UI) =====
MOBILE_APP_HTML = '''
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="mobile-web-app-capable" content="yes">
    <meta name="theme-color" content="#4285f4">
    <title>🎤 Voice RAG + Gemma AI + 웹검색</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; -webkit-tap-highlight-color: transparent; }
        body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; background: #0f0f23; min-height: 100vh; color: white; }
        .app { display: flex; flex-direction: column; height: 100vh; }
        
        /* 헤더 - Google 블루 그라데이션 (Gemma는 Google 모델) */
        .header { background: linear-gradient(135deg, #4285f4, #34a853); padding: 15px; text-align: center; }
        .header h1 { font-size: 1.2rem; margin-bottom: 5px; }
        .header .status { font-size: 0.75rem; opacity: 0.9; }
        .status-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #ff4757; margin-right: 5px; }
        .status-dot.ok { background: #2ed573; }
        
        /* 탭 */
        .tabs { display: flex; background: #1a1a2e; border-bottom: 1px solid #333; }
        .tab { flex: 1; padding: 12px; text-align: center; background: transparent; border: none; color: #888; font-size: 0.85rem; cursor: pointer; }
        .tab.active { color: #4285f4; border-bottom: 2px solid #4285f4; }
        
        /* 채팅 컨테이너 */
        .chat-container { flex: 1; overflow-y: auto; padding: 15px; display: flex; flex-direction: column; gap: 12px; }
        .message { max-width: 85%; padding: 12px 16px; border-radius: 18px; line-height: 1.5; font-size: 0.95rem; animation: fadeIn 0.3s; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        .message.user { background: linear-gradient(135deg, #4285f4, #34a853); align-self: flex-end; border-bottom-right-radius: 5px; }
        .message.bot { background: #2a2a4a; align-self: flex-start; border-bottom-left-radius: 5px; border: 1px solid #333; }
        .message .sources { font-size: 0.75rem; color: #888; margin-top: 8px; padding-top: 8px; border-top: 1px solid #444; }
        .message .web-sources { font-size: 0.75rem; color: #8be9fd; margin-top: 5px; }
        .message .web-sources a { color: #8be9fd; text-decoration: none; }
        .message .web-sources a:hover { text-decoration: underline; }
        
        /* 메시지 액션 버튼 */
        .message-actions { display: flex; gap: 8px; margin-top: 8px; }
        .message-actions button { background: rgba(66,133,244,0.2); border: 1px solid #4285f4; color: #4285f4; padding: 4px 10px; border-radius: 12px; font-size: 0.7rem; cursor: pointer; }
        .message-actions button:hover { background: rgba(66,133,244,0.4); }
        
        /* 타이핑 애니메이션 */
        .typing { display: flex; gap: 4px; padding: 15px; }
        .typing span { width: 8px; height: 8px; background: #4285f4; border-radius: 50%; animation: bounce 1.4s infinite; }
        .typing span:nth-child(1) { animation-delay: 0s; }
        .typing span:nth-child(2) { animation-delay: 0.2s; }
        .typing span:nth-child(3) { animation-delay: 0.4s; }
        @keyframes bounce { 0%, 80%, 100% { transform: scale(0); } 40% { transform: scale(1); } }
        
        /* 입력 영역 */
        .input-container { background: #0f0f23; padding: 15px; border-top: 1px solid #333; }
        
        /* 검색 옵션 */
        .search-options { display: flex; gap: 10px; margin-bottom: 10px; align-items: center; flex-wrap: wrap; }
        .search-options label { font-size: 0.8rem; color: #888; }
        .search-options select { padding: 5px 10px; background: #1a1a2e; border: 1px solid #333; border-radius: 8px; color: white; font-size: 0.8rem; }
        
        /* 토글 스위치 */
        .toggle-switch { position: relative; width: 44px; height: 24px; }
        .toggle-switch input { opacity: 0; width: 0; height: 0; }
        .toggle-slider { position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0; background-color: #333; transition: 0.4s; border-radius: 24px; }
        .toggle-slider:before { position: absolute; content: ""; height: 18px; width: 18px; left: 3px; bottom: 3px; background-color: white; transition: 0.4s; border-radius: 50%; }
        input:checked + .toggle-slider { background: linear-gradient(135deg, #4285f4, #34a853); }
        input:checked + .toggle-slider:before { transform: translateX(20px); }
        
        /* 입력 행 */
        .input-row { display: flex; gap: 10px; align-items: center; }
        
        /* 음성 버튼 */
        .voice-btn { width: 50px; height: 50px; border-radius: 50%; border: none; background: linear-gradient(135deg, #4285f4, #34a853); color: white; font-size: 1.3rem; cursor: pointer; flex-shrink: 0; transition: transform 0.1s; }
        .voice-btn:active { transform: scale(0.95); }
        .voice-btn.recording { background: linear-gradient(135deg, #ff4757, #ff6b81); animation: pulse 1s infinite; }
        @keyframes pulse { 0%, 100% { box-shadow: 0 0 0 0 rgba(255,71,87,0.4); } 50% { box-shadow: 0 0 0 15px rgba(255,71,87,0); } }
        
        /* 텍스트 입력 */
        .text-input { flex: 1; padding: 12px 15px; background: #1a1a2e; border: 2px solid #333; border-radius: 25px; color: white; font-size: 1rem; }
        .text-input:focus { outline: none; border-color: #4285f4; }
        
        /* 전송 버튼 */
        .send-btn { padding: 12px 20px; background: linear-gradient(135deg, #4285f4, #34a853); border: none; border-radius: 25px; color: white; font-weight: bold; font-size: 0.9rem; cursor: pointer; }
        .send-btn:disabled { opacity: 0.5; }
        
        /* 탭 컨텐츠 */
        .tab-content { flex: 1; overflow-y: auto; padding: 15px; display: none; }
        .tab-content.active { display: block; }
        
        /* 문서 입력 */
        .doc-input { width: 100%; padding: 12px; background: #1a1a2e; border: 1px solid #333; border-radius: 10px; color: white; margin-bottom: 10px; font-size: 0.95rem; }
        textarea.doc-input { min-height: 100px; resize: vertical; }
        
        /* 문서 버튼 */
        .doc-buttons { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 15px; }
        .doc-buttons button { padding: 10px 20px; border: none; border-radius: 10px; font-weight: bold; cursor: pointer; font-size: 0.85rem; }
        .btn-add { background: linear-gradient(135deg, #4285f4, #34a853); color: white; }
        .btn-refresh { background: #333; color: white; }
        .btn-clear { background: #ff4757; color: white; }
        
        /* 문서 아이템 */
        .doc-item { background: #1a1a2e; padding: 12px; border-radius: 10px; margin-bottom: 10px; border-left: 3px solid #4285f4; }
        .doc-item-id { color: #4285f4; font-size: 0.8rem; font-weight: bold; }
        .doc-item-text { color: #ccc; font-size: 0.9rem; margin-top: 5px; }
        .doc-item button { margin-top: 8px; padding: 5px 15px; background: #ff4757; border: none; border-radius: 5px; color: white; font-size: 0.75rem; cursor: pointer; }
        
        /* 빈 상태 */
        .empty-state { text-align: center; color: #666; padding: 40px 20px; }
        .empty-state .icon { font-size: 3rem; margin-bottom: 15px; }
        
        /* 설정 */
        .setting-item { background: #1a1a2e; padding: 15px; border-radius: 10px; margin-bottom: 10px; }
        .setting-item label { display: block; color: #888; font-size: 0.8rem; margin-bottom: 8px; }
        .setting-item select, .setting-item input { width: 100%; padding: 10px; background: #0f0f23; border: 1px solid #333; border-radius: 8px; color: white; font-size: 0.9rem; }
        
        /* 모델 정보 */
        .model-info { background: #1a1a2e; padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 3px solid #4285f4; }
        .model-info h3 { color: #4285f4; font-size: 0.9rem; margin-bottom: 8px; }
        .model-info p { color: #888; font-size: 0.8rem; line-height: 1.5; }
        
        /* 서버 상태 박스 */
        .server-status { background: #1a1a2e; padding: 15px; border-radius: 10px; margin-bottom: 10px; }
        .server-status.connected { border-left: 3px solid #2ed573; }
        .server-status.disconnected { border-left: 3px solid #ff4757; }
        
        /* 기능 배지 */
        .feature-badges { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px; }
        .badge { padding: 4px 10px; border-radius: 12px; font-size: 0.7rem; font-weight: bold; }
        .badge-rag { background: rgba(66,133,244,0.2); color: #4285f4; border: 1px solid #4285f4; }
        .badge-web { background: rgba(139,233,253,0.2); color: #8be9fd; border: 1px solid #8be9fd; }
        .badge-voice { background: rgba(80,250,123,0.2); color: #50fa7b; border: 1px solid #50fa7b; }
        .badge-local { background: rgba(251,188,5,0.2); color: #fbbc05; border: 1px solid #fbbc05; }
    </style>
</head>
<body>
    <div class="app">
        <div class="header">
            <h1>🎤 Voice RAG + Gemma AI + 🔍웹검색</h1>
            <div class="status">
                <span class="status-dot" id="statusDot"></span>
                <span id="statusText">연결 확인 중...</span>
                <span> | 📚 <span id="docCount">0</span>개 문서</span>
            </div>
            <div class="feature-badges">
                <span class="badge badge-rag">📚 RAG</span>
                <span class="badge badge-web">🌐 웹검색</span>
                <span class="badge badge-voice">🎤 음성</span>
                <span class="badge badge-local">💻 로컬AI</span>
            </div>
        </div>
        
        <div class="tabs">
            <button class="tab active" onclick="showTab('chat')">💬 채팅</button>
            <button class="tab" onclick="showTab('docs')">📄 문서</button>
            <button class="tab" onclick="showTab('settings')">⚙️ 설정</button>
        </div>
        
        <div class="chat-container" id="chatContainer">
            <div class="message bot">
                안녕하세요! 저는 Gemma 기반 AI 어시스턴트예요. 🤖<br><br>
                🎤 <b>음성 버튼</b>을 눌러 말하거나 텍스트로 입력하세요<br>
                📚 <b>RAG</b>: 저장된 문서에서 검색<br>
                🌐 <b>웹검색</b>: DuckDuckGo로 최신 정보 검색<br>
                💻 <b>로컬AI</b>: API 키 없이 완전 무료!<br><br>
                ⚠️ llama.cpp 서버가 실행 중이어야 합니다!
            </div>
        </div>
        
        <div class="tab-content" id="docsTab">
            <input type="text" class="doc-input" id="docId" placeholder="문서 ID (선택)">
            <textarea class="doc-input" id="docText" placeholder="문서 내용 입력..."></textarea>
            <div class="doc-buttons">
                <button class="btn-add" onclick="addDoc()">➕ 추가</button>
                <button class="btn-refresh" onclick="loadDocs()">🔄 새로고침</button>
                <button class="btn-clear" onclick="clearDocs()">🗑️ 전체삭제</button>
            </div>
            <div id="docList"></div>
        </div>
        
        <div class="tab-content" id="settingsTab">
            <div class="server-status" id="serverStatus">
                <h3>🖥️ llama.cpp 서버 상태</h3>
                <p id="serverStatusText">확인 중...</p>
            </div>
            
            <div class="model-info">
                <h3>🧠 현재 모델</h3>
                <p id="modelName">Gemma-2B (llama.cpp)</p>
                <p>Google의 Gemma 모델을 로컬에서 실행합니다.</p>
                <p style="color: #fbbc05; margin-top: 8px;">💡 API 키 불필요! 완전 무료!</p>
            </div>
            
            <div class="setting-item">
                <label>🖥️ llama.cpp 서버 주소</label>
                <input type="text" id="serverUrl" value="http://localhost:8080" placeholder="http://localhost:8080">
            </div>
            
            <div class="setting-item">
                <label>📄 RAG 검색 결과 수</label>
                <select id="numResultsSetting">
                    <option value="0">사용안함</option>
                    <option value="2">2개</option>
                    <option value="3" selected>3개</option>
                    <option value="5">5개</option>
                </select>
            </div>
            
            <div class="setting-item">
                <label>🌐 웹 검색 (DuckDuckGo)</label>
                <select id="webSearchSetting">
                    <option value="true" selected>켜기</option>
                    <option value="false">끄기</option>
                </select>
            </div>
            
            <div class="setting-item">
                <label>🔊 음성 자동 읽기 (TTS)</label>
                <select id="autoSpeak">
                    <option value="true" selected>켜기</option>
                    <option value="false">끄기</option>
                </select>
            </div>
            
            <div class="setting-item">
                <label>⏩ 음성 속도</label>
                <select id="speechRate">
                    <option value="0.8">느리게</option>
                    <option value="1.0" selected>보통</option>
                    <option value="1.2">빠르게</option>
                </select>
            </div>
        </div>
        
        <div class="input-container" id="inputContainer">
            <div class="search-options">
                <label>📄 RAG:</label>
                <select id="numResults">
                    <option value="0">OFF</option>
                    <option value="3" selected>3개</option>
                    <option value="5">5개</option>
                </select>
                
                <label style="margin-left: 10px;">🌐 웹검색:</label>
                <label class="toggle-switch">
                    <input type="checkbox" id="webSearchToggle" checked>
                    <span class="toggle-slider"></span>
                </label>
            </div>
            <div class="input-row">
                <button class="voice-btn" id="voiceBtn" onclick="toggleVoice()">🎤</button>
                <input type="text" class="text-input" id="userInput" placeholder="질문을 입력하세요...">
                <button class="send-btn" id="sendBtn" onclick="sendMessage()">전송</button>
            </div>
        </div>
    </div>

    <script>
        let recognition = null;
        let isRecording = false;
        let isProcessing = false;
        
        checkHealth();
        initSpeech();
        
        // ===== 탭 전환 =====
        function showTab(name) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.getElementById('chatContainer').style.display = 'none';
            document.getElementById('docsTab').classList.remove('active');
            document.getElementById('settingsTab').classList.remove('active');
            document.getElementById('inputContainer').style.display = 'none';
            
            event.target.classList.add('active');
            
            if (name === 'chat') {
                document.getElementById('chatContainer').style.display = 'flex';
                document.getElementById('inputContainer').style.display = 'block';
            } else if (name === 'docs') {
                document.getElementById('docsTab').classList.add('active');
                loadDocs();
            } else {
                document.getElementById('settingsTab').classList.add('active');
                checkHealth();
            }
        }
        
        // ===== 서버 상태 확인 =====
        async function checkHealth() {
            try {
                const res = await fetch('/health');
                const data = await res.json();
                
                const isConnected = data.llm_available;
                document.getElementById('statusDot').classList.toggle('ok', isConnected);
                document.getElementById('statusText').textContent = isConnected ? 'Gemma 연결됨' : '서버 연결 필요';
                document.getElementById('docCount').textContent = data.documents || 0;
                document.getElementById('modelName').textContent = data.model || 'Gemma-2B';
                
                // 서버 상태 박스 업데이트
                const serverStatus = document.getElementById('serverStatus');
                const serverStatusText = document.getElementById('serverStatusText');
                if (isConnected) {
                    serverStatus.classList.remove('disconnected');
                    serverStatus.classList.add('connected');
                    serverStatusText.innerHTML = '✅ llama.cpp 서버 연결됨<br>모델: ' + data.model;
                } else {
                    serverStatus.classList.remove('connected');
                    serverStatus.classList.add('disconnected');
                    serverStatusText.innerHTML = '❌ 서버 연결 안됨<br><br>Termux에서 실행하세요:<br><code>cd ~/llama.cpp<br>./build/bin/llama-server -m models/gemma-2b.gguf --host 0.0.0.0 --port 8080</code>';
                }
            } catch(e) {
                document.getElementById('statusDot').classList.remove('ok');
                document.getElementById('statusText').textContent = '서버 연결 안됨';
            }
        }
        
        // ===== 음성 인식 초기화 (STT) =====
        function initSpeech() {
            if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
                const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
                recognition = new SR();
                recognition.continuous = false;
                recognition.interimResults = true;
                recognition.lang = 'ko-KR';
                
                recognition.onresult = (e) => {
                    document.getElementById('userInput').value = e.results[0][0].transcript;
                };
                
                recognition.onend = () => {
                    isRecording = false;
                    document.getElementById('voiceBtn').classList.remove('recording');
                    if (document.getElementById('userInput').value.trim() && !isProcessing) {
                        sendMessage();
                    }
                };
                
                recognition.onerror = (e) => {
                    isRecording = false;
                    document.getElementById('voiceBtn').classList.remove('recording');
                    console.log('음성 인식 오류:', e.error);
                };
            }
        }
        
        // ===== 음성 인식 토글 =====
        function toggleVoice() {
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
            }
        }
        
        // ===== 음성 출력 (TTS) =====
        function speak(text) {
            if (!('speechSynthesis' in window)) return;
            if (document.getElementById('autoSpeak').value !== 'true') return;
            
            speechSynthesis.cancel();
            
            const cleanText = text
                .replace(/\\*\\*(.+?)\\*\\*/g, '$1')
                .replace(/\\*(.+?)\\*/g, '$1')
                .replace(/`(.+?)`/g, '$1')
                .replace(/#{1,6}\\s/g, '')
                .replace(/\\n/g, ' ');
            
            const utterance = new SpeechSynthesisUtterance(cleanText);
            utterance.lang = 'ko-KR';
            utterance.rate = parseFloat(document.getElementById('speechRate').value);
            speechSynthesis.speak(utterance);
        }
        
        function stopSpeak() { 
            speechSynthesis.cancel(); 
        }
        
        // ===== 메시지 추가 =====
        function addMsg(text, isUser, sources = [], webSources = []) {
            const c = document.getElementById('chatContainer');
            const d = document.createElement('div');
            d.className = 'message ' + (isUser ? 'user' : 'bot');
            
            let h = text
                .replace(/\\n/g, '<br>')
                .replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>')
                .replace(/\\*(.+?)\\*/g, '<em>$1</em>')
                .replace(/`(.+?)`/g, '<code style="background:#1a1a2e;padding:2px 5px;border-radius:3px;">$1</code>');
            
            if (!isUser) {
                if (sources && sources.length > 0) {
                    h += '<div class="sources">📚 참고 문서: ' + sources.map(s => s.id + ' (' + (s.similarity*100).toFixed(0) + '%)').join(', ') + '</div>';
                }
                
                if (webSources && webSources.length > 0) {
                    h += '<div class="web-sources">🌐 웹 검색: ';
                    h += webSources.map(s => '<a href="' + s.url + '" target="_blank">' + (s.title || '링크') + '</a>').join(', ');
                    h += '</div>';
                }
                
                const safeText = text.replace(/'/g, "\\\\'").replace(/"/g, '\\\\"');
                h += '<div class="message-actions">';
                h += '<button onclick="speak(\\'' + safeText + '\\')">🔊 듣기</button>';
                h += '<button onclick="stopSpeak()">⏹️ 정지</button>';
                h += '</div>';
            }
            
            d.innerHTML = h;
            c.appendChild(d);
            c.scrollTop = c.scrollHeight;
        }
        
        // ===== 타이핑 애니메이션 =====
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
        
        // ===== 메시지 전송 =====
        async function sendMessage() {
            const input = document.getElementById('userInput');
            const q = input.value.trim();
            if (!q || isProcessing) return;
            
            isProcessing = true;
            document.getElementById('sendBtn').disabled = true;
            document.getElementById('voiceBtn').disabled = true;
            
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
                addMsg('⚠️ 오류: ' + e.message, false);
            } finally {
                isProcessing = false;
                document.getElementById('sendBtn').disabled = false;
                document.getElementById('voiceBtn').disabled = false;
            }
        }
        
        // ===== 문서 관리 =====
        async function loadDocs() {
            try {
                const res = await fetch('/list?limit=50');
                const data = await res.json();
                document.getElementById('docCount').textContent = data.total;
                const list = document.getElementById('docList');
                if (data.documents && data.documents.length) {
                    list.innerHTML = data.documents.map(d => 
                        '<div class="doc-item">' +
                        '<div class="doc-item-id">🏷️ ' + d.id + '</div>' +
                        '<div class="doc-item-text">' + d.text + '</div>' +
                        '<button onclick="delDoc(\\'' + d.id + '\\')">🗑️ 삭제</button>' +
                        '</div>'
                    ).join('');
                } else {
                    list.innerHTML = '<div class="empty-state"><div class="icon">📄</div><p>저장된 문서가 없습니다.<br>문서를 추가하면 RAG 검색에 사용됩니다.</p></div>';
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
                alert('✅ 문서 추가됨: ' + data.id);
                document.getElementById('docText').value = '';
                document.getElementById('docId').value = '';
                loadDocs();
                checkHealth();
            } catch(e) { alert('❌ 추가 실패'); }
        }
        
        async function delDoc(id) {
            if (!confirm('이 문서를 삭제하시겠습니까?')) return;
            await fetch('/delete', {method:'DELETE', headers:{'Content-Type':'application/json'}, body:JSON.stringify({id})});
            loadDocs();
            checkHealth();
        }
        
        async function clearDocs() {
            if (!confirm('⚠️ 모든 문서를 삭제하시겠습니까?\\n이 작업은 되돌릴 수 없습니다.')) return;
            await fetch('/clear', {method:'DELETE'});
            loadDocs();
            checkHealth();
        }
        
        // ===== 엔터 키 전송 =====
        document.getElementById('userInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') { 
                e.preventDefault(); 
                sendMessage(); 
            }
        });
        
        // ===== 설정 동기화 =====
        document.getElementById('numResultsSetting').addEventListener('change', (e) => {
            document.getElementById('numResults').value = e.target.value;
        });
        
        document.getElementById('webSearchSetting').addEventListener('change', (e) => {
            document.getElementById('webSearchToggle').checked = e.target.value === 'true';
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
    use_web_search = data.get('use_web_search', True)
    
    if not question:
        return jsonify({"error": "질문을 입력해주세요"}), 400
    
    # RAG 검색
    sources = []
    if n_results > 0:
        sources = rag_search(question, n=n_results)
    
    # Gemma AI 호출 (llama.cpp 서버)
    answer, web_sources = ask_gemma_with_web_search(question, sources, use_web_search=use_web_search)
    
    return jsonify({
        "question": question,
        "answer": answer,
        "sources": sources,
        "web_sources": web_sources,
        "web_search_used": use_web_search
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
    llm_available = check_llama_server()
    return jsonify({
        "status": "running",
        "documents": len(documents),
        "llm_available": llm_available,
        "model": LLAMA_MODEL_NAME,
        "llm_type": "gemma (llama.cpp)",
        "server_url": LLAMA_SERVER_URL,
        "web_search_available": DDGS_AVAILABLE,
        "voice_available": True
    })


# ===== 시작 =====
load_data()

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🎤 Voice RAG + Gemma AI + 웹검색 (All-in-One)")
    print("="*60)
    print(f"🌐 웹 UI: http://localhost:5001")
    print(f"📚 저장된 문서 수: {len(documents)}")
    print(f"🧠 모델: {LLAMA_MODEL_NAME}")
    print(f"🖥️ llama.cpp 서버: {LLAMA_SERVER_URL}")
    
    print("\n📌 주요 기능:")
    print("   🎤 음성 인식 (STT): 마이크 버튼으로 음성 입력")
    print("   🔊 음성 출력 (TTS): AI 응답을 음성으로 읽어줌")
    print("   📚 RAG: 저장된 문서에서 관련 정보 검색")
    print("   🌐 웹검색: DuckDuckGo 무료 검색")
    print("   💻 로컬AI: API 키 불필요! 완전 무료!")
    
    # llama.cpp 서버 확인
    if check_llama_server():
        print("\n✅ llama.cpp 서버 연결됨")
    else:
        print("\n⚠️  llama.cpp 서버가 실행되지 않았습니다!")
        print("\n   다른 Termux 세션에서 다음 명령어를 실행하세요:")
        print("   ─────────────────────────────────────────")
        print("   cd ~/llama.cpp")
        print("   ./build/bin/llama-server \\")
        print("     -m models/gemma-2b.gguf \\")
        print("     --host 0.0.0.0 \\")
        print("     --port 8080")
        print("   ─────────────────────────────────────────")
    
    if DDGS_AVAILABLE:
        print("✅ DuckDuckGo 웹 검색 사용 가능")
    else:
        print("⚠️  DuckDuckGo 검색 불가 - 'pip install duckduckgo-search' 실행하세요")
    
    print("="*60)
    print("\n🚀 서버 시작! 브라우저에서 http://localhost:5001 열기\n")
    
    app.run(host='0.0.0.0', port=5001, debug=False)
