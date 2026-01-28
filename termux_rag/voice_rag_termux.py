#!/usr/bin/env python3
"""
🎤 Voice RAG System for Termux (API 키 불필요!)

특징:
    - API 키 없이 작동
    - Ollama (로컬 LLM) 또는 완전 오프라인 모드 지원
    - DuckDuckGo 무료 웹 검색
    - 음성 인식 (STT) + 음성 출력 (TTS)
    - RAG (문서 검색)

Termux 설치:
    pkg update && pkg upgrade
    pkg install python
    pip install flask requests beautifulsoup4

Ollama 사용시 (선택):
    # PC나 서버에 Ollama 설치 후
    # OLLAMA_HOST 환경변수 설정
    
실행:
    python voice_rag_termux.py
    
브라우저:
    http://localhost:5001
"""

from flask import Flask, request, jsonify, render_template_string
import json
import os
import re
import math
from collections import Counter
from urllib.parse import quote_plus
import requests
from datetime import datetime

app = Flask(__name__)

# ===== 설정 =====
# Ollama 설정 (선택사항 - 없으면 오프라인 모드)
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:1b")  # 작은 모델 (Termux용)

# 데이터 저장 경로
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_FILE = os.path.join(DATA_DIR, "rag_docs.json")

# ===== 전역 변수 =====
documents = {}
vocab = {}
idf_values = {}
ollama_available = False


# ===== 텍스트 처리 =====
def tokenize(text):
    """한국어/영어 토크나이저"""
    text = text.lower()
    text = re.sub(r'[^\w\s가-힣]', ' ', text)
    tokens = text.split()
    stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                 '이', '가', '은', '는', '을', '를', '의', '에', '에서', '으로',
                 '로', '와', '과', '도', '만', '까지', '부터', '그', '저', '것'}
    return [t for t in tokens if t not in stopwords and len(t) > 1]


def text_to_vector(text):
    """텍스트를 TF-IDF 벡터로 변환"""
    tokens = tokenize(text)
    if not tokens:
        return {}
    
    tf = Counter(tokens)
    total = len(tokens)
    vector = {}
    
    for word, count in tf.items():
        tf_val = count / total
        idf_val = idf_values.get(word, 1.0)
        vector[word] = tf_val * idf_val
    
    return vector


def cosine_similarity(vec1, vec2):
    """코사인 유사도 계산"""
    if not vec1 or not vec2:
        return 0.0
    
    # 공통 단어만 사용
    common_words = set(vec1.keys()) & set(vec2.keys())
    if not common_words:
        return 0.0
    
    dot_product = sum(vec1[w] * vec2[w] for w in common_words)
    norm1 = math.sqrt(sum(v**2 for v in vec1.values()))
    norm2 = math.sqrt(sum(v**2 for v in vec2.values()))
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


# ===== 데이터 관리 =====
def save_data():
    """데이터 저장"""
    save_obj = {
        "documents": documents,
        "vocab": vocab,
        "idf_values": idf_values
    }
    with open(DOCS_FILE, 'w', encoding='utf-8') as f:
        json.dump(save_obj, f, ensure_ascii=False, indent=2)


def load_data():
    """데이터 로드"""
    global documents, vocab, idf_values
    
    if os.path.exists(DOCS_FILE):
        try:
            with open(DOCS_FILE, 'r', encoding='utf-8') as f:
                save_obj = json.load(f)
            documents = save_obj.get("documents", {})
            vocab = save_obj.get("vocab", {})
            idf_values = save_obj.get("idf_values", {})
            print(f"📚 {len(documents)}개 문서 로드됨")
        except Exception as e:
            print(f"⚠️ 데이터 로드 실패: {e}")


def rebuild_vocab():
    """어휘 사전 재구축"""
    global vocab, idf_values
    
    word_doc_count = Counter()
    all_words = set()
    
    for doc in documents.values():
        tokens = set(tokenize(doc["text"]))
        all_words.update(tokens)
        for word in tokens:
            word_doc_count[word] += 1
    
    vocab = {word: idx for idx, word in enumerate(sorted(all_words))}
    n_docs = len(documents) + 1
    idf_values = {word: math.log(n_docs / (count + 1)) + 1
                  for word, count in word_doc_count.items()}


# ===== RAG 검색 =====
def rag_search(query, n=3):
    """RAG에서 관련 문서 검색"""
    if not documents:
        return []
    
    query_vec = text_to_vector(query)
    results = []
    
    for doc_id, doc in documents.items():
        doc_vec = text_to_vector(doc["text"])
        similarity = cosine_similarity(query_vec, doc_vec)
        
        if similarity > 0.01:  # 최소 유사도
            results.append({
                "id": doc_id,
                "text": doc["text"],
                "similarity": round(similarity, 4),
                "metadata": doc.get("metadata", {})
            })
    
    # 유사도 순 정렬
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:n]


# ===== 무료 웹 검색 (DuckDuckGo) =====
def web_search_duckduckgo(query, max_results=5):
    """DuckDuckGo를 이용한 무료 웹 검색"""
    try:
        # DuckDuckGo HTML 버전 사용 (API 키 불필요)
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Linux; Android 10) AppleWebKit/537.36"
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            return []
        
        # 간단한 HTML 파싱 (BeautifulSoup 없이)
        results = []
        html = response.text
        
        # 결과 링크 추출 (정규식 사용)
        pattern = r'<a[^>]+class="result__a"[^>]*href="([^"]+)"[^>]*>([^<]+)</a>'
        matches = re.findall(pattern, html)
        
        for url, title in matches[:max_results]:
            # DuckDuckGo 리다이렉트 URL 처리
            if "uddg=" in url:
                actual_url = re.search(r'uddg=([^&]+)', url)
                if actual_url:
                    from urllib.parse import unquote
                    url = unquote(actual_url.group(1))
            
            results.append({
                "title": title.strip(),
                "url": url,
                "snippet": ""
            })
        
        # 스니펫 추출 시도
        snippet_pattern = r'<a class="result__snippet"[^>]*>([^<]+)</a>'
        snippets = re.findall(snippet_pattern, html)
        
        for i, snippet in enumerate(snippets[:len(results)]):
            if i < len(results):
                results[i]["snippet"] = snippet.strip()[:200]
        
        return results
        
    except Exception as e:
        print(f"웹 검색 오류: {e}")
        return []


def fetch_web_content(url, max_chars=2000):
    """웹 페이지 내용 가져오기"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Linux; Android 10) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            return ""
        
        html = response.text
        
        # HTML 태그 제거 (간단한 방법)
        text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # HTML 엔티티 디코딩
        text = text.replace('&nbsp;', ' ')
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&quot;', '"')
        
        return text[:max_chars]
        
    except Exception as e:
        print(f"웹 콘텐츠 가져오기 오류: {e}")
        return ""


# ===== Ollama (로컬 LLM) =====
def check_ollama():
    """Ollama 사용 가능 여부 확인"""
    global ollama_available
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
        ollama_available = response.status_code == 200
        return ollama_available
    except:
        ollama_available = False
        return False


def ask_ollama(prompt, system_prompt=""):
    """Ollama API 호출"""
    try:
        request_body = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 500  # 토큰 제한 (Termux 메모리 고려)
            }
        }
        
        if system_prompt:
            request_body["system"] = system_prompt
        
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json=request_body,
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            return None
            
    except Exception as e:
        print(f"Ollama 오류: {e}")
        return None


# ===== 오프라인 AI (키워드 기반) =====
def offline_answer(question, context_docs, web_results):
    """
    완전 오프라인 답변 생성 (LLM 없이)
    키워드 매칭 + 템플릿 기반
    """
    answer_parts = []
    
    # 1. 인사말 처리
    greetings = ['안녕', '하이', 'hi', 'hello', '반가워']
    if any(g in question.lower() for g in greetings):
        return "안녕하세요! 무엇을 도와드릴까요? 📚 RAG 문서나 🌐 웹 검색을 통해 정보를 찾아드릴 수 있어요."
    
    # 2. RAG 문서 기반 답변
    if context_docs:
        answer_parts.append("📚 **저장된 문서에서 찾은 정보:**\n")
        for i, doc in enumerate(context_docs, 1):
            # 질문과 관련된 문장 추출
            sentences = doc["text"].split('.')
            relevant = []
            q_tokens = set(tokenize(question))
            
            for sent in sentences:
                sent_tokens = set(tokenize(sent))
                if q_tokens & sent_tokens:  # 교집합이 있으면
                    relevant.append(sent.strip())
            
            if relevant:
                answer_parts.append(f"[문서 {i}] {'. '.join(relevant[:2])}.")
            else:
                answer_parts.append(f"[문서 {i}] {doc['text'][:150]}...")
            answer_parts.append(f"(유사도: {doc['similarity']*100:.0f}%)\n")
    
    # 3. 웹 검색 결과 기반 답변
    if web_results:
        answer_parts.append("\n🌐 **웹 검색 결과:**\n")
        for i, result in enumerate(web_results[:3], 1):
            title = result.get("title", "")
            snippet = result.get("snippet", "")
            url = result.get("url", "")
            
            if snippet:
                answer_parts.append(f"{i}. **{title}**\n   {snippet}\n")
            elif title:
                answer_parts.append(f"{i}. **{title}**\n")
    
    # 4. 결과가 없는 경우
    if not answer_parts:
        return f"'{question}'에 대한 정보를 찾지 못했어요. 다른 키워드로 검색하거나, 문서 탭에서 관련 정보를 추가해보세요."
    
    return "\n".join(answer_parts)


# ===== 통합 답변 생성 =====
def generate_answer(question, context_docs, web_results, use_web_search=False):
    """
    LLM 또는 오프라인 모드로 답변 생성
    """
    web_sources = []
    
    # 웹 검색 실행
    if use_web_search:
        web_results = web_search_duckduckgo(question, max_results=3)
        web_sources = web_results
        
        # 웹 페이지 내용 가져오기 (첫 번째 결과만)
        if web_results and web_results[0].get("url"):
            content = fetch_web_content(web_results[0]["url"], max_chars=1500)
            if content:
                web_results[0]["content"] = content
    
    # Ollama 사용 가능하면 LLM으로 답변
    if ollama_available:
        # 컨텍스트 구성
        context = ""
        
        if context_docs:
            context += "=== 저장된 문서 ===\n"
            for i, doc in enumerate(context_docs, 1):
                context += f"[문서 {i}] {doc['text'][:500]}\n\n"
        
        if web_results:
            context += "=== 웹 검색 결과 ===\n"
            for i, result in enumerate(web_results[:3], 1):
                context += f"[{i}] {result.get('title', '')}\n"
                if result.get("snippet"):
                    context += f"    {result['snippet']}\n"
                if result.get("content"):
                    context += f"    내용: {result['content'][:500]}...\n"
        
        system_prompt = """당신은 친절한 AI 어시스턴트입니다.
주어진 문서와 웹 검색 결과를 참고하여 질문에 답변하세요.
답변은 간결하고 명확하게 한국어로 해주세요.
정보의 출처를 간단히 언급해주세요."""

        prompt = f"""{context}

질문: {question}

위 정보를 바탕으로 질문에 답변해주세요."""

        answer = ask_ollama(prompt, system_prompt)
        
        if answer:
            return answer, web_sources
    
    # Ollama 없으면 오프라인 모드
    answer = offline_answer(question, context_docs, web_results)
    return answer, web_sources


# ===== HTML 템플릿 =====
MOBILE_HTML = '''
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="mobile-web-app-capable" content="yes">
    <title>🎤 Voice RAG (Termux)</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; -webkit-tap-highlight-color: transparent; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0f0f23; min-height: 100vh; color: white; }
        .app { display: flex; flex-direction: column; height: 100vh; }
        
        .header { background: linear-gradient(135deg, #667eea, #764ba2); padding: 12px; text-align: center; }
        .header h1 { font-size: 1.1rem; margin-bottom: 4px; }
        .header .status { font-size: 0.7rem; opacity: 0.9; }
        .status-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 5px; }
        .status-dot.ok { background: #2ed573; }
        .status-dot.warn { background: #ffa502; }
        .status-dot.off { background: #ff4757; }
        
        .badges { display: flex; gap: 6px; justify-content: center; margin-top: 6px; flex-wrap: wrap; }
        .badge { padding: 2px 8px; border-radius: 10px; font-size: 0.65rem; }
        .badge-rag { background: rgba(102,126,234,0.3); border: 1px solid #667eea; }
        .badge-web { background: rgba(139,233,253,0.3); border: 1px solid #8be9fd; }
        .badge-llm { background: rgba(80,250,123,0.3); border: 1px solid #50fa7b; }
        .badge-offline { background: rgba(255,165,0,0.3); border: 1px solid #ffa502; }
        
        .tabs { display: flex; background: #1a1a2e; }
        .tab { flex: 1; padding: 10px; text-align: center; background: transparent; border: none; color: #888; font-size: 0.8rem; cursor: pointer; border-bottom: 2px solid transparent; }
        .tab.active { color: #667eea; border-bottom-color: #667eea; }
        
        .chat-container { flex: 1; overflow-y: auto; padding: 10px; display: flex; flex-direction: column; gap: 10px; }
        .message { max-width: 88%; padding: 10px 14px; border-radius: 16px; line-height: 1.5; font-size: 0.9rem; animation: fadeIn 0.3s; word-wrap: break-word; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        .message.user { background: linear-gradient(135deg, #667eea, #764ba2); align-self: flex-end; border-bottom-right-radius: 4px; }
        .message.bot { background: #1a1a2e; align-self: flex-start; border-bottom-left-radius: 4px; border: 1px solid #333; }
        .message strong { color: #8be9fd; }
        .message .sources { font-size: 0.7rem; color: #888; margin-top: 8px; padding-top: 6px; border-top: 1px solid #333; }
        .message .web-link { color: #8be9fd; text-decoration: none; font-size: 0.75rem; }
        
        .msg-actions { display: flex; gap: 6px; margin-top: 6px; }
        .msg-actions button { background: rgba(102,126,234,0.2); border: 1px solid #667eea; color: #667eea; padding: 3px 8px; border-radius: 10px; font-size: 0.65rem; cursor: pointer; }
        
        .typing { display: flex; gap: 4px; padding: 12px; }
        .typing span { width: 8px; height: 8px; background: #667eea; border-radius: 50%; animation: bounce 1.4s infinite; }
        .typing span:nth-child(1) { animation-delay: 0s; }
        .typing span:nth-child(2) { animation-delay: 0.2s; }
        .typing span:nth-child(3) { animation-delay: 0.4s; }
        @keyframes bounce { 0%, 80%, 100% { transform: scale(0); } 40% { transform: scale(1); } }
        
        .input-area { background: #0f0f23; padding: 10px; border-top: 1px solid #333; }
        .options { display: flex; gap: 8px; margin-bottom: 8px; align-items: center; flex-wrap: wrap; }
        .options label { font-size: 0.75rem; color: #888; }
        .options select { padding: 4px 8px; background: #1a1a2e; border: 1px solid #333; border-radius: 6px; color: white; font-size: 0.75rem; }
        
        .toggle { position: relative; width: 40px; height: 22px; }
        .toggle input { opacity: 0; width: 0; height: 0; }
        .toggle-slider { position: absolute; cursor: pointer; inset: 0; background: #333; border-radius: 22px; transition: 0.3s; }
        .toggle-slider:before { position: absolute; content: ""; height: 16px; width: 16px; left: 3px; bottom: 3px; background: white; border-radius: 50%; transition: 0.3s; }
        input:checked + .toggle-slider { background: linear-gradient(135deg, #667eea, #764ba2); }
        input:checked + .toggle-slider:before { transform: translateX(18px); }
        
        .input-row { display: flex; gap: 8px; align-items: center; }
        .voice-btn { width: 46px; height: 46px; border-radius: 50%; border: none; background: linear-gradient(135deg, #667eea, #764ba2); color: white; font-size: 1.2rem; cursor: pointer; flex-shrink: 0; }
        .voice-btn.recording { background: linear-gradient(135deg, #ff4757, #ff6b81); animation: pulse 1s infinite; }
        @keyframes pulse { 0%, 100% { box-shadow: 0 0 0 0 rgba(255,71,87,0.4); } 50% { box-shadow: 0 0 0 12px rgba(255,71,87,0); } }
        
        .text-input { flex: 1; padding: 10px 14px; background: #1a1a2e; border: 2px solid #333; border-radius: 22px; color: white; font-size: 0.95rem; }
        .text-input:focus { outline: none; border-color: #667eea; }
        .send-btn { padding: 10px 16px; background: linear-gradient(135deg, #667eea, #764ba2); border: none; border-radius: 22px; color: white; font-weight: bold; font-size: 0.85rem; cursor: pointer; }
        .send-btn:disabled { opacity: 0.5; }
        
        .tab-content { flex: 1; overflow-y: auto; padding: 12px; display: none; }
        .tab-content.active { display: block; }
        
        .doc-input { width: 100%; padding: 10px; background: #1a1a2e; border: 1px solid #333; border-radius: 8px; color: white; margin-bottom: 8px; font-size: 0.9rem; }
        textarea.doc-input { min-height: 80px; resize: vertical; }
        
        .doc-btns { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; }
        .doc-btns button { padding: 8px 16px; border: none; border-radius: 8px; font-weight: bold; cursor: pointer; font-size: 0.8rem; }
        .btn-add { background: linear-gradient(135deg, #667eea, #764ba2); color: white; }
        .btn-refresh { background: #333; color: white; }
        .btn-clear { background: #ff4757; color: white; }
        
        .doc-item { background: #1a1a2e; padding: 10px; border-radius: 8px; margin-bottom: 8px; border-left: 3px solid #667eea; }
        .doc-item-id { color: #667eea; font-size: 0.75rem; font-weight: bold; }
        .doc-item-text { color: #ccc; font-size: 0.85rem; margin-top: 4px; }
        .doc-item button { margin-top: 6px; padding: 4px 12px; background: #ff4757; border: none; border-radius: 4px; color: white; font-size: 0.7rem; cursor: pointer; }
        
        .empty { text-align: center; color: #666; padding: 30px 15px; }
        .empty .icon { font-size: 2.5rem; margin-bottom: 10px; }
        
        .setting { background: #1a1a2e; padding: 12px; border-radius: 8px; margin-bottom: 8px; }
        .setting label { display: block; color: #888; font-size: 0.75rem; margin-bottom: 6px; }
        .setting select { width: 100%; padding: 8px; background: #0f0f23; border: 1px solid #333; border-radius: 6px; color: white; font-size: 0.85rem; }
        
        .info-box { background: #1a1a2e; padding: 12px; border-radius: 8px; margin-bottom: 8px; border-left: 3px solid #667eea; }
        .info-box h3 { color: #667eea; font-size: 0.85rem; margin-bottom: 6px; }
        .info-box p { color: #888; font-size: 0.75rem; line-height: 1.5; }
    </style>
</head>
<body>
    <div class="app">
        <div class="header">
            <h1>🎤 Voice RAG (Termux)</h1>
            <div class="status">
                <span class="status-dot" id="statusDot"></span>
                <span id="statusText">연결 중...</span>
                <span> | 📚 <span id="docCount">0</span>개</span>
            </div>
            <div class="badges">
                <span class="badge badge-rag">📚 RAG</span>
                <span class="badge badge-web">🌐 웹검색</span>
                <span class="badge" id="llmBadge">🤖 확인중</span>
            </div>
        </div>
        
        <div class="tabs">
            <button class="tab active" onclick="showTab('chat')">💬 채팅</button>
            <button class="tab" onclick="showTab('docs')">📄 문서</button>
            <button class="tab" onclick="showTab('settings')">⚙️ 설정</button>
        </div>
        
        <div class="chat-container" id="chatBox">
            <div class="message bot">
                안녕하세요! 🎤 Voice RAG입니다.<br><br>
                🎤 음성 버튼으로 말하거나 텍스트 입력<br>
                📚 RAG: 저장된 문서 검색<br>
                🌐 웹검색: 인터넷 검색 (무료)<br><br>
                무엇이든 물어보세요!
            </div>
        </div>
        
        <div class="tab-content" id="docsTab">
            <input type="text" class="doc-input" id="docId" placeholder="문서 ID (선택)">
            <textarea class="doc-input" id="docText" placeholder="문서 내용..."></textarea>
            <div class="doc-btns">
                <button class="btn-add" onclick="addDoc()">➕ 추가</button>
                <button class="btn-refresh" onclick="loadDocs()">🔄</button>
                <button class="btn-clear" onclick="clearDocs()">🗑️</button>
            </div>
            <div id="docList"></div>
        </div>
        
        <div class="tab-content" id="settingsTab">
            <div class="info-box">
                <h3>🤖 AI 모드</h3>
                <p id="modeInfo">확인 중...</p>
            </div>
            <div class="setting">
                <label>📄 RAG 결과 수</label>
                <select id="numResultsSet">
                    <option value="0">OFF</option>
                    <option value="2">2개</option>
                    <option value="3" selected>3개</option>
                    <option value="5">5개</option>
                </select>
            </div>
            <div class="setting">
                <label>🔊 자동 음성 읽기</label>
                <select id="autoSpeak">
                    <option value="true" selected>켜기</option>
                    <option value="false">끄기</option>
                </select>
            </div>
            <div class="setting">
                <label>⏩ 음성 속도</label>
                <select id="speechRate">
                    <option value="0.8">느리게</option>
                    <option value="1.0" selected>보통</option>
                    <option value="1.2">빠르게</option>
                </select>
            </div>
            <div class="info-box">
                <h3>💡 Ollama 사용법</h3>
                <p>PC에서 Ollama 실행 후:<br>
                OLLAMA_HOST=http://PC_IP:11434<br>
                로 환경변수 설정하면 LLM 사용 가능!</p>
            </div>
        </div>
        
        <div class="input-area" id="inputArea">
            <div class="options">
                <label>📄 RAG:</label>
                <select id="numResults">
                    <option value="0">OFF</option>
                    <option value="3" selected>3개</option>
                </select>
                <label style="margin-left:8px">🌐 웹:</label>
                <label class="toggle">
                    <input type="checkbox" id="webToggle" checked>
                    <span class="toggle-slider"></span>
                </label>
            </div>
            <div class="input-row">
                <button class="voice-btn" id="voiceBtn" onclick="toggleVoice()">🎤</button>
                <input type="text" class="text-input" id="userInput" placeholder="질문하세요...">
                <button class="send-btn" id="sendBtn" onclick="sendMsg()">전송</button>
            </div>
        </div>
    </div>

    <script>
        let recognition = null;
        let isRecording = false;
        let processing = false;
        
        checkHealth();
        initSpeech();
        
        function showTab(name) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            event.target.classList.add('active');
            document.getElementById('chatBox').style.display = name === 'chat' ? 'flex' : 'none';
            document.getElementById('docsTab').classList.toggle('active', name === 'docs');
            document.getElementById('settingsTab').classList.toggle('active', name === 'settings');
            document.getElementById('inputArea').style.display = name === 'chat' ? 'block' : 'none';
            if (name === 'docs') loadDocs();
        }
        
        async function checkHealth() {
            try {
                const res = await fetch('/health');
                const data = await res.json();
                const dot = document.getElementById('statusDot');
                const txt = document.getElementById('statusText');
                const badge = document.getElementById('llmBadge');
                const info = document.getElementById('modeInfo');
                
                document.getElementById('docCount').textContent = data.documents || 0;
                
                if (data.ollama_available) {
                    dot.className = 'status-dot ok';
                    txt.textContent = 'Ollama 연결됨';
                    badge.className = 'badge badge-llm';
                    badge.textContent = '🤖 ' + (data.ollama_model || 'LLM');
                    info.textContent = 'Ollama LLM 사용 중: ' + (data.ollama_model || '');
                } else {
                    dot.className = 'status-dot warn';
                    txt.textContent = '오프라인 모드';
                    badge.className = 'badge badge-offline';
                    badge.textContent = '📴 오프라인';
                    info.textContent = '오프라인 모드 (키워드 매칭)\\nOllama 연결 시 LLM 사용 가능';
                }
            } catch(e) {
                document.getElementById('statusDot').className = 'status-dot off';
                document.getElementById('statusText').textContent = '서버 오류';
            }
        }
        
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
                    if (document.getElementById('userInput').value.trim() && !processing) sendMsg();
                };
                recognition.onerror = () => {
                    isRecording = false;
                    document.getElementById('voiceBtn').classList.remove('recording');
                };
            }
        }
        
        function toggleVoice() {
            if (!recognition) { alert('음성 인식 미지원. Chrome 사용하세요.'); return; }
            if (isRecording) { recognition.stop(); }
            else { recognition.start(); isRecording = true; document.getElementById('voiceBtn').classList.add('recording'); }
        }
        
        function speak(text) {
            if (!('speechSynthesis' in window)) return;
            if (document.getElementById('autoSpeak').value !== 'true') return;
            speechSynthesis.cancel();
            const clean = text.replace(/\\*\\*(.+?)\\*\\*/g, '$1').replace(/\\*(.+?)\\*/g, '$1').replace(/`(.+?)`/g, '$1').replace(/#{1,6}\\s/g, '').replace(/\\n/g, ' ');
            const u = new SpeechSynthesisUtterance(clean);
            u.lang = 'ko-KR';
            u.rate = parseFloat(document.getElementById('speechRate').value);
            speechSynthesis.speak(u);
        }
        
        function stopSpeak() { speechSynthesis.cancel(); }
        
        function addMsg(text, isUser, sources = [], webSources = []) {
            const c = document.getElementById('chatBox');
            const d = document.createElement('div');
            d.className = 'message ' + (isUser ? 'user' : 'bot');
            
            let h = text.replace(/\\n/g, '<br>').replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>');
            
            if (!isUser) {
                if (sources && sources.length) {
                    h += '<div class="sources">📚 ' + sources.map(s => s.id + ' (' + (s.similarity*100).toFixed(0) + '%)').join(', ') + '</div>';
                }
                if (webSources && webSources.length) {
                    h += '<div class="sources">🌐 ';
                    h += webSources.map(s => '<a class="web-link" href="' + s.url + '" target="_blank">' + (s.title || '링크').substring(0, 30) + '</a>').join(' | ');
                    h += '</div>';
                }
                const safe = text.replace(/'/g, "\\\\'");
                h += '<div class="msg-actions"><button onclick="speak(\\'' + safe + '\\')">🔊</button><button onclick="stopSpeak()">⏹️</button></div>';
            }
            
            d.innerHTML = h;
            c.appendChild(d);
            c.scrollTop = c.scrollHeight;
        }
        
        function showTyping() {
            const c = document.getElementById('chatBox');
            const d = document.createElement('div');
            d.className = 'message bot typing';
            d.id = 'typing';
            d.innerHTML = '<span></span><span></span><span></span>';
            c.appendChild(d);
            c.scrollTop = c.scrollHeight;
        }
        
        function hideTyping() { const t = document.getElementById('typing'); if (t) t.remove(); }
        
        async function sendMsg() {
            const input = document.getElementById('userInput');
            const q = input.value.trim();
            if (!q || processing) return;
            
            processing = true;
            document.getElementById('sendBtn').disabled = true;
            
            addMsg(q, true);
            input.value = '';
            showTyping();
            
            try {
                const res = await fetch('/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        question: q,
                        n_results: parseInt(document.getElementById('numResults').value),
                        use_web_search: document.getElementById('webToggle').checked
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
                processing = false;
                document.getElementById('sendBtn').disabled = false;
            }
        }
        
        async function loadDocs() {
            try {
                const res = await fetch('/list?limit=30');
                const data = await res.json();
                document.getElementById('docCount').textContent = data.total;
                const list = document.getElementById('docList');
                if (data.documents && data.documents.length) {
                    list.innerHTML = data.documents.map(d => 
                        '<div class="doc-item"><div class="doc-item-id">🏷️ ' + d.id + '</div><div class="doc-item-text">' + d.text + '</div><button onclick="delDoc(\\'' + d.id + '\\')">🗑️</button></div>'
                    ).join('');
                } else {
                    list.innerHTML = '<div class="empty"><div class="icon">📄</div><p>문서 없음</p></div>';
                }
            } catch(e) { document.getElementById('docList').innerHTML = '<div class="empty">로드 실패</div>'; }
        }
        
        async function addDoc() {
            const text = document.getElementById('docText').value.trim();
            if (!text) { alert('내용 입력!'); return; }
            const id = document.getElementById('docId').value.trim() || undefined;
            try {
                await fetch('/add', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({text, id}) });
                document.getElementById('docText').value = '';
                document.getElementById('docId').value = '';
                loadDocs(); checkHealth();
            } catch(e) { alert('추가 실패'); }
        }
        
        async function delDoc(id) {
            if (!confirm('삭제?')) return;
            await fetch('/delete', {method:'DELETE', headers:{'Content-Type':'application/json'}, body:JSON.stringify({id})});
            loadDocs(); checkHealth();
        }
        
        async function clearDocs() {
            if (!confirm('전체 삭제?')) return;
            await fetch('/clear', {method:'DELETE'});
            loadDocs(); checkHealth();
        }
        
        document.getElementById('userInput').addEventListener('keypress', (e) => { if (e.key === 'Enter') { e.preventDefault(); sendMsg(); } });
        document.getElementById('numResultsSet').addEventListener('change', (e) => { document.getElementById('numResults').value = e.target.value; });
    </script>
</body>
</html>
'''


# ===== API 엔드포인트 =====
@app.route('/')
def home():
    return render_template_string(MOBILE_HTML)


@app.route('/chat', methods=['POST'])
def chat():
    """채팅 API"""
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
    
    # 웹 검색
    web_results = []
    if use_web_search:
        web_results = web_search_duckduckgo(question, max_results=3)
    
    # 답변 생성
    answer, web_sources = generate_answer(question, sources, web_results, use_web_search)
    
    return jsonify({
        "question": question,
        "answer": answer,
        "sources": sources,
        "web_sources": web_sources
    })


@app.route('/add', methods=['POST'])
def add_document():
    """문서 추가"""
    data = request.json
    if not data or 'text' not in data:
        return jsonify({"error": "text 필드가 필요합니다"}), 400
    
    text = data['text']
    doc_id = data.get('id', f"doc_{len(documents) + 1}_{datetime.now().strftime('%H%M%S')}")
    metadata = data.get('metadata', {})
    
    documents[doc_id] = {"text": text, "metadata": metadata}
    rebuild_vocab()
    save_data()
    
    return jsonify({"status": "success", "id": doc_id, "total": len(documents)})


@app.route('/list', methods=['GET'])
def list_documents():
    """문서 목록"""
    limit = request.args.get('limit', 50, type=int)
    doc_list = []
    
    for i, (doc_id, doc) in enumerate(documents.items()):
        if i >= limit:
            break
        doc_list.append({
            "id": doc_id,
            "text": doc['text'][:150] + "..." if len(doc['text']) > 150 else doc['text'],
            "metadata": doc.get('metadata', {})
        })
    
    return jsonify({"total": len(documents), "documents": doc_list})


@app.route('/delete', methods=['DELETE'])
def delete_document():
    """문서 삭제"""
    data = request.json
    if not data or 'id' not in data:
        return jsonify({"error": "id 필드가 필요합니다"}), 400
    
    doc_id = data['id']
    if doc_id in documents:
        del documents[doc_id]
        rebuild_vocab()
        save_data()
        return jsonify({"status": "success", "deleted": doc_id})
    
    return jsonify({"error": "문서 없음"}), 404


@app.route('/clear', methods=['DELETE'])
def clear_all():
    """전체 삭제"""
    global documents, vocab, idf_values
    documents = {}
    vocab = {}
    idf_values = {}
    
    if os.path.exists(DOCS_FILE):
        os.remove(DOCS_FILE)
    
    return jsonify({"status": "success"})


@app.route('/health')
def health():
    """서버 상태"""
    check_ollama()
    
    return jsonify({
        "status": "running",
        "documents": len(documents),
        "ollama_available": ollama_available,
        "ollama_model": OLLAMA_MODEL if ollama_available else None,
        "web_search": "duckduckgo"
    })


# ===== 시작 =====
if __name__ == '__main__':
    print("\n" + "="*50)
    print("🎤 Voice RAG for Termux (API 키 불필요!)")
    print("="*50)
    
    # 데이터 로드
    load_data()
    
    # Ollama 확인
    check_ollama()
    
    print(f"\n📚 저장된 문서: {len(documents)}개")
    print(f"🌐 웹 검색: DuckDuckGo (무료)")
    
    if ollama_available:
        print(f"🤖 LLM: Ollama ({OLLAMA_MODEL})")
    else:
        print(f"📴 오프라인 모드 (키워드 매칭)")
        print(f"   💡 Ollama 연결하려면:")
        print(f"      export OLLAMA_HOST=http://PC_IP:11434")
    
    print(f"\n🌐 브라우저: http://localhost:5001")
    print("="*50 + "\n")
    
    app.run(host='0.0.0.0', port=5001, debug=False)
