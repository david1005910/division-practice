#!/usr/bin/env python3
"""
Voice RAG + OpenAI + Gmail + Google 웹검색 통합 시스템

주요 기능:
    1. RAG: 저장된 문서에서 관련 정보 검색
    2. OpenAI GPT: 자연어 답변 생성
    3. 🎤 Voice: 음성 인식(STT) + 음성 출력(TTS)
    4. 📧 Gmail: 메일 읽기 및 음성 요약
    5. 🌐 Google 웹검색: googlesearch-python 라이브러리

사용법:
    python app_openai_gmail_websearch_voice.py

브라우저:
    http://localhost:5001

필요한 설치:
    pip install googlesearch-python
    pip install google-auth-oauthlib google-auth-httplib2 google-api-python-client

필요한 설정:
    .env 파일에 OPENAI_API_KEY 설정
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
import base64
import time

# ===== 웹 검색 라이브러리 =====
WEBSEARCH_AVAILABLE = False
WEBSEARCH_METHOD = None

# 방법 1: googlesearch-python (권장, 무료)
try:
    from googlesearch import search as google_search
    WEBSEARCH_AVAILABLE = True
    WEBSEARCH_METHOD = "googlesearch"
    print("✅ 웹 검색: googlesearch-python 사용")
except ImportError:
    pass

# 방법 2: duckduckgo_search (백업)
if not WEBSEARCH_AVAILABLE:
    try:
        from duckduckgo_search import DDGS
        WEBSEARCH_AVAILABLE = True
        WEBSEARCH_METHOD = "duckduckgo"
        print("✅ 웹 검색: duckduckgo_search 사용")
    except ImportError:
        pass

if not WEBSEARCH_AVAILABLE:
    print("⚠️ 웹 검색 라이브러리가 설치되지 않았습니다.")
    print("   설치 (둘 중 하나 선택):")
    print("   pip install googlesearch-python  (권장)")
    print("   pip install duckduckgo_search")

# Gmail API 관련 import
try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    GMAIL_AVAILABLE = True
except ImportError:
    GMAIL_AVAILABLE = False
    print("⚠️ Gmail API 라이브러리가 설치되지 않았습니다.")
    print("   설치: pip install google-auth-oauthlib google-auth-httplib2 google-api-python-client")

app = Flask(__name__)

# ===== 설정 =====
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

# OpenAI API 설정
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

# RAG 설정
EMBEDDING_DIM = 100
MAX_ELEMENTS = 10000
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_FILE = os.path.join(DATA_DIR, "rag_index.bin")
DOCS_FILE = os.path.join(DATA_DIR, "rag_docs.json")

# Gmail API 설정
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
CREDENTIALS_FILE = os.path.join(DATA_DIR, 'credentials.json')
TOKEN_FILE = os.path.join(DATA_DIR, 'token.json')

# Gmail credentials 자동 생성
GMAIL_CREDENTIALS = {
    "installed": {
        "client_id": "610063787733-gu4h5bdfv2rl7bfkfbsmh7c20nk9h72k.apps.googleusercontent.com",
        "project_id": "my-project-1982-466022",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_secret": "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",
        "redirect_uris": ["http://localhost"]
    }
}

def ensure_credentials_file():
    if not os.path.exists(CREDENTIALS_FILE):
        with open(CREDENTIALS_FILE, 'w') as f:
            json.dump(GMAIL_CREDENTIALS, f)
        print(f"✅ credentials.json 자동 생성됨")

# 전역 변수
index = None
documents = {}
idx_to_doc_id = {}
current_idx = 0
vocab = {}
idf_values = {}
gmail_service = None


# ===== Google 웹 검색 함수 =====
def web_search_google(query, max_results=5):
    """googlesearch-python을 사용한 Google 검색"""
    try:
        results = []
        # Google 검색 수행
        search_results = list(google_search(query, num_results=max_results, lang='ko'))
        
        for url in search_results:
            # URL에서 제목 추출 시도
            title = url.split('//')[-1].split('/')[0]  # 도메인 이름
            results.append({
                'title': title,
                'snippet': f"검색 결과: {url}",
                'url': url
            })
        
        if results:
            # 각 URL에서 실제 내용 가져오기 시도
            for i, r in enumerate(results[:3]):  # 상위 3개만
                try:
                    headers = {'User-Agent': 'Mozilla/5.0 (Linux; Android 10) AppleWebKit/537.36'}
                    resp = requests.get(r['url'], headers=headers, timeout=5)
                    if resp.status_code == 200:
                        # 제목 추출
                        title_match = re.search(r'<title>([^<]+)</title>', resp.text, re.IGNORECASE)
                        if title_match:
                            results[i]['title'] = title_match.group(1).strip()[:100]
                        
                        # 본문 일부 추출
                        text = re.sub(r'<script[^>]*>.*?</script>', '', resp.text, flags=re.DOTALL|re.IGNORECASE)
                        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL|re.IGNORECASE)
                        text = re.sub(r'<[^>]+>', ' ', text)
                        text = re.sub(r'\s+', ' ', text).strip()
                        if len(text) > 100:
                            results[i]['snippet'] = text[:300] + "..."
                except:
                    pass
                time.sleep(0.3)  # 요청 간격
            
            return results, None
        else:
            return [], "검색 결과가 없습니다."
            
    except Exception as e:
        return [], f"Google 검색 오류: {str(e)}"


def web_search_duckduckgo(query, max_results=5):
    """DuckDuckGo 검색 (백업)"""
    try:
        with DDGS() as ddgs:
            results = []
            for r in ddgs.text(query, region='kr-kr', max_results=max_results):
                results.append({
                    'title': r.get('title', ''),
                    'snippet': r.get('body', ''),
                    'url': r.get('href', '')
                })
            return results, None if results else ([], "검색 결과 없음")
    except Exception as e:
        return [], f"DuckDuckGo 검색 오류: {str(e)}"


def web_search(query, max_results=5):
    """통합 웹 검색 함수"""
    if not WEBSEARCH_AVAILABLE:
        return [], "웹 검색 라이브러리가 설치되지 않았습니다.\npip install googlesearch-python"
    
    if WEBSEARCH_METHOD == "googlesearch":
        return web_search_google(query, max_results)
    elif WEBSEARCH_METHOD == "duckduckgo":
        return web_search_duckduckgo(query, max_results)
    else:
        return [], "웹 검색을 사용할 수 없습니다."


def needs_web_search(question):
    """질문이 웹 검색이 필요한지 판단"""
    keywords = [
        '검색', '찾아', '알아봐', '조사해', '인터넷', '구글',
        '오늘', '현재', '지금', '최근', '요즘', '올해',
        '2024', '2025', '2026',
        '뉴스', '날씨', '주가', '환율', '주식', '가격', '시세',
        'search', 'find', 'google', 'today', 'now', 'current', 'recent', 'news', 'weather'
    ]
    question_lower = question.lower()
    return any(kw in question_lower for kw in keywords)


def is_news_request(question):
    """뉴스 검색 요청인지 확인"""
    return any(kw in question.lower() for kw in ['뉴스', '소식', '기사', 'news', '헤드라인'])


# ===== Gmail API 함수 =====
def init_gmail_service():
    global gmail_service
    if not GMAIL_AVAILABLE:
        return False
    ensure_credentials_file()
    if not os.path.exists(CREDENTIALS_FILE):
        return False
    
    creds = None
    if os.path.exists(TOKEN_FILE):
        try:
            creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
        except:
            pass
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except:
                creds = None
        if not creds:
            try:
                flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
                creds = flow.run_console()
            except Exception as e:
                print(f"Gmail 인증 실패: {e}")
                return False
        with open(TOKEN_FILE, 'w') as token:
            token.write(creds.to_json())
    
    try:
        gmail_service = build('gmail', 'v1', credentials=creds)
        print("✅ Gmail API 연결 성공!")
        return True
    except:
        return False


def get_recent_emails(max_results=5):
    global gmail_service
    if not gmail_service and not init_gmail_service():
        return None, "Gmail API 미설정. 서버 재시작 후 터미널에서 인증하세요."
    
    try:
        results = gmail_service.users().messages().list(
            userId='me', labelIds=['INBOX'], maxResults=max_results
        ).execute()
        messages = results.get('messages', [])
        if not messages:
            return [], "메일 없음"
        
        emails = []
        for msg in messages:
            message = gmail_service.users().messages().get(
                userId='me', id=msg['id'], format='full'
            ).execute()
            headers = message.get('payload', {}).get('headers', [])
            email_data = {
                'id': msg['id'], 'subject': '', 'from': '', 'date': '',
                'snippet': message.get('snippet', ''), 'body': ''
            }
            for h in headers:
                name = h['name'].lower()
                if name == 'subject': email_data['subject'] = h['value']
                elif name == 'from': email_data['from'] = h['value']
                elif name == 'date': email_data['date'] = h['value']
            body = extract_email_body(message.get('payload', {}))
            email_data['body'] = body[:1000] if body else email_data['snippet']
            emails.append(email_data)
        return emails, None
    except Exception as e:
        if 'invalid_grant' in str(e):
            if os.path.exists(TOKEN_FILE): os.remove(TOKEN_FILE)
            gmail_service = None
            return None, "Gmail 인증 만료. 서버 재시작 필요"
        return None, f"메일 오류: {str(e)}"


def extract_email_body(payload):
    body = ""
    if 'body' in payload and payload['body'].get('data'):
        body = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8', errors='ignore')
    elif 'parts' in payload:
        for part in payload['parts']:
            mime = part.get('mimeType', '')
            if mime == 'text/plain' and 'data' in part.get('body', {}):
                body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8', errors='ignore')
                break
            elif mime == 'text/html' and not body and 'data' in part.get('body', {}):
                html = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8', errors='ignore')
                body = re.sub(r'<[^>]+>', '', html)
            elif 'parts' in part:
                body = extract_email_body(part)
                if body: break
    return re.sub(r'\s+', ' ', body).strip()


def summarize_emails_with_openai(emails):
    if not emails: return "요약할 메일 없음"
    if not OPENAI_API_KEY: return "⚠️ OpenAI API 키가 필요합니다."
    
    email_text = "\n".join([
        f"[메일 {i+1}] 보낸이: {e['from']}, 제목: {e['subject']}, 내용: {e['body'][:300]}..."
        for i, e in enumerate(emails)
    ])
    
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": OPENAI_MODEL,
                "messages": [
                    {"role": "system", "content": "이메일 요약 비서. 각 메일 핵심 1-2문장, 한국어, 자연스럽게"},
                    {"role": "user", "content": f"최근 메일 {len(emails)}개 요약해줘:\n{email_text}"}
                ],
                "max_tokens": 1024
            },
            timeout=60
        )
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        return f"⚠️ 요약 실패: {response.status_code}"
    except Exception as e:
        return f"⚠️ 오류: {str(e)}"


def is_email_summary_request(question):
    email_kw = ['메일', '이메일', 'email', 'mail', '편지함', '받은편지', '인박스']
    action_kw = ['요약', '알려', '읽어', '확인', '보여', '뭐 왔', '왔어', '있어']
    q = question.lower()
    return any(k in q for k in email_kw) and any(k in q for k in action_kw)


# ===== RAG 함수 =====
def tokenize(text):
    text = re.sub(r'[^\w\s가-힣]', ' ', text.lower())
    stopwords = {'the', 'a', 'an', 'is', 'are', 'to', 'of', 'in', 'for', 'on', 'with',
                 '이', '가', '은', '는', '을', '를', '의', '에', '에서', '으로'}
    return [t for t in text.split() if t not in stopwords and len(t) > 1]


def build_vocab(all_texts):
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
    idf_values = {word: math.log(n_docs / (count + 1)) + 1 for word, count in word_doc_count.items()}


def text_to_embedding(text):
    tokens = tokenize(text)
    if not tokens:
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)
    tf = Counter(tokens)
    total = len(tokens)
    embedding = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    for word, count in tf.items():
        if word in vocab:
            embedding[vocab[word]] = (count / total) * idf_values.get(word, 1.0)
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm > 0 else embedding


def init_index():
    global index
    index = hnswlib.Index(space='cosine', dim=EMBEDDING_DIM)
    index.init_index(max_elements=MAX_ELEMENTS, ef_construction=200, M=16)
    index.set_ef(50)


def save_data():
    if index and index.get_current_count() > 0:
        index.save_index(INDEX_FILE)
    with open(DOCS_FILE, 'w', encoding='utf-8') as f:
        json.dump({
            "documents": documents,
            "idx_to_doc_id": {str(k): v for k, v in idx_to_doc_id.items()},
            "current_idx": current_idx, "vocab": vocab, "idf_values": idf_values
        }, f, ensure_ascii=False)


def load_data():
    global documents, idx_to_doc_id, current_idx, vocab, idf_values
    init_index()
    if os.path.exists(DOCS_FILE):
        try:
            with open(DOCS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            documents = data.get("documents", {})
            idx_to_doc_id = {int(k): v for k, v in data.get("idx_to_doc_id", {}).items()}
            current_idx = data.get("current_idx", 0)
            vocab = data.get("vocab", {})
            idf_values = data.get("idf_values", {})
            if os.path.exists(INDEX_FILE) and documents:
                index.load_index(INDEX_FILE, max_elements=MAX_ELEMENTS)
        except:
            pass


def rebuild_index():
    global current_idx, idx_to_doc_id
    if not documents:
        return
    build_vocab([doc["text"] for doc in documents.values()])
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


def rag_search(query, n=3):
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
                "id": doc_id, "text": doc['text'],
                "similarity": round(1 - distance, 4),
                "metadata": doc.get('metadata', {})
            })
    return results


# ===== OpenAI API (웹 검색 포함) =====
def ask_openai(question, context_docs, use_web_search=False):
    if not OPENAI_API_KEY:
        return "⚠️ OpenAI API 키가 설정되지 않았습니다.\n.env 파일에 OPENAI_API_KEY를 설정해주세요.", []
    
    web_results = []
    web_context = ""
    
    # 웹 검색 수행
    if use_web_search and WEBSEARCH_AVAILABLE:
        print(f"🔍 웹 검색 중: {question}")
        results, error = web_search(question)
        if error:
            print(f"   검색 오류: {error}")
        if results:
            web_results = results
            web_context = "\n\n=== 🌐 웹 검색 결과 ===\n"
            for i, r in enumerate(results, 1):
                web_context += f"[{i}] {r['title']}\n{r['snippet']}\nURL: {r['url']}\n\n"
            print(f"   ✅ {len(results)}개 결과 찾음")
    
    # 프롬프트 구성
    context = ""
    if context_docs:
        context += "=== 📚 RAG 문서 ===\n"
        context += "\n".join([f"[문서 {i+1}] {doc['text']}" for i, doc in enumerate(context_docs)])
    context += web_context
    
    if context:
        system_prompt = """당신은 AI 어시스턴트입니다.
제공된 문서와 웹 검색 결과를 참고하여 답변하세요.
웹 검색 결과가 있으면 그 정보를 우선적으로 활용하세요.
답변은 친절하고 자연스럽게 한국어로 해주세요."""
        user_prompt = f"{context}\n\n질문: {question}\n\n위 정보를 참고하여 답변해주세요."
    else:
        system_prompt = "당신은 친절한 AI 어시스턴트입니다. 한국어로 답변해주세요."
        user_prompt = question

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": OPENAI_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "max_tokens": 2048,
                "temperature": 0.7
            },
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"], web_results
        else:
            error_msg = response.json().get("error", {}).get("message", str(response.status_code))
            return f"⚠️ API 오류: {error_msg}", []
    except Exception as e:
        return f"⚠️ 오류: {str(e)}", []


# ===== HTML 템플릿 =====
MOBILE_APP_HTML = '''
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>🎤 Voice RAG + GPT + 📧메일 + 🌐웹</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, sans-serif; background: linear-gradient(135deg, #1a1a2e, #16213e); min-height: 100vh; color: white; }
        .app { display: flex; flex-direction: column; height: 100vh; }
        
        .header { background: linear-gradient(135deg, #10a37f, #1a7f5a); padding: 12px; text-align: center; }
        .header h1 { font-size: 1rem; }
        .status { font-size: 0.7rem; opacity: 0.9; margin-top: 4px; }
        .status-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #ff4757; margin-right: 4px; }
        .status-dot.ok { background: #2ed573; }
        .badges { display: flex; gap: 5px; justify-content: center; margin-top: 6px; flex-wrap: wrap; }
        .badge { padding: 2px 8px; border-radius: 10px; font-size: 0.6rem; font-weight: bold; }
        .badge-web { background: rgba(139,233,253,0.3); color: #8be9fd; }
        .badge-mail { background: rgba(255,184,108,0.3); color: #ffb86c; }
        .badge-voice { background: rgba(80,250,123,0.3); color: #50fa7b; }
        
        .tabs { display: flex; background: #1a1a2e; border-bottom: 1px solid #333; }
        .tab { flex: 1; padding: 10px; text-align: center; background: transparent; border: none; color: #888; font-size: 0.8rem; cursor: pointer; }
        .tab.active { color: #10a37f; border-bottom: 2px solid #10a37f; }
        
        .chat-container { flex: 1; overflow-y: auto; padding: 12px; display: flex; flex-direction: column; gap: 10px; }
        .message { max-width: 88%; padding: 10px 14px; border-radius: 16px; line-height: 1.5; font-size: 0.9rem; word-wrap: break-word; }
        .message.user { background: linear-gradient(135deg, #10a37f, #1a7f5a); align-self: flex-end; }
        .message.bot { background: #2a2a4a; align-self: flex-start; border: 1px solid #333; }
        .message .sources { font-size: 0.7rem; color: #888; margin-top: 6px; padding-top: 6px; border-top: 1px solid #444; }
        .message .web-sources { font-size: 0.7rem; color: #8be9fd; margin-top: 4px; }
        .message .web-sources a { color: #8be9fd; text-decoration: none; }
        .message .mail-info { font-size: 0.7rem; color: #ffb86c; margin-top: 4px; }
        .message-actions { display: flex; gap: 6px; margin-top: 6px; }
        .message-actions button { background: rgba(16,163,127,0.2); border: 1px solid #10a37f; color: #10a37f; padding: 3px 8px; border-radius: 10px; font-size: 0.65rem; cursor: pointer; }
        
        .typing { display: flex; gap: 4px; padding: 12px; }
        .typing span { width: 8px; height: 8px; background: #10a37f; border-radius: 50%; animation: bounce 1.4s infinite; }
        @keyframes bounce { 0%, 80%, 100% { transform: scale(0); } 40% { transform: scale(1); } }
        .typing span:nth-child(2) { animation-delay: 0.2s; }
        .typing span:nth-child(3) { animation-delay: 0.4s; }
        
        .input-container { background: #1a1a2e; padding: 10px; border-top: 1px solid #333; }
        .quick-actions { display: flex; gap: 6px; margin-bottom: 8px; flex-wrap: wrap; }
        .quick-btn { padding: 5px 10px; background: rgba(139,233,253,0.15); border: 1px solid #8be9fd; color: #8be9fd; border-radius: 12px; font-size: 0.7rem; cursor: pointer; }
        .quick-btn.mail { background: rgba(255,184,108,0.15); border-color: #ffb86c; color: #ffb86c; }
        
        .search-options { display: flex; gap: 8px; margin-bottom: 8px; align-items: center; flex-wrap: wrap; }
        .search-options label { font-size: 0.7rem; color: #888; }
        .search-options select { padding: 4px 8px; background: #2a2a4a; border: 1px solid #333; border-radius: 6px; color: white; font-size: 0.7rem; }
        .toggle-switch { position: relative; width: 36px; height: 20px; display: inline-block; vertical-align: middle; }
        .toggle-switch input { opacity: 0; width: 0; height: 0; }
        .toggle-slider { position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0; background: #333; border-radius: 20px; transition: 0.3s; }
        .toggle-slider:before { content: ""; position: absolute; height: 14px; width: 14px; left: 3px; bottom: 3px; background: white; border-radius: 50%; transition: 0.3s; }
        input:checked + .toggle-slider { background: #10a37f; }
        input:checked + .toggle-slider:before { transform: translateX(16px); }
        
        .input-row { display: flex; gap: 6px; align-items: center; }
        .voice-btn { width: 44px; height: 44px; border-radius: 50%; border: none; background: linear-gradient(135deg, #10a37f, #1a7f5a); color: white; font-size: 1.1rem; cursor: pointer; }
        .voice-btn.recording { background: linear-gradient(135deg, #ff4757, #ff6b81); animation: pulse 1s infinite; }
        @keyframes pulse { 0%, 100% { box-shadow: 0 0 0 0 rgba(255,71,87,0.4); } 50% { box-shadow: 0 0 0 10px rgba(255,71,87,0); } }
        .mail-btn { width: 44px; height: 44px; border-radius: 50%; border: none; background: linear-gradient(135deg, #ffb86c, #ff9f43); color: white; font-size: 1.1rem; cursor: pointer; }
        .text-input { flex: 1; padding: 10px 14px; background: #2a2a4a; border: 2px solid #333; border-radius: 20px; color: white; font-size: 0.9rem; }
        .text-input:focus { outline: none; border-color: #10a37f; }
        .send-btn { padding: 10px 14px; background: linear-gradient(135deg, #10a37f, #1a7f5a); border: none; border-radius: 20px; color: white; font-weight: bold; cursor: pointer; }
        .send-btn:disabled { opacity: 0.5; }
        
        .tab-content { flex: 1; overflow-y: auto; padding: 12px; display: none; }
        .tab-content.active { display: block; }
        .doc-input { width: 100%; padding: 10px; background: #2a2a4a; border: 1px solid #333; border-radius: 8px; color: white; margin-bottom: 8px; }
        textarea.doc-input { min-height: 80px; resize: vertical; }
        .doc-buttons { display: flex; gap: 6px; margin-bottom: 10px; }
        .doc-buttons button { padding: 8px 14px; border: none; border-radius: 8px; font-weight: bold; cursor: pointer; font-size: 0.75rem; }
        .btn-add { background: #10a37f; color: white; }
        .btn-clear { background: #ff4757; color: white; }
        .doc-item { background: #2a2a4a; padding: 10px; border-radius: 8px; margin-bottom: 8px; border-left: 3px solid #10a37f; }
        .doc-item-id { color: #10a37f; font-size: 0.7rem; font-weight: bold; }
        .doc-item-text { color: #ccc; font-size: 0.8rem; margin-top: 4px; }
        .setting-item { background: #2a2a4a; padding: 12px; border-radius: 8px; margin-bottom: 8px; }
        .setting-item label { display: block; color: #888; font-size: 0.75rem; margin-bottom: 6px; }
        .setting-item select { width: 100%; padding: 8px; background: #1a1a2e; border: 1px solid #333; border-radius: 6px; color: white; }
    </style>
</head>
<body>
    <div class="app">
        <div class="header">
            <h1>🎤 Voice RAG + GPT + 📧 + 🌐</h1>
            <div class="status">
                <span class="status-dot" id="statusDot"></span>
                <span id="statusText">연결 확인 중...</span>
                | 📚 <span id="docCount">0</span>개
                | 🌐 <span id="webStatus">-</span>
            </div>
            <div class="badges">
                <span class="badge badge-web">🌐 Google검색</span>
                <span class="badge badge-voice">🎤 음성</span>
                <span class="badge badge-mail">📧 메일</span>
            </div>
        </div>
        
        <div class="tabs">
            <button class="tab active" onclick="showTab('chat')">💬 채팅</button>
            <button class="tab" onclick="showTab('docs')">📄 문서</button>
            <button class="tab" onclick="showTab('settings')">⚙️ 설정</button>
        </div>
        
        <div class="chat-container" id="chatContainer">
            <div class="message bot">
                안녕하세요! 🤖 AI 어시스턴트입니다.<br><br>
                🌐 <b>웹검색</b>: "오늘 뉴스", "날씨" 등<br>
                📧 <b>메일</b>: "메일 요약해줘"<br>
                🎤 <b>음성</b>: 마이크 버튼 클릭<br><br>
                무엇을 도와드릴까요?
            </div>
        </div>
        
        <div class="tab-content" id="docsTab">
            <input type="text" class="doc-input" id="docId" placeholder="문서 ID (선택)">
            <textarea class="doc-input" id="docText" placeholder="문서 내용..."></textarea>
            <div class="doc-buttons">
                <button class="btn-add" onclick="addDoc()">➕ 추가</button>
                <button class="btn-clear" onclick="clearDocs()">🗑️ 전체삭제</button>
            </div>
            <div id="docList"></div>
        </div>
        
        <div class="tab-content" id="settingsTab">
            <div class="setting-item">
                <label>📄 RAG 검색 결과 수</label>
                <select id="numResultsSetting"><option value="0">OFF</option><option value="3" selected>3개</option><option value="5">5개</option></select>
            </div>
            <div class="setting-item">
                <label>📧 메일 요약 개수</label>
                <select id="mailCountSetting"><option value="3">3개</option><option value="5" selected>5개</option><option value="10">10개</option></select>
            </div>
            <div class="setting-item">
                <label>🔊 음성 자동 읽기</label>
                <select id="autoSpeak"><option value="true" selected>켜기</option><option value="false">끄기</option></select>
            </div>
            <div class="setting-item">
                <label>⏩ 음성 속도</label>
                <select id="speechRate"><option value="0.8">느리게</option><option value="1.0" selected>보통</option><option value="1.2">빠르게</option></select>
            </div>
        </div>
        
        <div class="input-container" id="inputContainer">
            <div class="quick-actions">
                <button class="quick-btn mail" onclick="quickMail()">📧 메일 요약</button>
                <button class="quick-btn" onclick="quickNews()">📰 오늘 뉴스</button>
                <button class="quick-btn" onclick="quickWeather()">🌤️ 날씨</button>
                <button class="quick-btn" onclick="quickSearch()">🔍 검색</button>
            </div>
            <div class="search-options">
                <label>📄 RAG:</label>
                <select id="numResults"><option value="0">OFF</option><option value="3" selected>3개</option></select>
                <label style="margin-left:8px">🌐 웹:</label>
                <label class="toggle-switch"><input type="checkbox" id="webSearchToggle" checked><span class="toggle-slider"></span></label>
            </div>
            <div class="input-row">
                <button class="voice-btn" id="voiceBtn" onclick="toggleVoice()">🎤</button>
                <button class="mail-btn" onclick="quickMail()">📧</button>
                <input type="text" class="text-input" id="userInput" placeholder="질문을 입력하세요...">
                <button class="send-btn" id="sendBtn" onclick="sendMessage()">전송</button>
            </div>
        </div>
    </div>

    <script>
        let recognition = null, isRecording = false, isProcessing = false;
        
        checkHealth();
        initSpeech();
        
        function showTab(name) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            event.target.classList.add('active');
            document.getElementById('chatContainer').style.display = name === 'chat' ? 'flex' : 'none';
            document.getElementById('docsTab').classList.toggle('active', name === 'docs');
            document.getElementById('settingsTab').classList.toggle('active', name === 'settings');
            document.getElementById('inputContainer').style.display = name === 'chat' ? 'block' : 'none';
            if (name === 'docs') loadDocs();
        }
        
        async function checkHealth() {
            try {
                const res = await fetch('/health');
                const data = await res.json();
                document.getElementById('statusDot').classList.toggle('ok', data.llm_available);
                document.getElementById('statusText').textContent = data.llm_available ? 'OpenAI 연결' : 'API 키 필요';
                document.getElementById('docCount').textContent = data.documents || 0;
                document.getElementById('webStatus').textContent = data.web_search_available ? '사용가능' : '미설치';
            } catch(e) {
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
                recognition.onresult = (e) => { document.getElementById('userInput').value = e.results[0][0].transcript; };
                recognition.onend = () => {
                    isRecording = false;
                    document.getElementById('voiceBtn').classList.remove('recording');
                    if (document.getElementById('userInput').value.trim() && !isProcessing) sendMessage();
                };
                recognition.onerror = () => { isRecording = false; document.getElementById('voiceBtn').classList.remove('recording'); };
            }
        }
        
        function toggleVoice() {
            if (!recognition) { alert('음성 인식 미지원'); return; }
            if (isRecording) { recognition.stop(); }
            else { recognition.start(); isRecording = true; document.getElementById('voiceBtn').classList.add('recording'); }
        }
        
        function speak(text) {
            if (!('speechSynthesis' in window) || document.getElementById('autoSpeak').value !== 'true') return;
            speechSynthesis.cancel();
            const u = new SpeechSynthesisUtterance(text.replace(/[*#`]/g, '').replace(/\\n/g, ' '));
            u.lang = 'ko-KR';
            u.rate = parseFloat(document.getElementById('speechRate').value);
            speechSynthesis.speak(u);
        }
        
        function stopSpeak() { speechSynthesis.cancel(); }
        
        function addMsg(text, isUser, sources = [], webSources = [], mailInfo = null) {
            const c = document.getElementById('chatContainer');
            const d = document.createElement('div');
            d.className = 'message ' + (isUser ? 'user' : 'bot');
            let h = text.replace(/\\n/g, '<br>').replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>');
            if (!isUser) {
                if (sources && sources.length) h += '<div class="sources">📚 ' + sources.map(s => s.id).join(', ') + '</div>';
                if (webSources && webSources.length) h += '<div class="web-sources">🌐 ' + webSources.slice(0,3).map(s => '<a href="'+s.url+'" target="_blank">'+(s.title||'링크').substring(0,25)+'</a>').join(' | ') + '</div>';
                if (mailInfo) h += '<div class="mail-info">📧 ' + mailInfo + '</div>';
                h += '<div class="message-actions"><button onclick="speak(`'+text.replace(/`/g,"'").replace(/\\n/g,' ')+'`)">🔊</button><button onclick="stopSpeak()">⏹️</button></div>';
            }
            d.innerHTML = h;
            c.appendChild(d);
            c.scrollTop = c.scrollHeight;
        }
        
        function showTyping() {
            const d = document.createElement('div');
            d.className = 'message bot typing';
            d.id = 'typing';
            d.innerHTML = '<span></span><span></span><span></span>';
            document.getElementById('chatContainer').appendChild(d);
        }
        
        function hideTyping() { const t = document.getElementById('typing'); if (t) t.remove(); }
        
        function quickMail() { document.getElementById('userInput').value = '메일 요약해줘'; sendMessage(); }
        function quickNews() { document.getElementById('userInput').value = '오늘 주요 뉴스 알려줘'; sendMessage(); }
        function quickWeather() { document.getElementById('userInput').value = '서울 날씨 알려줘'; sendMessage(); }
        function quickSearch() { const q = prompt('검색어 입력:'); if (q) { document.getElementById('userInput').value = q + ' 검색해줘'; sendMessage(); } }
        
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
                const res = await fetch('/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        question: q,
                        n_results: parseInt(document.getElementById('numResults').value),
                        use_web_search: document.getElementById('webSearchToggle').checked,
                        mail_count: parseInt(document.getElementById('mailCountSetting')?.value || 5)
                    })
                });
                hideTyping();
                const data = await res.json();
                const mailInfo = data.mail_used ? data.mail_count + '개 메일' : null;
                addMsg(data.answer, false, data.sources, data.web_sources || [], mailInfo);
                speak(data.answer);
            } catch(e) {
                hideTyping();
                addMsg('⚠️ 오류: ' + e.message, false);
            } finally {
                isProcessing = false;
                document.getElementById('sendBtn').disabled = false;
            }
        }
        
        async function loadDocs() {
            try {
                const data = await (await fetch('/list?limit=50')).json();
                document.getElementById('docCount').textContent = data.total;
                const list = document.getElementById('docList');
                list.innerHTML = data.documents?.length ? data.documents.map(d => '<div class="doc-item"><div class="doc-item-id">'+d.id+'</div><div class="doc-item-text">'+d.text+'</div></div>').join('') : '<p style="color:#666">문서 없음</p>';
            } catch(e) {}
        }
        
        async function addDoc() {
            const text = document.getElementById('docText').value.trim();
            if (!text) return alert('내용 입력');
            await fetch('/add', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({text, id: document.getElementById('docId').value.trim() || undefined}) });
            document.getElementById('docText').value = '';
            document.getElementById('docId').value = '';
            loadDocs();
        }
        
        async function clearDocs() {
            if (!confirm('전체 삭제?')) return;
            await fetch('/clear', {method:'DELETE'});
            loadDocs();
        }
        
        document.getElementById('userInput').addEventListener('keypress', (e) => { if (e.key === 'Enter') sendMessage(); });
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
    data = request.json
    question = data.get('question', '')
    n_results = data.get('n_results', 3)
    use_web_search = data.get('use_web_search', True)
    mail_count = data.get('mail_count', 5)
    
    if not question:
        return jsonify({"error": "질문을 입력해주세요"}), 400
    
    # 메일 요약 요청
    if is_email_summary_request(question):
        emails, error = get_recent_emails(max_results=mail_count)
        if error:
            return jsonify({"question": question, "answer": f"📧 메일 오류: {error}", "sources": [], "web_sources": [], "mail_used": False})
        if emails:
            summary = summarize_emails_with_openai(emails)
            return jsonify({"question": question, "answer": summary, "sources": [], "web_sources": [], "mail_used": True, "mail_count": len(emails)})
    
    # 일반 질문
    sources = rag_search(question, n=n_results) if n_results > 0 else []
    should_use_web = use_web_search and needs_web_search(question)
    answer, web_sources = ask_openai(question, sources, use_web_search=should_use_web)
    
    return jsonify({"question": question, "answer": answer, "sources": sources, "web_sources": web_sources, "mail_used": False})


@app.route('/add', methods=['POST'])
def add_document():
    global current_idx
    data = request.json
    if not data or 'text' not in data:
        return jsonify({"error": "text 필요"}), 400
    text = data['text']
    doc_id = data.get('id', f"doc_{len(documents)+1}")
    if doc_id in documents:
        documents[doc_id] = {"text": text, "metadata": {}}
        rebuild_index()
        return jsonify({"status": "updated", "id": doc_id})
    tokens = tokenize(text)
    for word in tokens:
        if word not in vocab and len(vocab) < EMBEDDING_DIM:
            vocab[word] = len(vocab)
        if word not in idf_values:
            idf_values[word] = 1.0
    embedding = text_to_embedding(text)
    index.add_items(np.array([embedding]), [current_idx])
    documents[doc_id] = {"text": text, "metadata": {}, "idx": current_idx}
    idx_to_doc_id[current_idx] = doc_id
    current_idx += 1
    save_data()
    return jsonify({"status": "success", "id": doc_id, "total": len(documents)})


@app.route('/list', methods=['GET'])
def list_documents():
    limit = request.args.get('limit', 100, type=int)
    doc_list = [{"id": doc_id, "text": doc['text'][:200] + "..." if len(doc['text']) > 200 else doc['text']}
                for i, (doc_id, doc) in enumerate(documents.items()) if i < limit]
    return jsonify({"total": len(documents), "documents": doc_list})


@app.route('/delete', methods=['DELETE'])
def delete_document():
    doc_id = request.json.get('id')
    if doc_id in documents:
        del documents[doc_id]
        rebuild_index()
        return jsonify({"status": "deleted"})
    return jsonify({"error": "not found"}), 404


@app.route('/clear', methods=['DELETE'])
def clear_all():
    global documents, idx_to_doc_id, current_idx, vocab, idf_values
    documents, idx_to_doc_id, current_idx, vocab, idf_values = {}, {}, 0, {}, {}
    init_index()
    for f in [INDEX_FILE, DOCS_FILE]:
        if os.path.exists(f): os.remove(f)
    return jsonify({"status": "cleared"})


@app.route('/health')
def health():
    return jsonify({
        "status": "running",
        "documents": len(documents),
        "llm_available": bool(OPENAI_API_KEY),
        "model": OPENAI_MODEL,
        "gmail_available": gmail_service is not None,
        "web_search_available": WEBSEARCH_AVAILABLE,
        "web_search_method": WEBSEARCH_METHOD
    })


# ===== 시작 =====
load_data()
ensure_credentials_file()
if GMAIL_AVAILABLE:
    init_gmail_service()

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🎤 Voice RAG + OpenAI + 📧Gmail + 🌐Google검색")
    print("="*60)
    print(f"🌐 웹 UI: http://localhost:5001")
    print(f"📚 문서: {len(documents)}개")
    print(f"🧠 모델: {OPENAI_MODEL}")
    
    print("\n📌 상태:")
    print(f"   {'✅' if OPENAI_API_KEY else '❌'} OpenAI API")
    print(f"   {'✅' if WEBSEARCH_AVAILABLE else '❌'} 웹 검색 ({WEBSEARCH_METHOD or '미설치'})")
    print(f"   {'✅' if gmail_service else '❌'} Gmail API")
    
    if not WEBSEARCH_AVAILABLE:
        print("\n⚠️  웹 검색 설치 필요:")
        print("   pip install googlesearch-python")
    
    print("\n" + "="*60)
    print("💡 사용법: '오늘 뉴스', '날씨', '메일 요약해줘'")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5001, debug=False)
