#!/usr/bin/env python3
"""
Voice RAG + OpenAI + Gmail 메일요약 통합 시스템 (All-in-One)

주요 기능:
    1. RAG: 저장된 문서에서 관련 정보 검색
    2. OpenAI GPT: 자연어 답변 생성
    3. 🎤 Voice: 음성 인식(STT) + 음성 출력(TTS)
    4. 📧 Gmail: 메일 읽기 및 음성 요약

사용법:
    python app_openai_gmail_voice.py

브라우저:
    http://localhost:5001

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

# 웹 검색 라이브러리
try:
    from duckduckgo_search import DDGS
    WEBSEARCH_AVAILABLE = True
except ImportError:
    WEBSEARCH_AVAILABLE = False
    print("⚠️ 웹 검색 라이브러리가 설치되지 않았습니다.")
    print("   설치: pip install duckduckgo_search")

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

# ===== Gmail credentials.json 자동 생성 =====
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
    """credentials.json 파일이 없으면 자동 생성"""
    if not os.path.exists(CREDENTIALS_FILE):
        with open(CREDENTIALS_FILE, 'w') as f:
            json.dump(GMAIL_CREDENTIALS, f)
        print(f"✅ credentials.json 자동 생성됨: {CREDENTIALS_FILE}")

# ===== RAG 전역 변수 =====
index = None
documents = {}
idx_to_doc_id = {}
current_idx = 0
vocab = {}
idf_values = {}

# Gmail 서비스 전역 변수
gmail_service = None


# ===== 웹 검색 함수 (duckduckgo_search 라이브러리) =====
def web_search(query, max_results=5):
    """DuckDuckGo 웹 검색"""
    if not WEBSEARCH_AVAILABLE:
        return [], "웹 검색 라이브러리가 설치되지 않았습니다. pip install duckduckgo_search"
    
    try:
        with DDGS() as ddgs:
            results = []
            for r in ddgs.text(query, region='kr-kr', max_results=max_results):
                results.append({
                    'title': r.get('title', ''),
                    'snippet': r.get('body', ''),
                    'url': r.get('href', '')
                })
            
            if results:
                return results, None
            else:
                return [], "검색 결과가 없습니다."
                
    except Exception as e:
        error_msg = str(e)
        print(f"웹 검색 오류: {error_msg}")
        return [], f"검색 오류: {error_msg}"


def web_search_news(query, max_results=5):
    """DuckDuckGo 뉴스 검색"""
    if not WEBSEARCH_AVAILABLE:
        return [], "웹 검색 라이브러리가 설치되지 않았습니다."
    
    try:
        with DDGS() as ddgs:
            results = []
            for r in ddgs.news(query, region='kr-kr', max_results=max_results):
                results.append({
                    'title': r.get('title', ''),
                    'snippet': r.get('body', ''),
                    'url': r.get('url', ''),
                    'date': r.get('date', '')
                })
            
            if results:
                return results, None
            else:
                return [], "뉴스 검색 결과가 없습니다."
                
    except Exception as e:
        return [], f"뉴스 검색 오류: {str(e)}"


def needs_web_search(question):
    """질문이 웹 검색이 필요한지 판단"""
    keywords = [
        '검색', '찾아', '알아봐', '조사해', '인터넷',
        '오늘', '현재', '지금', '최근', '요즘', '올해',
        '2024', '2025', '2026',
        '뉴스', '날씨', '주가', '환율', '주식', '가격', '시세',
        'search', 'find', 'today', 'now', 'current', 'recent', 'latest', 'news', 'weather'
    ]
    question_lower = question.lower()
    return any(kw in question_lower for kw in keywords)


def is_news_request(question):
    """뉴스 검색 요청인지 확인"""
    news_keywords = ['뉴스', '소식', '기사', 'news', '헤드라인', '속보']
    return any(kw in question.lower() for kw in news_keywords)


# ===== Gmail API 함수 =====
def init_gmail_service():
    """Gmail API 서비스 초기화"""
    global gmail_service
    
    if not GMAIL_AVAILABLE:
        return False
    
    ensure_credentials_file()
    
    if not os.path.exists(CREDENTIALS_FILE):
        print(f"⚠️ Gmail credentials.json 파일이 없습니다: {CREDENTIALS_FILE}")
        return False
    
    creds = None
    
    if os.path.exists(TOKEN_FILE):
        try:
            creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
        except Exception as e:
            print(f"토큰 로드 실패: {e}")
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                print(f"토큰 갱신 실패: {e}")
                creds = None
        
        if not creds:
            try:
                flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
                # Termux 환경을 위해 run_console() 사용
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
    except Exception as e:
        print(f"Gmail 서비스 생성 실패: {e}")
        return False


def get_recent_emails(max_results=5):
    """최근 이메일 가져오기"""
    global gmail_service
    
    if not gmail_service:
        if not init_gmail_service():
            return None, "Gmail API가 설정되지 않았습니다. 첫 실행 시 인증이 필요합니다."
    
    try:
        results = gmail_service.users().messages().list(
            userId='me',
            labelIds=['INBOX'],
            maxResults=max_results
        ).execute()
        
        messages = results.get('messages', [])
        
        if not messages:
            return [], "받은편지함에 메일이 없습니다."
        
        emails = []
        for msg in messages:
            message = gmail_service.users().messages().get(
                userId='me',
                id=msg['id'],
                format='full'
            ).execute()
            
            headers = message.get('payload', {}).get('headers', [])
            
            email_data = {
                'id': msg['id'],
                'subject': '',
                'from': '',
                'date': '',
                'snippet': message.get('snippet', ''),
                'body': ''
            }
            
            for header in headers:
                name = header.get('name', '').lower()
                value = header.get('value', '')
                if name == 'subject':
                    email_data['subject'] = value
                elif name == 'from':
                    email_data['from'] = value
                elif name == 'date':
                    email_data['date'] = value
            
            body = extract_email_body(message.get('payload', {}))
            email_data['body'] = body[:1000] if body else email_data['snippet']
            
            emails.append(email_data)
        
        return emails, None
        
    except Exception as e:
        error_msg = str(e)
        if 'invalid_grant' in error_msg or 'Token has been expired' in error_msg:
            if os.path.exists(TOKEN_FILE):
                os.remove(TOKEN_FILE)
            gmail_service = None
            return None, "Gmail 인증이 만료되었습니다. 서버를 재시작해주세요."
        return None, f"메일 가져오기 실패: {error_msg}"


def extract_email_body(payload):
    """이메일 본문 추출"""
    body = ""
    
    if 'body' in payload and payload['body'].get('data'):
        body = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8', errors='ignore')
    elif 'parts' in payload:
        for part in payload['parts']:
            mime_type = part.get('mimeType', '')
            if mime_type == 'text/plain':
                if 'data' in part.get('body', {}):
                    body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8', errors='ignore')
                    break
            elif mime_type == 'text/html' and not body:
                if 'data' in part.get('body', {}):
                    html = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8', errors='ignore')
                    body = re.sub(r'<[^>]+>', '', html)
            elif 'parts' in part:
                body = extract_email_body(part)
                if body:
                    break
    
    body = re.sub(r'\s+', ' ', body).strip()
    return body


def summarize_emails_with_openai(emails):
    """OpenAI를 사용하여 이메일 요약"""
    if not emails:
        return "요약할 메일이 없습니다."
    
    if not OPENAI_API_KEY:
        return "⚠️ OpenAI API 키가 설정되지 않았습니다.\n\n.env 파일에 OPENAI_API_KEY를 설정해주세요."
    
    email_text = ""
    for i, email in enumerate(emails, 1):
        email_text += f"""
=== 메일 {i} ===
보낸 사람: {email['from']}
제목: {email['subject']}
날짜: {email['date']}
내용 미리보기: {email['body'][:300]}...
"""
    
    system_prompt = """당신은 이메일을 요약해주는 친절한 비서입니다.
사용자의 최근 이메일들을 분석하고 핵심 내용을 간결하게 요약해주세요.

요약 규칙:
1. 각 메일의 핵심 내용을 1-2문장으로 요약
2. 긴급하거나 중요한 메일은 먼저 언급
3. 스팸이나 광고성 메일은 간단히 "광고 메일"로 표시
4. 한국어로 친절하고 자연스럽게 설명
5. 음성으로 읽어줄 것이므로 자연스럽게 말하듯이 작성"""

    user_prompt = f"""다음은 사용자의 최근 이메일 {len(emails)}개입니다. 요약해주세요.

{email_text}

위 메일들의 핵심 내용을 요약해주세요."""

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
                "max_tokens": 2048,
                "temperature": 0.7
            },
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            return data["choices"][0]["message"]["content"]
        else:
            error_msg = response.json().get("error", {}).get("message", "알 수 없는 오류")
            return f"⚠️ 요약 실패: {error_msg}"
            
    except Exception as e:
        return f"⚠️ 오류: {str(e)}"


def is_email_summary_request(question):
    """질문이 메일 요약 요청인지 확인"""
    email_keywords = ['메일', '이메일', 'email', 'mail', '편지함', '받은편지', '메일함', '인박스', 'inbox']
    summary_keywords = ['요약', '알려', '읽어', '확인', '체크', '보여', '뭐가 왔', '뭐 왔', '새로운', '최근', '확인해', '있어', '왔어']
    
    question_lower = question.lower()
    has_email = any(kw in question_lower for kw in email_keywords)
    has_summary = any(kw in question_lower for kw in summary_keywords)
    
    return has_email and has_summary


# ===== RAG 텍스트 처리 =====
def tokenize(text):
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
                "id": doc_id,
                "text": doc['text'],
                "similarity": round(1 - distance, 4),
                "metadata": doc.get('metadata', {})
            })
    return results


# ===== OpenAI API (웹 검색 포함) =====
def ask_openai(question, context_docs, use_web_search=False):
    """OpenAI API 호출 (웹 검색 결과 포함)"""
    if not OPENAI_API_KEY:
        return "⚠️ OpenAI API 키가 설정되지 않았습니다.\n\n.env 파일에 OPENAI_API_KEY를 설정해주세요.", []
    
    web_results = []
    web_context = ""
    
    # 웹 검색 수행
    if use_web_search and WEBSEARCH_AVAILABLE:
        if is_news_request(question):
            results, error = web_search_news(question)
        else:
            results, error = web_search(question)
        
        if results:
            web_results = results
            web_context = "\n\n=== 웹 검색 결과 ===\n"
            for i, r in enumerate(results, 1):
                date_str = f" ({r.get('date', '')})" if r.get('date') else ""
                web_context += f"[검색 {i}] {r['title']}{date_str}\n{r['snippet']}\n\n"
    
    # 시스템 프롬프트 구성
    if context_docs or web_context:
        context = ""
        if context_docs:
            context += "=== RAG 문서 ===\n"
            context += "\n\n".join([
                f"[문서 {i+1}] (유사도: {doc['similarity']*100:.1f}%)\n{doc['text']}"
                for i, doc in enumerate(context_docs)
            ])
        
        context += web_context
        
        system_prompt = """당신은 RAG 기반 AI 어시스턴트입니다.
사용자의 질문에 대해 제공된 문서와 웹 검색 결과를 참고하여 답변하세요.
답변은 친절하고 자연스럽게 한국어로 해주세요.
웹 검색 결과를 인용할 때는 출처를 간략히 언급해주세요."""
        user_prompt = f"{context}\n\n=== 질문 ===\n{question}\n\n위 정보를 참고하여 답변해주세요."
    else:
        system_prompt = "당신은 친절한 AI 어시스턴트입니다. 한국어로 답변해주세요."
        user_prompt = f"질문: {question}"

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
                "max_tokens": 2048,
                "temperature": 0.7
            },
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            return data["choices"][0]["message"]["content"], web_results
        elif response.status_code == 401:
            return "⚠️ OpenAI API 키가 유효하지 않습니다.", []
        elif response.status_code == 429:
            return "⚠️ API 호출 한도 초과. 잠시 후 다시 시도해주세요.", []
        else:
            error_msg = response.json().get("error", {}).get("message", "알 수 없는 오류")
            return f"⚠️ API 오류: {error_msg}", []
            
    except requests.exceptions.Timeout:
        return "⚠️ API 응답 시간 초과. 다시 시도해주세요.", []
    except Exception as e:
        return f"⚠️ 오류 발생: {str(e)}", []


# ===== HTML 템플릿 (Voice + Gmail UI) =====
MOBILE_APP_HTML = '''
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="mobile-web-app-capable" content="yes">
    <meta name="theme-color" content="#10a37f">
    <title>🎤 Voice RAG + OpenAI + 📧메일</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; -webkit-tap-highlight-color: transparent; }
        body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); min-height: 100vh; color: white; }
        .app { display: flex; flex-direction: column; height: 100vh; max-width: 100%; margin: 0 auto; }
        
        .header { background: linear-gradient(135deg, #10a37f, #1a7f5a); padding: 12px; text-align: center; }
        .header h1 { font-size: 1.1rem; margin-bottom: 4px; }
        .header .status { font-size: 0.7rem; opacity: 0.9; }
        .status-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #ff4757; margin-right: 4px; }
        .status-dot.ok { background: #2ed573; }
        
        .feature-badges { display: flex; gap: 5px; flex-wrap: wrap; margin-top: 6px; justify-content: center; }
        .badge { padding: 2px 8px; border-radius: 10px; font-size: 0.6rem; font-weight: bold; }
        .badge-rag { background: rgba(16,163,127,0.3); color: #10a37f; }
        .badge-openai { background: rgba(16,163,127,0.3); color: #10a37f; }
        .badge-voice { background: rgba(80,250,123,0.3); color: #50fa7b; }
        .badge-mail { background: rgba(255,184,108,0.3); color: #ffb86c; }
        
        .tabs { display: flex; background: #1a1a2e; border-bottom: 1px solid #333; }
        .tab { flex: 1; padding: 10px; text-align: center; background: transparent; border: none; color: #888; font-size: 0.8rem; cursor: pointer; }
        .tab.active { color: #10a37f; border-bottom: 2px solid #10a37f; }
        
        .chat-container { flex: 1; overflow-y: auto; padding: 12px; display: flex; flex-direction: column; gap: 10px; }
        .message { max-width: 88%; padding: 10px 14px; border-radius: 16px; line-height: 1.5; font-size: 0.9rem; animation: fadeIn 0.3s; word-wrap: break-word; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        .message.user { background: linear-gradient(135deg, #10a37f, #1a7f5a); align-self: flex-end; border-bottom-right-radius: 4px; }
        .message.bot { background: #2a2a4a; align-self: flex-start; border-bottom-left-radius: 4px; border: 1px solid #333; }
        .message .sources { font-size: 0.7rem; color: #888; margin-top: 6px; padding-top: 6px; border-top: 1px solid #444; }
        .message .web-sources { font-size: 0.7rem; color: #8be9fd; margin-top: 4px; }
        .message .web-sources a { color: #8be9fd; text-decoration: none; }
        .message .mail-info { font-size: 0.7rem; color: #ffb86c; margin-top: 4px; padding: 4px 8px; background: rgba(255,184,108,0.1); border-radius: 4px; }
        
        .toggle-switch { position: relative; width: 36px; height: 20px; display: inline-block; vertical-align: middle; }
        .toggle-switch input { opacity: 0; width: 0; height: 0; }
        .toggle-slider { position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0; background-color: #333; transition: 0.3s; border-radius: 20px; }
        .toggle-slider:before { position: absolute; content: ""; height: 14px; width: 14px; left: 3px; bottom: 3px; background-color: white; transition: 0.3s; border-radius: 50%; }
        input:checked + .toggle-slider { background: linear-gradient(135deg, #10a37f, #1a7f5a); }
        input:checked + .toggle-slider:before { transform: translateX(16px); }
        
        .message-actions { display: flex; gap: 6px; margin-top: 6px; }
        .message-actions button { background: rgba(16,163,127,0.2); border: 1px solid #10a37f; color: #10a37f; padding: 3px 8px; border-radius: 10px; font-size: 0.65rem; cursor: pointer; }
        
        .typing { display: flex; gap: 4px; padding: 12px; }
        .typing span { width: 8px; height: 8px; background: #10a37f; border-radius: 50%; animation: bounce 1.4s infinite; }
        .typing span:nth-child(1) { animation-delay: 0s; }
        .typing span:nth-child(2) { animation-delay: 0.2s; }
        .typing span:nth-child(3) { animation-delay: 0.4s; }
        @keyframes bounce { 0%, 80%, 100% { transform: scale(0); } 40% { transform: scale(1); } }
        
        .input-container { background: #1a1a2e; padding: 10px; border-top: 1px solid #333; }
        
        .quick-actions { display: flex; gap: 6px; margin-bottom: 8px; flex-wrap: wrap; }
        .quick-btn { padding: 5px 10px; background: rgba(255,184,108,0.15); border: 1px solid #ffb86c; color: #ffb86c; border-radius: 12px; font-size: 0.7rem; cursor: pointer; }
        .quick-btn:hover { background: rgba(255,184,108,0.25); }
        
        .search-options { display: flex; gap: 8px; margin-bottom: 8px; align-items: center; flex-wrap: wrap; }
        .search-options label { font-size: 0.7rem; color: #888; }
        .search-options select { padding: 4px 8px; background: #2a2a4a; border: 1px solid #333; border-radius: 6px; color: white; font-size: 0.7rem; }
        
        .input-row { display: flex; gap: 6px; align-items: center; }
        
        .voice-btn { width: 44px; height: 44px; border-radius: 50%; border: none; background: linear-gradient(135deg, #10a37f, #1a7f5a); color: white; font-size: 1.1rem; cursor: pointer; flex-shrink: 0; transition: transform 0.1s; }
        .voice-btn:active { transform: scale(0.95); }
        .voice-btn.recording { background: linear-gradient(135deg, #ff4757, #ff6b81); animation: pulse 1s infinite; }
        @keyframes pulse { 0%, 100% { box-shadow: 0 0 0 0 rgba(255,71,87,0.4); } 50% { box-shadow: 0 0 0 10px rgba(255,71,87,0); } }
        
        .mail-btn { width: 44px; height: 44px; border-radius: 50%; border: none; background: linear-gradient(135deg, #ffb86c, #ff9f43); color: white; font-size: 1.1rem; cursor: pointer; flex-shrink: 0; }
        .mail-btn:active { transform: scale(0.95); }
        
        .text-input { flex: 1; padding: 10px 14px; background: #2a2a4a; border: 2px solid #333; border-radius: 20px; color: white; font-size: 0.9rem; }
        .text-input:focus { outline: none; border-color: #10a37f; }
        
        .send-btn { padding: 10px 14px; background: linear-gradient(135deg, #10a37f, #1a7f5a); border: none; border-radius: 20px; color: white; font-weight: bold; font-size: 0.8rem; cursor: pointer; }
        .send-btn:disabled { opacity: 0.5; }
        
        .tab-content { flex: 1; overflow-y: auto; padding: 12px; display: none; }
        .tab-content.active { display: block; }
        
        .doc-input { width: 100%; padding: 10px; background: #2a2a4a; border: 1px solid #333; border-radius: 8px; color: white; margin-bottom: 8px; font-size: 0.85rem; }
        textarea.doc-input { min-height: 80px; resize: vertical; }
        
        .doc-buttons { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 10px; }
        .doc-buttons button { padding: 8px 14px; border: none; border-radius: 8px; font-weight: bold; cursor: pointer; font-size: 0.75rem; }
        .btn-add { background: linear-gradient(135deg, #10a37f, #1a7f5a); color: white; }
        .btn-refresh { background: #333; color: white; }
        .btn-clear { background: #ff4757; color: white; }
        
        .doc-item { background: #2a2a4a; padding: 10px; border-radius: 8px; margin-bottom: 8px; border-left: 3px solid #10a37f; }
        .doc-item-id { color: #10a37f; font-size: 0.7rem; font-weight: bold; }
        .doc-item-text { color: #ccc; font-size: 0.8rem; margin-top: 4px; }
        .doc-item button { margin-top: 6px; padding: 4px 10px; background: #ff4757; border: none; border-radius: 4px; color: white; font-size: 0.65rem; cursor: pointer; }
        
        .empty-state { text-align: center; color: #666; padding: 30px 20px; }
        .empty-state .icon { font-size: 2.5rem; margin-bottom: 10px; }
        
        .setting-item { background: #2a2a4a; padding: 12px; border-radius: 8px; margin-bottom: 8px; }
        .setting-item label { display: block; color: #888; font-size: 0.75rem; margin-bottom: 6px; }
        .setting-item select { width: 100%; padding: 8px; background: #1a1a2e; border: 1px solid #333; border-radius: 6px; color: white; font-size: 0.8rem; }
        
        .model-info { background: #2a2a4a; padding: 12px; border-radius: 8px; margin-bottom: 8px; border-left: 3px solid #10a37f; }
        .model-info h3 { color: #10a37f; font-size: 0.8rem; margin-bottom: 6px; }
        .model-info p { color: #888; font-size: 0.7rem; line-height: 1.4; }
    </style>
</head>
<body>
    <div class="app">
        <div class="header">
            <h1>🎤 Voice RAG + OpenAI + 📧메일</h1>
            <div class="status">
                <span class="status-dot" id="statusDot"></span>
                <span id="statusText">연결 확인 중...</span>
                <span> | 📚 <span id="docCount">0</span>개</span>
                <span> | 📧 <span id="gmailStatus">-</span></span>
            </div>
            <div class="feature-badges">
                <span class="badge badge-rag">📚 RAG</span>
                <span class="badge badge-openai">🤖 GPT</span>
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
                안녕하세요! OpenAI GPT 기반 AI 어시스턴트입니다. 🤖<br><br>
                🎤 <b>음성 버튼</b>: 말로 질문하기<br>
                📧 <b>메일 버튼</b>: 메일 요약 듣기<br>
                📚 <b>RAG</b>: 저장된 문서 검색<br><br>
                💡 <b>"메일 요약해줘"</b>라고 말해보세요!
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
            <div class="model-info">
                <h3>🧠 현재 모델</h3>
                <p id="modelName">GPT-4o-mini</p>
                <p>OpenAI GPT + Gmail 메일 요약 지원</p>
            </div>
            <div class="setting-item">
                <label>📄 RAG 검색 결과 수</label>
                <select id="numResultsSetting">
                    <option value="0">사용안함</option>
                    <option value="3" selected>3개</option>
                    <option value="5">5개</option>
                </select>
            </div>
            <div class="setting-item">
                <label>📧 메일 요약 개수</label>
                <select id="mailCountSetting">
                    <option value="3">3개</option>
                    <option value="5" selected>5개</option>
                    <option value="10">10개</option>
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
            <div class="quick-actions">
                <button class="quick-btn" onclick="quickMail()">📧 메일 요약해줘</button>
                <button class="quick-btn" onclick="quickNews()">📰 오늘 뉴스</button>
                <button class="quick-btn" onclick="quickWeather()">🌤️ 날씨</button>
            </div>
            <div class="search-options">
                <label>📄 RAG:</label>
                <select id="numResults">
                    <option value="0">OFF</option>
                    <option value="3" selected>3개</option>
                </select>
                <label style="margin-left: 8px;">🌐 웹:</label>
                <label class="toggle-switch">
                    <input type="checkbox" id="webSearchToggle" checked>
                    <span class="toggle-slider"></span>
                </label>
            </div>
            <div class="input-row">
                <button class="voice-btn" id="voiceBtn" onclick="toggleVoice()">🎤</button>
                <button class="mail-btn" id="mailBtn" onclick="quickMail()">📧</button>
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
            }
        }
        
        async function checkHealth() {
            try {
                const res = await fetch('/health');
                const data = await res.json();
                document.getElementById('statusDot').classList.toggle('ok', data.llm_available);
                document.getElementById('statusText').textContent = data.llm_available ? 'OpenAI 연결' : 'API 키 필요';
                document.getElementById('docCount').textContent = data.documents || 0;
                document.getElementById('gmailStatus').textContent = data.gmail_available ? '연결됨' : '미설정';
                document.getElementById('modelName').textContent = data.model || 'GPT';
            } catch(e) {
                document.getElementById('statusDot').classList.remove('ok');
                document.getElementById('statusText').textContent = '서버 오류';
            }
        }
        
        // ===== 음성 인식 (STT) =====
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
                
                recognition.onerror = () => {
                    isRecording = false;
                    document.getElementById('voiceBtn').classList.remove('recording');
                };
            }
        }
        
        function toggleVoice() {
            if (!recognition) { alert('음성 인식을 지원하지 않습니다. Chrome을 사용하세요.'); return; }
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
            const cleanText = text.replace(/\\*\\*(.+?)\\*\\*/g, '$1').replace(/\\*(.+?)\\*/g, '$1').replace(/`(.+?)`/g, '$1').replace(/#{1,6}\\s/g, '').replace(/\\n/g, ' ');
            const u = new SpeechSynthesisUtterance(cleanText);
            u.lang = 'ko-KR';
            u.rate = parseFloat(document.getElementById('speechRate').value);
            speechSynthesis.speak(u);
        }
        
        function stopSpeak() { speechSynthesis.cancel(); }
        
        function addMsg(text, isUser, sources = [], webSources = [], mailInfo = null) {
            const c = document.getElementById('chatContainer');
            const d = document.createElement('div');
            d.className = 'message ' + (isUser ? 'user' : 'bot');
            let h = text.replace(/\\n/g, '<br>').replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>').replace(/\\*(.+?)\\*/g, '<em>$1</em>');
            if (!isUser) {
                if (sources && sources.length > 0) {
                    h += '<div class="sources">📚 참고: ' + sources.map(s => s.id).join(', ') + '</div>';
                }
                if (webSources && webSources.length > 0) {
                    h += '<div class="web-sources">🌐 ' + webSources.slice(0,3).map(s => '<a href="'+s.url+'" target="_blank">'+(s.title||'링크').substring(0,30)+'</a>').join(', ') + '</div>';
                }
                if (mailInfo) {
                    h += '<div class="mail-info">📧 ' + mailInfo + '</div>';
                }
                const safeText = text.replace(/'/g, "\\\\'").replace(/"/g, '\\\\"').replace(/\\n/g, ' ');
                h += '<div class="message-actions"><button onclick="speak(\\''+safeText+'\\')">🔊</button><button onclick="stopSpeak()">⏹️</button></div>';
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
        
        // 빠른 액션
        function quickMail() {
            document.getElementById('userInput').value = '메일 요약해줘';
            sendMessage();
        }
        
        function quickNews() {
            document.getElementById('userInput').value = '오늘 주요 뉴스 알려줘';
            sendMessage();
        }
        
        function quickWeather() {
            document.getElementById('userInput').value = '서울 날씨 알려줘';
            sendMessage();
        }
        
        async function sendMessage() {
            const input = document.getElementById('userInput');
            const q = input.value.trim();
            if (!q || isProcessing) return;
            
            isProcessing = true;
            document.getElementById('sendBtn').disabled = true;
            document.getElementById('voiceBtn').disabled = true;
            document.getElementById('mailBtn').disabled = true;
            
            addMsg(q, true);
            input.value = '';
            showTyping();
            
            try {
                const webSearchEnabled = document.getElementById('webSearchToggle').checked;
                const numResults = parseInt(document.getElementById('numResults').value);
                const mailCount = parseInt(document.getElementById('mailCountSetting')?.value || 5);
                
                const res = await fetch('/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        question: q, 
                        n_results: numResults,
                        use_web_search: webSearchEnabled,
                        mail_count: mailCount
                    })
                });
                
                hideTyping();
                const data = await res.json();
                
                const mailInfo = data.mail_used ? data.mail_count + '개 메일 분석됨' : null;
                addMsg(data.answer, false, data.sources, data.web_sources || [], mailInfo);
                speak(data.answer);
                checkHealth();
            } catch(e) {
                hideTyping();
                addMsg('⚠️ 오류: ' + e.message, false);
            } finally {
                isProcessing = false;
                document.getElementById('sendBtn').disabled = false;
                document.getElementById('voiceBtn').disabled = false;
                document.getElementById('mailBtn').disabled = false;
            }
        }
        
        async function loadDocs() {
            try {
                const res = await fetch('/list?limit=50');
                const data = await res.json();
                document.getElementById('docCount').textContent = data.total;
                const list = document.getElementById('docList');
                if (data.documents && data.documents.length) {
                    list.innerHTML = data.documents.map(d => '<div class="doc-item"><div class="doc-item-id">🏷️ '+d.id+'</div><div class="doc-item-text">'+d.text+'</div><button onclick="delDoc(\\''+d.id+'\\')">🗑️</button></div>').join('');
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
                await fetch('/add', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({text, id}) });
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
    """채팅 API (메일 요약 + 웹 검색 지원)"""
    data = request.json
    question = data.get('question', '')
    n_results = data.get('n_results', 3)
    use_web_search = data.get('use_web_search', True)
    mail_count = data.get('mail_count', 5)
    
    if not question:
        return jsonify({"error": "질문을 입력해주세요"}), 400
    
    # 메일 요약 요청인지 확인
    if is_email_summary_request(question):
        emails, error = get_recent_emails(max_results=mail_count)
        
        if error:
            return jsonify({
                "question": question,
                "answer": f"📧 메일 확인 중 문제가 발생했습니다.\n\n{error}\n\n💡 첫 실행 시 터미널에서 Google 인증이 필요합니다.\n서버를 재시작하고 터미널의 안내를 따라주세요.",
                "sources": [],
                "web_sources": [],
                "mail_used": False
            })
        
        if emails:
            summary = summarize_emails_with_openai(emails)
            return jsonify({
                "question": question,
                "answer": summary,
                "sources": [],
                "web_sources": [],
                "mail_used": True,
                "mail_count": len(emails)
            })
    
    # 일반 질문 처리
    sources = []
    if n_results > 0:
        sources = rag_search(question, n=n_results)
    
    # 웹 검색 자동 판단
    should_use_web = use_web_search and needs_web_search(question)
    
    answer, web_sources = ask_openai(question, sources, use_web_search=should_use_web)
    
    return jsonify({
        "question": question,
        "answer": answer,
        "sources": sources,
        "web_sources": web_sources,
        "mail_used": False
    })


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


@app.route('/health')
def health():
    gmail_ok = gmail_service is not None
    return jsonify({
        "status": "running",
        "documents": len(documents),
        "llm_available": bool(OPENAI_API_KEY),
        "model": OPENAI_MODEL,
        "llm_type": "openai",
        "gmail_available": gmail_ok,
        "web_search_available": WEBSEARCH_AVAILABLE,
        "voice_available": True
    })


# ===== 시작 =====
load_data()
ensure_credentials_file()

if GMAIL_AVAILABLE:
    init_gmail_service()

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🎤 Voice RAG + OpenAI + 📧메일요약 + 🌐웹검색")
    print("="*60)
    print(f"🌐 웹 UI: http://localhost:5001")
    print(f"📚 저장된 문서 수: {len(documents)}")
    print(f"🧠 모델: {OPENAI_MODEL}")
    
    print("\n📌 주요 기능:")
    print("   🎤 음성 인식/출력 (STT/TTS)")
    print("   📚 RAG 문서 검색")
    print("   🌐 웹검색 (DuckDuckGo)")
    print("   📧 Gmail 메일 요약 (hskimh1982@gmail.com)")
    
    if OPENAI_API_KEY:
        print("\n✅ OpenAI API 키 설정됨")
    else:
        print("\n⚠️  OpenAI API 키 없음")
        print("   .env 파일에 OPENAI_API_KEY=sk-xxxxx 설정하세요")
    
    if WEBSEARCH_AVAILABLE:
        print("✅ 웹 검색 라이브러리 설치됨")
    else:
        print("⚠️  웹 검색 라이브러리 미설치")
        print("   pip install duckduckgo_search")
    
    if GMAIL_AVAILABLE:
        if gmail_service:
            print("✅ Gmail API 연결됨")
        else:
            print("⚠️  Gmail 인증 필요 - 터미널의 URL을 브라우저에서 열어주세요")
    else:
        print("⚠️  Gmail 라이브러리 미설치")
        print("   pip install google-auth-oauthlib google-auth-httplib2 google-api-python-client")
    
    print("\n" + "="*60)
    print("💡 사용법:")
    print("   • '메일 요약해줘' - Gmail 메일 요약")
    print("   • '오늘 뉴스 알려줘' - 웹 검색으로 최신 뉴스")
    print("   • '서울 날씨' - 웹 검색으로 날씨 정보")
    print("="*60)
    print("\n🚀 서버 시작! http://localhost:5001\n")
    
    app.run(host='0.0.0.0', port=5001, debug=False)
