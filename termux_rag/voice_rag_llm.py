#!/usr/bin/env python3
"""
Voice RAG + LLM 통합 시스템 (OpenAI API 버전)
음성 질문 → RAG 검색 → GPT 답변 → 음성 출력

사용법:
1. 먼저 RAG 서버 실행: python voice_rag.py
2. 이 서버 실행: python voice_rag_llm.py
3. 브라우저에서: http://localhost:5001
"""

from flask import Flask, request, jsonify, render_template_string
import requests
import os

app = Flask(__name__)

# ===== 설정 =====
# OpenAI API 키
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "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")

# OpenAI 모델 선택
# - gpt-4o: 가장 똑똑함 (비쌈)
# - gpt-4o-mini: 빠르고 저렴함 (추천)
# - gpt-3.5-turbo: 가장 저렴함
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

# RAG 서버 주소
RAG_SERVER = os.environ.get("RAG_SERVER", "http://localhost:5000")


def rag_search(query, n=3):
    """RAG 서버에서 관련 문서 검색"""
    try:
        response = requests.post(
            f"{RAG_SERVER}/search",
            json={"query": query, "n": n},
            timeout=30
        )
        if response.status_code == 200:
            return response.json().get("results", [])
    except Exception as e:
        print(f"RAG 검색 오류: {e}")
    return []


def build_prompt(question, context_docs):
    """LLM에게 보낼 프롬프트 생성"""
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
        system_prompt = """당신은 RAG 기반 AI 어시스턴트입니다.
한국어로 친절하게 답변해주세요."""
        
        user_prompt = f"""질문: {question}

관련 문서를 찾지 못했습니다. 
저장된 문서가 없거나 관련 내용이 없다고 안내해주세요."""

    return system_prompt, user_prompt


def ask_openai(question, context_docs):
    """OpenAI API 호출"""
    if not OPENAI_API_KEY:
        return "⚠️ OpenAI API 키가 설정되지 않았습니다.\n\n설정 방법:\nexport OPENAI_API_KEY='sk-your-key-here'"
    
    system_prompt, user_prompt = build_prompt(question, context_docs)
    
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
            data = response.json()
            return data["choices"][0]["message"]["content"]
        elif response.status_code == 401:
            return "⚠️ OpenAI API 키가 유효하지 않습니다. 키를 확인해주세요."
        elif response.status_code == 429:
            return "⚠️ API 호출 한도를 초과했습니다. 잠시 후 다시 시도해주세요."
        else:
            error_msg = response.json().get("error", {}).get("message", "알 수 없는 오류")
            return f"⚠️ OpenAI API 오류: {error_msg}"
            
    except requests.exceptions.Timeout:
        return "⚠️ 요청 시간이 초과되었습니다. 다시 시도해주세요."
    except Exception as e:
        return f"⚠️ 오류 발생: {str(e)}"


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
            gap: 8px;
            align-items: center;
        }
        .voice-btn {
            width: 55px;
            height: 55px;
            border-radius: 50%;
            border: none;
            background: linear-gradient(135deg, #10a37f, #1a7f5a);
            color: white;
            font-size: 1.4rem;
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
        .input-wrapper {
            flex: 1;
            display: flex;
            gap: 8px;
        }
        input[type="text"] {
            flex: 1;
            padding: 14px 16px;
            border: 2px solid #333;
            border-radius: 25px;
            background: #1a1a2e;
            color: white;
            font-size: 1rem;
        }
        input:focus {
            outline: none;
            border-color: #10a37f;
        }
        .send-btn {
            padding: 14px 20px;
            border: none;
            border-radius: 25px;
            background: linear-gradient(135deg, #10a37f, #1a7f5a);
            color: white;
            font-weight: bold;
            cursor: pointer;
            font-size: 0.9rem;
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
        .quick-actions {
            display: flex;
            gap: 8px;
            margin-bottom: 10px;
            flex-wrap: wrap;
        }
        .quick-btn {
            padding: 6px 12px;
            border: 1px solid #444;
            border-radius: 15px;
            background: #1a1a2e;
            color: #888;
            font-size: 0.75rem;
            cursor: pointer;
        }
        .quick-btn:hover {
            border-color: #10a37f;
            color: #10a37f;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Voice RAG + GPT</h1>
        <p class="subtitle">음성으로 질문하면 AI가 문서를 검색해서 답변해요</p>
        
        <div class="status-bar">
            <div class="status-item">
                <div class="status-dot" id="ragDot"></div>
                <span>RAG</span>
            </div>
            <div class="status-item">
                <div class="status-dot" id="llmDot"></div>
                <span id="llmName">GPT</span>
            </div>
            <div class="status-item">
                📚 <span id="docCount">0</span>개 문서
            </div>
        </div>
        
        <div class="chat-box" id="chatBox">
            <div class="message bot-msg">
                안녕하세요! 저는 RAG 기반 GPT 어시스턴트예요. 🤖<br><br>
                저장된 문서에서 정보를 찾아 답변해드릴게요.<br>
                🎤 버튼을 누르거나 텍스트로 질문하세요!
            </div>
        </div>
        
        <div class="quick-actions">
            <button class="quick-btn" onclick="askQuestion('저장된 문서 목록 보여줘')">📋 문서 목록</button>
            <button class="quick-btn" onclick="askQuestion('무엇을 알고 있어?')">❓ 뭘 알아?</button>
            <button class="quick-btn" onclick="window.open('/docs', '_blank')">📄 문서 관리</button>
        </div>
        
        <p class="status-text" id="status"></p>
        
        <div class="input-area">
            <button class="voice-btn" id="voiceBtn" onclick="toggleVoice()">🎤</button>
            <div class="input-wrapper">
                <input type="text" id="userInput" placeholder="질문을 입력하세요...">
                <button class="send-btn" id="sendBtn" onclick="sendMessage()">전송</button>
            </div>
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
        setInterval(checkHealth, 30000);
        
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
                console.error('음성 인식 오류:', event.error);
                let errorMsg = '음성 인식 오류';
                if (event.error === 'not-allowed') {
                    errorMsg = '마이크 권한을 허용해주세요';
                } else if (event.error === 'no-speech') {
                    errorMsg = '음성이 감지되지 않았어요';
                }
                document.getElementById('status').textContent = '❌ ' + errorMsg;
                isRecording = false;
                document.getElementById('voiceBtn').classList.remove('recording');
            };
        }
        
        async function checkHealth() {
            try {
                const res = await fetch('/health');
                const data = await res.json();
                
                document.getElementById('ragDot').classList.toggle('ok', data.rag_server);
                document.getElementById('llmDot').classList.toggle('ok', data.llm_available);
                document.getElementById('llmName').textContent = data.model || 'GPT';
                document.getElementById('docCount').textContent = data.rag_documents || 0;
            } catch (e) {
                console.error('상태 확인 실패:', e);
            }
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
            if ('speechSynthesis' in window) {
                speechSynthesis.cancel();
            }
        }
        
        function addMessage(text, isUser, sources = null) {
            const chatBox = document.getElementById('chatBox');
            const msgDiv = document.createElement('div');
            msgDiv.className = 'message ' + (isUser ? 'user-msg' : 'bot-msg');
            
            let html = text.replace(/\\n/g, '<br>');
            
            if (!isUser) {
                if (sources && sources.length > 0) {
                    html += '<div class="sources">📚 참고: ';
                    html += sources.map((s, i) => s.id).join(', ');
                    html += '</div>';
                }
                const safeText = text.replace(/`/g, "'").replace(/\\/g, "\\\\");
                html += '<div class="actions">';
                html += '<button class="action-btn" onclick="speak(`' + safeText + '`)">🔊 읽기</button>';
                html += '<button class="action-btn" onclick="stopSpeaking()">⏹️ 중지</button>';
                html += '<button class="action-btn" onclick="copyText(`' + safeText + '`)">📋 복사</button>';
                html += '</div>';
            }
            
            msgDiv.innerHTML = html;
            chatBox.appendChild(msgDiv);
            chatBox.scrollTop = chatBox.scrollHeight;
        }
        
        function copyText(text) {
            navigator.clipboard.writeText(text).then(() => {
                document.getElementById('status').textContent = '📋 복사되었습니다';
                setTimeout(() => {
                    document.getElementById('status').textContent = '';
                }, 2000);
            });
        }
        
        function askQuestion(question) {
            document.getElementById('userInput').value = question;
            sendMessage();
        }
        
        async function sendMessage() {
            const input = document.getElementById('userInput');
            const question = input.value.trim();
            
            if (!question || isProcessing) return;
            
            isProcessing = true;
            document.getElementById('sendBtn').disabled = true;
            document.getElementById('voiceBtn').disabled = true;
            
            addMessage(question, true);
            input.value = '';
            
            document.getElementById('status').innerHTML = '<span class="loading"></span>검색하고 답변 생성 중...';
            
            try {
                const numResults = document.getElementById('numResults').value;
                
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        question: question,
                        n_results: parseInt(numResults)
                    })
                });
                
                const data = await response.json();
                
                document.getElementById('status').textContent = '';
                addMessage(data.answer, false, data.sources);
                speak(data.answer);
                
            } catch (error) {
                document.getElementById('status').textContent = '';
                addMessage('죄송해요, 오류가 발생했어요: ' + error.message, false);
            } finally {
                isProcessing = false;
                document.getElementById('sendBtn').disabled = false;
                document.getElementById('voiceBtn').disabled = false;
            }
        }
        
        // Enter 키로 전송
        document.getElementById('userInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !isProcessing) sendMessage();
        });
    </script>
</body>
</html>
'''

# 문서 관리 페이지
DOCS_HTML = '''
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>📄 문서 관리</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, sans-serif;
            background: #1a1a2e;
            color: white;
            padding: 20px;
        }
        .container { max-width: 600px; margin: 0 auto; }
        h1 { color: #10a37f; margin-bottom: 20px; }
        .card {
            background: #0f0f23;
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            border: 1px solid #333;
        }
        h2 { color: #10a37f; margin-bottom: 15px; font-size: 1.1rem; }
        textarea, input {
            width: 100%;
            padding: 12px;
            border: 1px solid #333;
            border-radius: 10px;
            background: #1a1a2e;
            color: white;
            margin-bottom: 10px;
        }
        textarea { min-height: 100px; }
        button {
            padding: 12px 24px;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            font-weight: bold;
            margin-right: 10px;
            margin-bottom: 10px;
        }
        .btn-primary { background: #10a37f; color: white; }
        .btn-danger { background: #ff4757; color: white; }
        .btn-secondary { background: #333; color: white; }
        .doc-list { max-height: 300px; overflow-y: auto; }
        .doc-item {
            background: #1a1a2e;
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 8px;
            border-left: 3px solid #10a37f;
        }
        .doc-item .id { color: #10a37f; font-size: 0.85rem; }
        .doc-item .text { color: #ccc; margin-top: 5px; font-size: 0.9rem; }
        .doc-item .actions { margin-top: 8px; }
        .doc-item button { padding: 5px 10px; font-size: 0.8rem; }
        .status { color: #2ed573; margin: 10px 0; }
        a { color: #10a37f; }
    </style>
</head>
<body>
    <div class="container">
        <h1>📄 문서 관리</h1>
        <p style="margin-bottom:20px"><a href="/">← 채팅으로 돌아가기</a></p>
        
        <div class="card">
            <h2>➕ 문서 추가</h2>
            <input type="text" id="docId" placeholder="문서 ID (선택사항)">
            <textarea id="docText" placeholder="문서 내용을 입력하세요..."></textarea>
            <button class="btn-primary" onclick="addDoc()">추가</button>
            <p class="status" id="addStatus"></p>
        </div>
        
        <div class="card">
            <h2>📚 저장된 문서 (<span id="totalDocs">0</span>개)</h2>
            <button class="btn-secondary" onclick="loadDocs()">새로고침</button>
            <button class="btn-danger" onclick="clearAll()">전체 삭제</button>
            <div class="doc-list" id="docList"></div>
        </div>
    </div>
    
    <script>
        loadDocs();
        
        async function loadDocs() {
            const res = await fetch('http://localhost:5000/list?limit=100');
            const data = await res.json();
            document.getElementById('totalDocs').textContent = data.total;
            
            let html = '';
            data.documents.forEach(doc => {
                html += '<div class="doc-item">';
                html += '<div class="id">🏷️ ' + doc.id + '</div>';
                html += '<div class="text">' + doc.text + '</div>';
                html += '<div class="actions"><button class="btn-danger" onclick="deleteDoc(\\'' + doc.id + '\\')">삭제</button></div>';
                html += '</div>';
            });
            document.getElementById('docList').innerHTML = html || '<p style="color:#888">저장된 문서가 없습니다</p>';
        }
        
        async function addDoc() {
            const text = document.getElementById('docText').value.trim();
            if (!text) { alert('내용을 입력하세요'); return; }
            
            const id = document.getElementById('docId').value.trim() || undefined;
            const res = await fetch('http://localhost:5000/add', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({text, id})
            });
            const data = await res.json();
            document.getElementById('addStatus').textContent = '✅ 추가됨: ' + data.id;
            document.getElementById('docText').value = '';
            document.getElementById('docId').value = '';
            loadDocs();
        }
        
        async function deleteDoc(id) {
            if (!confirm('삭제할까요?')) return;
            await fetch('http://localhost:5000/delete', {
                method: 'DELETE',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({id})
            });
            loadDocs();
        }
        
        async function clearAll() {
            if (!confirm('모든 문서를 삭제할까요?')) return;
            await fetch('http://localhost:5000/clear', {method: 'DELETE'});
            loadDocs();
        }
    </script>
</body>
</html>
'''


@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)


@app.route('/docs')
def docs_page():
    return render_template_string(DOCS_HTML)


@app.route('/chat', methods=['POST'])
def chat():
    """메인 채팅 API"""
    data = request.json
    question = data.get('question', '')
    n_results = data.get('n_results', 3)
    
    if not question:
        return jsonify({"error": "질문을 입력해주세요"}), 400
    
    # 1. RAG 검색
    sources = rag_search(question, n=n_results)
    
    # 2. OpenAI 답변 생성
    answer = ask_openai(question, sources)
    
    return jsonify({
        "question": question,
        "answer": answer,
        "sources": sources
    })


@app.route('/health')
def health():
    """서버 상태 확인"""
    # RAG 서버 연결 확인
    rag_ok = False
    rag_docs = 0
    try:
        rag_res = requests.get(f"{RAG_SERVER}/", timeout=5)
        if rag_res.status_code == 200:
            rag_ok = True
            rag_docs = rag_res.json().get("documents", 0)
    except:
        pass
    
    # OpenAI API 키 확인
    llm_available = bool(OPENAI_API_KEY)
    
    return jsonify({
        "status": "running",
        "rag_server": rag_ok,
        "rag_documents": rag_docs,
        "llm_type": "openai",
        "model": OPENAI_MODEL,
        "llm_available": llm_available
    })


if __name__ == '__main__':
    print("\n" + "="*50)
    print("🤖 Voice RAG + GPT System (OpenAI)")
    print("="*50)
    print(f"💬 채팅 UI: http://localhost:5001")
    print(f"📄 문서 관리: http://localhost:5001/docs")
    print(f"🔍 RAG 서버: {RAG_SERVER}")
    print(f"🤖 모델: {OPENAI_MODEL}")
    print("="*50)
    
    if not OPENAI_API_KEY:
        print("\n⚠️  OpenAI API 키가 설정되지 않았습니다!")
        print("   설정 방법:")
        print("   export OPENAI_API_KEY='sk-...'")
    else:
        print(f"\n✅ OpenAI API 키 설정됨")
    
    print("\n📌 먼저 RAG 서버를 실행하세요:")
    print("   python voice_rag.py")
    print("="*50 + "\n")
    
    app.run(host='0.0.0.0', port=5001, debug=False)
