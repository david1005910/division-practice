#!/usr/bin/env python3
"""
초등학생을 위한 나눗셈/곱셈 연습 프로그램 - 웹 버전
Flask를 사용한 웹 애플리케이션
"""

from flask import Flask, render_template, jsonify, request, session
import random
import os
from datetime import datetime

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'division-practice-secret-key-2024')

# 과일 이모지 목록
FRUITS = ['🍎', '🍊', '🍋', '🍇', '🍓', '🍑', '🍒', '🥝', '🍌', '🫐']

def generate_problems(count, number, mode):
    """문제 생성 (나눗셈/곱셈)"""
    problems = []
    all_problems = []

    for i in range(1, 11):
        if mode == "division":
            dividend = number * i
            all_problems.append({
                'num1': dividend,
                'num2': number,
                'answer': i
            })
        else:  # multiplication
            product = number * i
            all_problems.append({
                'num1': number,
                'num2': i,
                'answer': product
            })

    random.shuffle(all_problems)

    # 요청된 수만큼 선택
    for problem in all_problems:
        if len(problems) >= count:
            break
        problems.append(problem)

    # 10개 초과시 반복
    while len(problems) < count:
        random.shuffle(all_problems)
        for problem in all_problems:
            if len(problems) >= count:
                break
            problems.append(problem.copy())

    return problems

@app.route('/')
def index():
    """메인 페이지"""
    return render_template('index.html')

@app.route('/api/start', methods=['POST'])
def start_quiz():
    """퀴즈 시작"""
    data = request.json
    mode = data.get('mode', 'division')
    number = int(data.get('number', 2))
    count = int(data.get('count', 10))

    problems = generate_problems(count, number, mode)

    session['problems'] = problems
    session['current_index'] = 0
    session['correct_count'] = 0
    session['wrong_problems'] = []
    session['mode'] = mode
    session['number'] = number

    return jsonify({
        'success': True,
        'total': len(problems),
        'problem': problems[0],
        'index': 0,
        'mode': mode,
        'number': number
    })

@app.route('/api/answer', methods=['POST'])
def check_answer():
    """정답 확인"""
    data = request.json
    user_answer = int(data.get('answer', 0))

    problems = session.get('problems', [])
    current_index = session.get('current_index', 0)
    correct_count = session.get('correct_count', 0)
    wrong_problems = session.get('wrong_problems', [])
    mode = session.get('mode', 'division')

    if current_index >= len(problems):
        return jsonify({'error': 'No more problems'}), 400

    problem = problems[current_index]
    correct = user_answer == problem['answer']

    if correct:
        correct_count += 1
    else:
        wrong_problems.append(problem)

    session['correct_count'] = correct_count
    session['wrong_problems'] = wrong_problems

    return jsonify({
        'correct': correct,
        'answer': problem['answer'],
        'user_answer': user_answer,
        'correct_count': correct_count,
        'current_index': current_index,
        'total': len(problems)
    })

@app.route('/api/next', methods=['POST'])
def next_problem():
    """다음 문제"""
    problems = session.get('problems', [])
    current_index = session.get('current_index', 0) + 1
    session['current_index'] = current_index

    if current_index >= len(problems):
        # 퀴즈 종료
        correct_count = session.get('correct_count', 0)
        wrong_problems = session.get('wrong_problems', [])
        percentage = (correct_count / len(problems)) * 100 if problems else 0

        return jsonify({
            'finished': True,
            'correct_count': correct_count,
            'total': len(problems),
            'percentage': round(percentage, 1),
            'wrong_problems': wrong_problems
        })

    return jsonify({
        'finished': False,
        'problem': problems[current_index],
        'index': current_index,
        'total': len(problems),
        'correct_count': session.get('correct_count', 0)
    })

@app.route('/api/retry', methods=['POST'])
def retry_wrong():
    """틀린 문제 다시 풀기"""
    wrong_problems = session.get('wrong_problems', [])

    if not wrong_problems:
        return jsonify({'error': 'No wrong problems'}), 400

    session['problems'] = wrong_problems.copy()
    session['current_index'] = 0
    session['correct_count'] = 0
    session['wrong_problems'] = []

    return jsonify({
        'success': True,
        'total': len(wrong_problems),
        'problem': wrong_problems[0],
        'index': 0
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
