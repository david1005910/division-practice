#!/usr/bin/env python3
"""
초등학생을 위한 나눗셈 연습 프로그램 - GUI 버전
tkinter를 사용한 윈도우 대화형 모드
"""

import tkinter as tk
from tkinter import messagebox, font, simpledialog
import random
import subprocess
import platform
import json
import os
from datetime import datetime

# 점수 저장 파일 경로
SCORE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gui_scores.json")

def load_scores():
    """저장된 점수 기록을 불러옵니다."""
    if os.path.exists(SCORE_FILE):
        try:
            with open(SCORE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []
    return []

def save_score(name, correct, total, percentage, number, mode):
    """점수를 파일에 저장합니다."""
    scores = load_scores()

    score_entry = {
        "name": name,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "correct": correct,
        "total": total,
        "percentage": round(percentage, 1),
        "number": number,
        "mode": mode
    }

    scores.append(score_entry)

    # 최근 100개만 유지
    if len(scores) > 100:
        scores = scores[-100:]

    with open(SCORE_FILE, 'w', encoding='utf-8') as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)

    return score_entry

def play_sound(sound_type):
    """효과음 재생"""
    try:
        if platform.system() == "Darwin":  # macOS
            if sound_type == "correct":
                # 정답 효과음 (Glass)
                subprocess.Popen(["afplay", "/System/Library/Sounds/Glass.aiff"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            elif sound_type == "wrong":
                # 오답 효과음 (Basso)
                subprocess.Popen(["afplay", "/System/Library/Sounds/Basso.aiff"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            elif sound_type == "complete":
                # 완료 효과음 (Fanfare)
                subprocess.Popen(["afplay", "/System/Library/Sounds/Glass.aiff"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif platform.system() == "Windows":
            import winsound
            if sound_type == "correct":
                winsound.MessageBeep(winsound.MB_OK)
            elif sound_type == "wrong":
                winsound.MessageBeep(winsound.MB_ICONHAND)
            elif sound_type == "complete":
                winsound.MessageBeep(winsound.MB_ICONASTERISK)
    except Exception:
        pass  # 소리 재생 실패해도 계속 진행

# 과일 이모지 목록
FRUITS = ['🍎', '🍊', '🍋', '🍇', '🍓', '🍑', '🍒', '🥝', '🍌', '🫐']

class DivisionPracticeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🎈 초등학생 나눗셈 연습 🎈")
        self.root.geometry("600x700")
        self.root.configure(bg="#FFF8E7")

        # 문제 설정
        self.problems = []
        self.current_index = 0
        self.correct_count = 0
        self.current_fruit = '🍎'
        self.selected_number = 2  # 나누기 또는 곱하기 숫자
        self.problem_count = 10
        self.wrong_problems = []  # 틀린 문제 저장
        self.mode = "division"  # "division" 또는 "multiplication"

        # 폰트 설정
        self.title_font = font.Font(family="Arial", size=24, weight="bold")
        self.problem_font = font.Font(family="Arial", size=36, weight="bold")
        self.fruit_font = font.Font(family="Arial", size=20)
        self.button_font = font.Font(family="Arial", size=14)

        self.create_widgets()
        self.show_start_screen()

    def create_widgets(self):
        """위젯 생성"""
        # 메인 프레임
        self.main_frame = tk.Frame(self.root, bg="#FFF8E7")
        self.main_frame.pack(expand=True, fill="both", padx=20, pady=20)

        # 제목
        self.title_label = tk.Label(
            self.main_frame,
            text="🎈 나눗셈 연습 🎈",
            font=self.title_font,
            bg="#FFF8E7",
            fg="#FF6B6B"
        )
        self.title_label.pack(pady=10)

        # 진행 상황
        self.progress_label = tk.Label(
            self.main_frame,
            text="",
            font=self.button_font,
            bg="#FFF8E7",
            fg="#666666"
        )
        self.progress_label.pack(pady=5)

        # 문제 표시
        self.problem_label = tk.Label(
            self.main_frame,
            text="",
            font=self.problem_font,
            bg="#FFF8E7",
            fg="#333333"
        )
        self.problem_label.pack(pady=10)

        # 분수 표시 프레임
        self.fraction_frame = tk.Frame(self.main_frame, bg="#FFF8E7")
        self.fraction_frame.pack(pady=10)

        # 분수 설명
        self.fraction_title = tk.Label(
            self.fraction_frame,
            text="📐 분수로 보면:",
            font=self.button_font,
            bg="#FFF8E7",
            fg="#666666"
        )
        self.fraction_title.pack(pady=(0, 5))

        # 분수 내용 프레임
        self.fraction_content = tk.Frame(self.fraction_frame, bg="#FFFAF0", padx=20, pady=10)
        self.fraction_content.pack()

        # 분수 - 분자 (위)
        self.numerator_label = tk.Label(
            self.fraction_content,
            text="",
            font=self.problem_font,
            bg="#FFFAF0",
            fg="#FF6B6B"
        )
        self.numerator_label.pack()

        # 분수 - 분수선
        self.fraction_line = tk.Frame(
            self.fraction_content,
            bg="#333333",
            height=4,
            width=100
        )
        self.fraction_line.pack(pady=5)

        # 분수 - 분모 (아래)
        self.denominator_label = tk.Label(
            self.fraction_content,
            text="",
            font=self.problem_font,
            bg="#FFFAF0",
            fg="#4ECDC4"
        )
        self.denominator_label.pack()

        # 분수 설명 추가
        self.fraction_explain = tk.Label(
            self.fraction_frame,
            text="",
            font=self.button_font,
            bg="#FFF8E7",
            fg="#888888"
        )
        self.fraction_explain.pack(pady=(5, 0))

        # 과일 시각화
        self.fruit_frame = tk.Frame(self.main_frame, bg="#FFF8E7")
        self.fruit_frame.pack(pady=10)

        self.fruit_label = tk.Label(
            self.fruit_frame,
            text="",
            font=self.fruit_font,
            bg="#FFF8E7",
            wraplength=500
        )
        self.fruit_label.pack()

        # 설명
        self.explain_label = tk.Label(
            self.main_frame,
            text="",
            font=self.button_font,
            bg="#FFF8E7",
            fg="#666666"
        )
        self.explain_label.pack(pady=10)

        # 입력 프레임
        self.input_frame = tk.Frame(self.main_frame, bg="#FFF8E7")
        self.input_frame.pack(pady=20)

        self.answer_entry = tk.Entry(
            self.input_frame,
            font=self.problem_font,
            width=5,
            justify="center"
        )
        self.answer_entry.pack(side="left", padx=10)
        self.answer_entry.bind("<Return>", lambda e: self.check_answer())

        self.submit_btn = tk.Button(
            self.input_frame,
            text="확인",
            font=self.button_font,
            bg="#4ECDC4",
            fg="white",
            command=self.check_answer,
            padx=20,
            pady=5
        )
        self.submit_btn.pack(side="left", padx=10)

        # 결과 메시지
        self.result_label = tk.Label(
            self.main_frame,
            text="",
            font=self.title_font,
            bg="#FFF8E7"
        )
        self.result_label.pack(pady=10)

        # 정답 시각화
        self.answer_visual_label = tk.Label(
            self.main_frame,
            text="",
            font=self.fruit_font,
            bg="#FFF8E7",
            justify="left"
        )
        self.answer_visual_label.pack(pady=10)

        # 버튼 프레임
        self.button_frame = tk.Frame(self.main_frame, bg="#FFF8E7")
        self.button_frame.pack(pady=20)

        self.start_btn = tk.Button(
            self.button_frame,
            text="🎯 시작하기",
            font=self.button_font,
            bg="#FF6B6B",
            fg="white",
            command=self.start_quiz,
            padx=30,
            pady=10
        )

        self.next_btn = tk.Button(
            self.button_frame,
            text="다음 문제 ➡️",
            font=self.button_font,
            bg="#4ECDC4",
            fg="white",
            command=self.next_problem,
            padx=30,
            pady=10
        )

        self.hint_btn = tk.Button(
            self.button_frame,
            text="💡 힌트",
            font=self.button_font,
            bg="#FFE66D",
            fg="#333333",
            command=self.show_hint,
            padx=20,
            pady=10
        )

        self.restart_btn = tk.Button(
            self.button_frame,
            text="🔄 다시 시작",
            font=self.button_font,
            bg="#FF6B6B",
            fg="white",
            command=self.show_start_screen,
            padx=30,
            pady=10
        )

        # 틀린 문제 다시 풀기 버튼
        self.retry_btn = tk.Button(
            self.button_frame,
            text="🔄 틀린 문제 다시 풀기",
            font=self.button_font,
            bg="#FF9800",
            fg="white",
            command=self.retry_wrong_problems,
            padx=20,
            pady=10
        )

        # 점수 저장 버튼
        self.save_btn = tk.Button(
            self.button_frame,
            text="💾 점수 저장",
            font=self.button_font,
            bg="#2196F3",
            fg="white",
            command=self.save_current_score,
            padx=20,
            pady=10
        )

        # 기록 보기 버튼
        self.history_btn = tk.Button(
            self.button_frame,
            text="📊 기록 보기",
            font=self.button_font,
            bg="#9C27B0",
            fg="white",
            command=self.show_score_history,
            padx=20,
            pady=10
        )

        # 모드 선택 프레임 (곱셈/나눗셈)
        self.mode_frame = tk.Frame(self.main_frame, bg="#FFF8E7")

        self.mode_label = tk.Label(
            self.mode_frame,
            text="무엇을 연습할까요?",
            font=self.title_font,
            bg="#FFF8E7",
            fg="#333333"
        )
        self.mode_label.pack(pady=20)

        self.mode_buttons_frame = tk.Frame(self.mode_frame, bg="#FFF8E7")
        self.mode_buttons_frame.pack(pady=10)

        self.division_mode_btn = tk.Button(
            self.mode_buttons_frame,
            text="➗ 나눗셈",
            font=self.title_font,
            bg="#FF6B6B",
            fg="white",
            command=lambda: self.select_mode("division"),
            padx=30,
            pady=20
        )
        self.division_mode_btn.pack(side="left", padx=20)

        self.multiplication_mode_btn = tk.Button(
            self.mode_buttons_frame,
            text="✖️ 곱셈",
            font=self.title_font,
            bg="#4ECDC4",
            fg="white",
            command=lambda: self.select_mode("multiplication"),
            padx=30,
            pady=20
        )
        self.multiplication_mode_btn.pack(side="left", padx=20)

        # 숫자 선택 프레임 (나누기/곱하기 숫자)
        self.number_frame = tk.Frame(self.main_frame, bg="#FFF8E7")

        self.number_label = tk.Label(
            self.number_frame,
            text="① 숫자를 선택하세요:",
            font=self.button_font,
            bg="#FFF8E7",
            fg="#333333"
        )
        self.number_label.pack(pady=10)

        self.number_buttons_frame = tk.Frame(self.number_frame, bg="#FFF8E7")
        self.number_buttons_frame.pack(pady=10)

        # 2~5 버튼
        for num in range(2, 6):
            colors = {2: "#FF6B6B", 3: "#4ECDC4", 4: "#FFE66D", 5: "#95E1D3"}
            fg_colors = {2: "white", 3: "white", 4: "#333333", 5: "#333333"}
            btn = tk.Button(
                self.number_buttons_frame,
                text=f"{num}",
                font=self.title_font,
                bg=colors[num],
                fg=fg_colors[num],
                command=lambda n=num: self.select_number(n),
                padx=25,
                pady=15
            )
            btn.pack(side="left", padx=8)

        # 두 번째 줄 (6~10)
        self.number_buttons_frame2 = tk.Frame(self.number_frame, bg="#FFF8E7")
        self.number_buttons_frame2.pack(pady=10)

        for num in range(6, 11):
            colors = {6: "#DDA0DD", 7: "#9370DB", 8: "#8A2BE2", 9: "#7B68EE", 10: "#6A5ACD"}
            btn = tk.Button(
                self.number_buttons_frame2,
                text=f"{num}",
                font=self.title_font,
                bg=colors[num],
                fg="white",
                command=lambda n=num: self.select_number(n),
                padx=25,
                pady=15
            )
            btn.pack(side="left", padx=8)

        # 나누기 선택 프레임 (기존 - 사용 안함)
        self.divisor_frame = tk.Frame(self.main_frame, bg="#FFF8E7")

        self.divisor_label = tk.Label(
            self.divisor_frame,
            text="① 나누는 수를 선택하세요:",
            font=self.button_font,
            bg="#FFF8E7",
            fg="#333333"
        )
        self.divisor_label.pack(pady=10)

        self.divisor_buttons_frame = tk.Frame(self.divisor_frame, bg="#FFF8E7")
        self.divisor_buttons_frame.pack(pady=10)

        # 나누기 2, 3, 4, 5 버튼
        self.div2_btn = tk.Button(
            self.divisor_buttons_frame,
            text="➗ 2",
            font=self.title_font,
            bg="#FF6B6B",
            fg="white",
            command=lambda: self.select_divisor(2),
            padx=20,
            pady=15
        )
        self.div2_btn.pack(side="left", padx=10)

        self.div3_btn = tk.Button(
            self.divisor_buttons_frame,
            text="➗ 3",
            font=self.title_font,
            bg="#4ECDC4",
            fg="white",
            command=lambda: self.select_divisor(3),
            padx=20,
            pady=15
        )
        self.div3_btn.pack(side="left", padx=10)

        self.div4_btn = tk.Button(
            self.divisor_buttons_frame,
            text="➗ 4",
            font=self.title_font,
            bg="#FFE66D",
            fg="#333333",
            command=lambda: self.select_divisor(4),
            padx=20,
            pady=15
        )
        self.div4_btn.pack(side="left", padx=10)

        self.div5_btn = tk.Button(
            self.divisor_buttons_frame,
            text="➗ 5",
            font=self.title_font,
            bg="#95E1D3",
            fg="#333333",
            command=lambda: self.select_divisor(5),
            padx=20,
            pady=15
        )
        self.div5_btn.pack(side="left", padx=10)

        # 두 번째 줄 (6~10)
        self.divisor_buttons_frame2 = tk.Frame(self.divisor_frame, bg="#FFF8E7")
        self.divisor_buttons_frame2.pack(pady=10)

        self.div6_btn = tk.Button(
            self.divisor_buttons_frame2,
            text="➗ 6",
            font=self.title_font,
            bg="#DDA0DD",
            fg="white",
            command=lambda: self.select_divisor(6),
            padx=20,
            pady=15
        )
        self.div6_btn.pack(side="left", padx=10)

        self.div7_btn = tk.Button(
            self.divisor_buttons_frame2,
            text="➗ 7",
            font=self.title_font,
            bg="#9370DB",
            fg="white",
            command=lambda: self.select_divisor(7),
            padx=20,
            pady=15
        )
        self.div7_btn.pack(side="left", padx=10)

        self.div8_btn = tk.Button(
            self.divisor_buttons_frame2,
            text="➗ 8",
            font=self.title_font,
            bg="#8A2BE2",
            fg="white",
            command=lambda: self.select_divisor(8),
            padx=20,
            pady=15
        )
        self.div8_btn.pack(side="left", padx=10)

        self.div9_btn = tk.Button(
            self.divisor_buttons_frame2,
            text="➗ 9",
            font=self.title_font,
            bg="#7B68EE",
            fg="white",
            command=lambda: self.select_divisor(9),
            padx=20,
            pady=15
        )
        self.div9_btn.pack(side="left", padx=10)

        self.div10_btn = tk.Button(
            self.divisor_buttons_frame2,
            text="➗ 10",
            font=self.title_font,
            bg="#6A5ACD",
            fg="white",
            command=lambda: self.select_divisor(10),
            padx=18,
            pady=15
        )
        self.div10_btn.pack(side="left", padx=10)

        # 문제 수 선택 프레임
        self.count_frame = tk.Frame(self.main_frame, bg="#FFF8E7")

        self.count_label = tk.Label(
            self.count_frame,
            text="② 문제 수를 선택하세요:",
            font=self.button_font,
            bg="#FFF8E7",
            fg="#333333"
        )
        self.count_label.pack(pady=10)

        self.count_buttons_frame = tk.Frame(self.count_frame, bg="#FFF8E7")
        self.count_buttons_frame.pack(pady=10)

        # 10, 20, 50 문제 버튼
        self.count10_btn = tk.Button(
            self.count_buttons_frame,
            text="10문제",
            font=self.title_font,
            bg="#88D8B0",
            fg="#333333",
            command=lambda: self.start_quiz(10),
            padx=20,
            pady=15
        )
        self.count10_btn.pack(side="left", padx=10)

        self.count20_btn = tk.Button(
            self.count_buttons_frame,
            text="20문제",
            font=self.title_font,
            bg="#6BC5A0",
            fg="white",
            command=lambda: self.start_quiz(20),
            padx=20,
            pady=15
        )
        self.count20_btn.pack(side="left", padx=10)

        self.count50_btn = tk.Button(
            self.count_buttons_frame,
            text="50문제",
            font=self.title_font,
            bg="#4A9F8E",
            fg="white",
            command=lambda: self.start_quiz(50),
            padx=20,
            pady=15
        )
        self.count50_btn.pack(side="left", padx=10)

        # 선택된 나누기 표시
        self.selected_info = tk.Label(
            self.count_frame,
            text="",
            font=self.button_font,
            bg="#FFF8E7",
            fg="#FF6B6B"
        )
        self.selected_info.pack(pady=10)

    def show_start_screen(self):
        """시작 화면 표시"""
        self.title_label.config(text="🎈 나눗셈 연습 🎈")
        self.progress_label.config(text="")
        self.problem_label.config(text="환영합니다! 👋")
        self.fruit_label.config(text="🍎🍊🍋🍇🍓🍑🍒🥝🍌🫐")
        self.explain_label.config(text="과일로 나눗셈을 배워봐요!")
        self.result_label.config(text="")
        self.answer_visual_label.config(text="")

        # 분수 표시 숨기기
        self.fraction_frame.pack_forget()

        self.input_frame.pack_forget()
        self.hint_btn.pack_forget()
        self.next_btn.pack_forget()
        self.restart_btn.pack_forget()
        self.retry_btn.pack_forget()
        self.save_btn.pack_forget()
        self.start_btn.pack_forget()
        self.count_frame.pack_forget()
        self.divisor_frame.pack_forget()
        self.number_frame.pack_forget()
        self.mode_frame.pack(pady=20)
        self.history_btn.pack(pady=10)

        self.problems = []
        self.current_index = 0
        self.correct_count = 0
        self.wrong_problems = []

    def select_mode(self, mode):
        """모드 선택 (곱셈/나눗셈)"""
        self.mode = mode
        self.mode_frame.pack_forget()
        self.history_btn.pack_forget()

        if mode == "division":
            self.number_label.config(text="① 나누는 수를 선택하세요:")
            self.root.title("🎈 초등학생 나눗셈 연습 🎈")
        else:
            self.number_label.config(text="① 곱하는 수를 선택하세요:")
            self.root.title("🎈 초등학생 곱셈 연습 🎈")

        self.number_frame.pack(pady=20)

    def select_number(self, number):
        """숫자 선택"""
        self.selected_number = number
        self.number_frame.pack_forget()

        if self.mode == "division":
            self.selected_info.config(text=f"✅ 나누기 {number} 선택됨!")
        else:
            self.selected_info.config(text=f"✅ 곱하기 {number} 선택됨!")

        self.count_frame.pack(pady=20)

    def select_divisor(self, divisor):
        """나누는 수 선택 (기존 호환)"""
        self.selected_number = divisor
        self.divisor_frame.pack_forget()
        self.selected_info.config(text=f"✅ 나누기 {divisor} 선택됨!")
        self.count_frame.pack(pady=20)

    def generate_problems(self, count=10, number=2):
        """문제 생성 (나눗셈/곱셈)"""
        problems = []
        for _ in range(count):
            if self.mode == "division":
                quotient = random.randint(1, 10)
                dividend = number * quotient
                problems.append((dividend, number, quotient))
            else:  # multiplication
                multiplicand = random.randint(1, 10)
                product = number * multiplicand
                problems.append((number, multiplicand, product))
        return problems

    def start_quiz(self, count=10):
        """퀴즈 시작"""
        self.problem_count = count
        self.problems = self.generate_problems(count=count, number=self.selected_number)
        self.current_index = 0
        self.correct_count = 0
        self.wrong_problems = []

        self.start_btn.pack_forget()
        self.divisor_frame.pack_forget()
        self.number_frame.pack_forget()
        self.count_frame.pack_forget()
        self.retry_btn.pack_forget()
        self.save_btn.pack_forget()
        self.history_btn.pack_forget()
        self.mode_frame.pack_forget()
        self.save_btn.config(state="normal", text="💾 점수 저장")  # 저장 버튼 초기화
        self.input_frame.pack(pady=20)
        self.hint_btn.pack(side="left", padx=10)

        self.show_problem()

    def show_problem(self):
        """현재 문제 표시"""
        if self.current_index >= len(self.problems):
            self.show_final_result()
            return

        num1, num2, answer = self.problems[self.current_index]
        self.current_fruit = '🍎'  # 항상 사과 사용

        if self.mode == "division":
            # 나눗셈 모드
            self.title_label.config(text=f"🎈 나누기 {self.selected_number} 연습 🎈")
            self.progress_label.config(
                text=f"문제 {self.current_index + 1}/{len(self.problems)} | ✅ 맞은 개수: {self.correct_count}"
            )
            self.problem_label.config(text=f"{num1} ÷ {num2} = ?")

            # 분수 형태로 표시
            self.numerator_label.config(text=f"{num1}")
            self.denominator_label.config(text=f"{num2}")
            self.fraction_explain.config(text=f"(분자 {num1} ÷ 분모 {num2} = ?)")
            self.fraction_frame.pack(after=self.problem_label, pady=10)

            # 10개 이상이면 색깔 변화
            if num1 >= 10:
                fruit_display = ""
                for i in range(num1):
                    if i < 10:
                        fruit_display += "🍎"
                    elif i < 20:
                        fruit_display += "🍏"
                    elif i < 30:
                        fruit_display += "🍊"
                    elif i < 40:
                        fruit_display += "🍋"
                    else:
                        fruit_display += "🍇"
                self.fruit_label.config(text=f"🧺 사과 {num1}개:\n{fruit_display}")
            else:
                self.fruit_label.config(text=f"🧺 {self.current_fruit} {num1}개:\n{self.current_fruit * num1}")
            self.explain_label.config(text=f"👉 {num1}개를 {num2}명에게 똑같이 나눠주면?")
        else:
            # 곱셈 모드
            self.title_label.config(text=f"🎈 곱하기 {self.selected_number} 연습 🎈")
            self.progress_label.config(
                text=f"문제 {self.current_index + 1}/{len(self.problems)} | ✅ 맞은 개수: {self.correct_count}"
            )
            self.problem_label.config(text=f"{num1} × {num2} = ?")

            # 분수 프레임 숨기기
            self.fraction_frame.pack_forget()

            # 곱셈 시각화: num1개씩 num2묶음
            fruit_display = ""
            for row in range(num2):
                colors = ["🍎", "🍏", "🍊", "🍋", "🍇"]
                fruit = colors[row % len(colors)]
                fruit_display += fruit * num1 + "\n"
            self.fruit_label.config(text=f"🧺 {num1}개씩 {num2}묶음:\n{fruit_display}")
            self.explain_label.config(text=f"👉 {num1}개씩 {num2}묶음이면 모두 몇 개?")
        self.result_label.config(text="")
        self.answer_visual_label.config(text="")

        self.answer_entry.delete(0, tk.END)
        self.answer_entry.focus()

        self.next_btn.pack_forget()
        self.restart_btn.pack_forget()
        self.input_frame.pack(pady=20)
        self.hint_btn.pack(side="left", padx=10)

    def check_answer(self):
        """정답 확인"""
        try:
            user_answer = int(self.answer_entry.get())
        except ValueError:
            messagebox.showwarning("알림", "숫자를 입력해주세요!")
            return

        num1, num2, answer = self.problems[self.current_index]

        if user_answer == answer:
            play_sound("correct")
            self.result_label.config(text="🎉 정답이에요! 🎉", fg="#4ECDC4")
            self.correct_count += 1
        else:
            play_sound("wrong")
            self.result_label.config(text=f"😊 정답은 {answer}이에요!", fg="#FF6B6B")
            self.wrong_problems.append((num1, num2, answer))  # 틀린 문제 저장

        # 정답 시각화 (색깔 변화 적용)
        fruit_colors = ["🍎", "🍏", "🍊", "🍋", "🍇"]
        if self.mode == "division":
            visual_text = f"📦 나눠진 모습:\n"
            for i in range(num2):
                color_idx = i % len(fruit_colors)
                visual_text += f"  [{i+1}번] {fruit_colors[color_idx] * answer}\n"
        else:
            visual_text = f"📦 모두 합치면:\n"
            for i in range(num2):
                color_idx = i % len(fruit_colors)
                visual_text += f"  [{i+1}묶음] {fruit_colors[color_idx] * num1}\n"
            visual_text += f"\n  = 총 {answer}개!"
        self.answer_visual_label.config(text=visual_text)

        # 버튼 변경
        self.input_frame.pack_forget()
        self.hint_btn.pack_forget()
        self.next_btn.pack(pady=10)

    def next_problem(self):
        """다음 문제로"""
        self.current_index += 1
        self.show_problem()

    def show_hint(self):
        """힌트 표시"""
        num1, num2, answer = self.problems[self.current_index]
        if self.mode == "division":
            hint_text = f"💡 힌트: {self.current_fruit} {num2}개씩 묶어보세요!\n"
            hint_text += f"   {self.current_fruit * num2} ← 이게 1묶음이에요"
        else:
            hint_text = f"💡 힌트: {num1} + {num1} + ... ({num2}번 더하기)\n"
            hint_text += f"   {num1}을 {num2}번 더해보세요!"
        messagebox.showinfo("힌트", hint_text)

    def save_current_score(self):
        """현재 점수 저장"""
        percentage = (self.correct_count / len(self.problems)) * 100

        # 이름 입력 받기
        name = simpledialog.askstring("이름 입력", "이름을 입력하세요:", parent=self.root)
        if not name:
            name = "익명"

        # 점수 저장
        saved = save_score(name, self.correct_count, len(self.problems), percentage, self.selected_divisor)

        # 저장 완료 메시지
        messagebox.showinfo("저장 완료",
            f"✅ 점수가 저장되었습니다!\n\n"
            f"👤 이름: {saved['name']}\n"
            f"📅 날짜: {saved['date']}\n"
            f"✏️ 점수: {saved['correct']}/{saved['total']}\n"
            f"📈 정답률: {saved['percentage']}%\n"
            f"➗ 나누기: {saved['divisor']}")

        # 저장 버튼 비활성화
        self.save_btn.config(state="disabled", text="✅ 저장됨")

    def show_score_history(self):
        """점수 기록 보기"""
        scores = load_scores()

        if not scores:
            messagebox.showinfo("기록 없음", "아직 저장된 기록이 없어요!\n문제를 풀고 점수를 저장해보세요! 🎯")
            return

        # 새 창 만들기
        history_window = tk.Toplevel(self.root)
        history_window.title("📊 점수 기록")
        history_window.geometry("500x400")
        history_window.configure(bg="#FFF8E7")

        # 제목
        title = tk.Label(
            history_window,
            text="🏆 점수 기록 🏆",
            font=self.title_font,
            bg="#FFF8E7",
            fg="#FF6B6B"
        )
        title.pack(pady=10)

        # 스크롤 가능한 리스트
        frame = tk.Frame(history_window, bg="#FFF8E7")
        frame.pack(fill="both", expand=True, padx=20, pady=10)

        scrollbar = tk.Scrollbar(frame)
        scrollbar.pack(side="right", fill="y")

        listbox = tk.Listbox(
            frame,
            font=("Arial", 12),
            yscrollcommand=scrollbar.set,
            bg="#FFFAF0",
            height=15
        )
        listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=listbox.yview)

        # 최근 기록부터 표시
        for i, record in enumerate(reversed(scores[-20:]), 1):
            name = record.get('name', '익명')[:6]
            date = record.get('date', '-')[:10]
            score = f"{record['correct']}/{record['total']}"
            pct = f"{record['percentage']}%"
            div = f"÷{record.get('divisor', '?')}"
            listbox.insert("end", f"{i}. {name} | {date} | {score} ({pct}) | {div}")

        # 통계
        if scores:
            total_problems = sum(s['total'] for s in scores)
            total_correct = sum(s['correct'] for s in scores)
            avg_pct = (total_correct / total_problems * 100) if total_problems > 0 else 0

            stats = tk.Label(
                history_window,
                text=f"📈 총 {len(scores)}회 | 총 {total_problems}문제 | 평균 {avg_pct:.1f}%",
                font=self.button_font,
                bg="#FFF8E7",
                fg="#666666"
            )
            stats.pack(pady=10)

        # 닫기 버튼
        close_btn = tk.Button(
            history_window,
            text="닫기",
            font=self.button_font,
            bg="#FF6B6B",
            fg="white",
            command=history_window.destroy,
            padx=30,
            pady=5
        )
        close_btn.pack(pady=10)

    def retry_wrong_problems(self):
        """틀린 문제 다시 풀기"""
        if not self.wrong_problems:
            return

        # 틀린 문제들을 새 문제로 설정
        self.problems = self.wrong_problems.copy()
        self.wrong_problems = []
        self.current_index = 0
        self.correct_count = 0

        self.title_label.config(text="🔄 틀린 문제 다시 풀기 🔄")

        self.restart_btn.pack_forget()
        self.retry_btn.pack_forget()
        self.input_frame.pack(pady=20)
        self.hint_btn.pack(side="left", padx=10)

        self.show_problem()

    def show_final_result(self):
        """최종 결과 표시"""
        play_sound("complete")
        percentage = (self.correct_count / len(self.problems)) * 100

        self.title_label.config(text="🏆 결과 발표 🏆")
        self.progress_label.config(text="")
        self.problem_label.config(text=f"{self.correct_count}/{len(self.problems)} 정답!")

        # 분수 표시 숨기기
        self.fraction_frame.pack_forget()

        if percentage == 100:
            msg = "🌟 완벽해요! 천재예요! 🌟"
            self.fruit_label.config(text="⭐" * 10)
        elif percentage >= 80:
            msg = "🎉 아주 잘했어요! 🎉"
            self.fruit_label.config(text="🎉" * 8)
        elif percentage >= 60:
            msg = "👍 잘했어요! 👍"
            self.fruit_label.config(text="👍" * 6)
        else:
            msg = "💪 다시 도전해봐요! 💪"
            self.fruit_label.config(text="💪" * 5)

        self.explain_label.config(text=msg)
        self.result_label.config(text=f"정답률: {percentage:.0f}%", fg="#333333")

        # 틀린 문제 목록 표시
        if self.wrong_problems:
            wrong_text = "📖 틀린 문제:\n"
            for dividend, divisor, quotient in self.wrong_problems:
                wrong_text += f"  {dividend} ÷ {divisor} = {quotient}\n"
            self.answer_visual_label.config(text=wrong_text)
        else:
            self.answer_visual_label.config(text="")

        self.input_frame.pack_forget()
        self.hint_btn.pack_forget()
        self.next_btn.pack_forget()
        self.history_btn.pack_forget()

        # 버튼들 표시
        self.save_btn.pack(side="left", padx=5, pady=20)
        if self.wrong_problems:
            self.retry_btn.pack(side="left", padx=5, pady=20)
        self.restart_btn.pack(side="left", padx=5, pady=20)


def main():
    root = tk.Tk()
    app = DivisionPracticeGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
