#!/usr/bin/env python3
"""
초등학생을 위한 나눗셈 연습 프로그램
시각적인 과일 모델을 사용하여 나눗셈을 쉽게 이해할 수 있도록 도와줍니다.
"""

import random
import time
import json
import os
import threading
from datetime import datetime, timedelta

# 점수 기록 파일 경로
SCORE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scores.json")
BADGE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "badges.json")
DAILY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "daily.json")

# 과일 이모지 목록
FRUITS = ['🍎', '🍊', '🍋', '🍇', '🍓', '🍑', '🍒', '🥝', '🍌', '🫐']

# 업적 뱃지 정의
BADGES = {
    # 첫 도전
    "first_try": {
        "name": "첫 발걸음",
        "icon": "👶",
        "description": "첫 번째 연습 완료!",
        "condition": "첫 번째 퀴즈 완료"
    },
    # 완벽한 점수
    "perfect_10": {
        "name": "완벽한 10문제",
        "icon": "⭐",
        "description": "10문제 모두 정답!",
        "condition": "10문제 100% 달성"
    },
    "perfect_50": {
        "name": "완벽한 50문제",
        "icon": "🌟",
        "description": "50문제 모두 정답!",
        "condition": "50문제 100% 달성"
    },
    "perfect_100": {
        "name": "완벽한 100문제",
        "icon": "💫",
        "description": "100문제 모두 정답! 대단해요!",
        "condition": "100문제 100% 달성"
    },
    # 연습량
    "practice_5": {
        "name": "연습벌레",
        "icon": "🐛",
        "description": "5번 연습 완료!",
        "condition": "5회 연습"
    },
    "practice_10": {
        "name": "연습왕",
        "icon": "👑",
        "description": "10번 연습 완료!",
        "condition": "10회 연습"
    },
    "practice_30": {
        "name": "나눗셈 마스터",
        "icon": "🎓",
        "description": "30번 연습 완료!",
        "condition": "30회 연습"
    },
    # 문제 수
    "solve_100": {
        "name": "100문제 돌파",
        "icon": "💯",
        "description": "총 100문제 해결!",
        "condition": "누적 100문제"
    },
    "solve_500": {
        "name": "500문제 돌파",
        "icon": "🔥",
        "description": "총 500문제 해결!",
        "condition": "누적 500문제"
    },
    "solve_1000": {
        "name": "1000문제 돌파",
        "icon": "🏆",
        "description": "총 1000문제 해결! 전설이에요!",
        "condition": "누적 1000문제"
    },
    # 정답률
    "avg_70": {
        "name": "안정적인 실력",
        "icon": "📗",
        "description": "평균 정답률 70% 달성!",
        "condition": "평균 70% 이상"
    },
    "avg_90": {
        "name": "뛰어난 실력",
        "icon": "📘",
        "description": "평균 정답률 90% 달성!",
        "condition": "평균 90% 이상"
    },
    # 연속 기록
    "streak_3": {
        "name": "3연속 성장",
        "icon": "📈",
        "description": "3회 연속 점수 상승!",
        "condition": "3연속 성장"
    },
    "streak_5": {
        "name": "5연속 성장",
        "icon": "🚀",
        "description": "5회 연속 점수 상승!",
        "condition": "5연속 성장"
    },
    # 난이도 도전
    "difficulty_3": {
        "name": "어려움 도전자",
        "icon": "🌳",
        "description": "어려움 난이도 80% 이상!",
        "condition": "어려움 80%+"
    },
    "difficulty_4": {
        "name": "도전 정복자",
        "icon": "🔥",
        "description": "도전 난이도 80% 이상!",
        "condition": "도전! 80%+"
    },
    # 속도
    "speed_master": {
        "name": "번개 손가락",
        "icon": "⚡",
        "description": "번개 모드로 70% 이상!",
        "condition": "번개모드 70%+"
    },
    # 특별 업적
    "early_bird": {
        "name": "아침형 인간",
        "icon": "🌅",
        "description": "오전 7시 이전에 연습!",
        "condition": "오전 7시 전 연습"
    },
    "night_owl": {
        "name": "밤샘 공부왕",
        "icon": "🦉",
        "description": "밤 10시 이후에 연습!",
        "condition": "밤 10시 후 연습"
    },
    # 일일 도전 업적
    "daily_first": {
        "name": "첫 일일 도전",
        "icon": "📅",
        "description": "첫 번째 일일 도전 완료!",
        "condition": "일일 도전 1회"
    },
    "daily_7": {
        "name": "일주일 연속 도전",
        "icon": "🔥",
        "description": "7일 연속 일일 도전!",
        "condition": "7일 연속 도전"
    },
    "daily_30": {
        "name": "한 달 연속 도전",
        "icon": "🌙",
        "description": "30일 연속 일일 도전! 대단해요!",
        "condition": "30일 연속 도전"
    },
    "daily_perfect": {
        "name": "완벽한 하루",
        "icon": "💎",
        "description": "일일 도전 100% 달성!",
        "condition": "일일 도전 100%"
    },
}

# 타이머 설정
TIMER_SETTINGS = {
    0: {"name": "없음 ⏸️", "seconds": 0, "description": "시간 제한 없이 천천히"},
    1: {"name": "여유 🐢", "seconds": 30, "description": "문제당 30초"},
    2: {"name": "보통 🐇", "seconds": 15, "description": "문제당 15초"},
    3: {"name": "빠름 🚀", "seconds": 10, "description": "문제당 10초"},
    4: {"name": "번개 ⚡", "seconds": 5, "description": "문제당 5초"}
}

class TimerInput:
    """시간 제한이 있는 입력을 처리하는 클래스"""

    def __init__(self, timeout):
        self.timeout = timeout
        self.answer = None
        self.timed_out = False
        self.remaining = timeout

    def get_input(self, prompt):
        """시간 제한 내에 입력을 받습니다."""
        if self.timeout <= 0:
            # 시간 제한 없음
            return input(prompt)

        self.answer = None
        self.timed_out = False
        self.remaining = self.timeout

        # 입력 스레드
        def input_thread():
            try:
                self.answer = input(prompt)
            except EOFError:
                self.answer = 'q'

        thread = threading.Thread(target=input_thread)
        thread.daemon = True
        thread.start()

        # 타이머 카운트다운
        start_time = time.time()
        while thread.is_alive() and self.remaining > 0:
            thread.join(timeout=1)
            elapsed = time.time() - start_time
            self.remaining = max(0, self.timeout - int(elapsed))

            if thread.is_alive() and self.remaining > 0:
                # 남은 시간 표시 (같은 줄에 업데이트)
                if self.remaining <= 5:
                    print(f"\r⏰ 남은 시간: {self.remaining}초 ⚠️  ", end="", flush=True)
                else:
                    print(f"\r⏰ 남은 시간: {self.remaining}초    ", end="", flush=True)

        if thread.is_alive():
            # 시간 초과
            self.timed_out = True
            print(f"\r⏰ 시간 초과! ⏰           ")
            return None

        print("\r                              ", end="\r")  # 타이머 표시 지우기
        return self.answer

def format_time(seconds):
    """초를 분:초 형식으로 변환합니다."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes > 0:
        return f"{minutes}분 {secs}초"
    return f"{secs}초"

def select_timer():
    """타이머 설정을 선택합니다."""
    print("\n" + "⏱️" * 20)
    print("\n      ⏰ 시간 제한 선택 ⏰\n")
    print("⏱️" * 20)

    for level, settings in TIMER_SETTINGS.items():
        print(f"\n  {level}. {settings['name']}")
        print(f"     {settings['description']}")

    print("\n" + "-" * 40)

    while True:
        choice = input("\n시간 제한을 선택하세요 (0-4): ").strip()
        if choice in ['0', '1', '2', '3', '4']:
            level = int(choice)
            print(f"\n✅ '{TIMER_SETTINGS[level]['name']}' 모드를 선택했어요!")
            return level
        else:
            print("❌ 0, 1, 2, 3, 4 중에서 선택해주세요!")

def load_scores():
    """저장된 점수 기록을 불러옵니다."""
    if os.path.exists(SCORE_FILE):
        try:
            with open(SCORE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []
    return []

def save_score(name, correct, total, percentage):
    """점수를 파일에 저장합니다."""
    scores = load_scores()

    score_entry = {
        "name": name,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "correct": correct,
        "total": total,
        "percentage": round(percentage, 1)
    }

    scores.append(score_entry)

    # 최근 100개만 유지
    if len(scores) > 100:
        scores = scores[-100:]

    with open(SCORE_FILE, 'w', encoding='utf-8') as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)

    return score_entry

def load_badges():
    """저장된 뱃지를 불러옵니다."""
    if os.path.exists(BADGE_FILE):
        try:
            with open(BADGE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {"earned": [], "history": []}
    return {"earned": [], "history": []}

def save_badges(badges_data):
    """뱃지를 저장합니다."""
    with open(BADGE_FILE, 'w', encoding='utf-8') as f:
        json.dump(badges_data, f, ensure_ascii=False, indent=2)

def award_badge(badge_id):
    """새 뱃지를 수여합니다."""
    badges_data = load_badges()

    if badge_id not in badges_data["earned"]:
        badges_data["earned"].append(badge_id)
        badges_data["history"].append({
            "badge_id": badge_id,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M")
        })
        save_badges(badges_data)
        return True
    return False

def check_and_award_badges(correct, total, percentage, difficulty=2, timer_level=0):
    """
    조건을 확인하고 새 뱃지를 수여합니다.
    새로 획득한 뱃지 목록을 반환합니다.
    """
    new_badges = []
    scores = load_scores()
    badges_data = load_badges()
    earned = badges_data.get("earned", [])

    # 현재 시간
    current_hour = datetime.now().hour

    # 1. 첫 도전
    if "first_try" not in earned and len(scores) >= 1:
        if award_badge("first_try"):
            new_badges.append("first_try")

    # 2. 완벽한 점수
    if percentage == 100:
        if total >= 10 and "perfect_10" not in earned:
            if award_badge("perfect_10"):
                new_badges.append("perfect_10")
        if total >= 50 and "perfect_50" not in earned:
            if award_badge("perfect_50"):
                new_badges.append("perfect_50")
        if total >= 100 and "perfect_100" not in earned:
            if award_badge("perfect_100"):
                new_badges.append("perfect_100")

    # 3. 연습 횟수
    practice_count = len(scores)
    if practice_count >= 5 and "practice_5" not in earned:
        if award_badge("practice_5"):
            new_badges.append("practice_5")
    if practice_count >= 10 and "practice_10" not in earned:
        if award_badge("practice_10"):
            new_badges.append("practice_10")
    if practice_count >= 30 and "practice_30" not in earned:
        if award_badge("practice_30"):
            new_badges.append("practice_30")

    # 4. 총 문제 수
    total_problems = sum(s['total'] for s in scores)
    if total_problems >= 100 and "solve_100" not in earned:
        if award_badge("solve_100"):
            new_badges.append("solve_100")
    if total_problems >= 500 and "solve_500" not in earned:
        if award_badge("solve_500"):
            new_badges.append("solve_500")
    if total_problems >= 1000 and "solve_1000" not in earned:
        if award_badge("solve_1000"):
            new_badges.append("solve_1000")

    # 5. 평균 정답률
    if len(scores) >= 3:
        total_correct = sum(s['correct'] for s in scores)
        avg_pct = (total_correct / total_problems * 100) if total_problems > 0 else 0
        if avg_pct >= 70 and "avg_70" not in earned:
            if award_badge("avg_70"):
                new_badges.append("avg_70")
        if avg_pct >= 90 and "avg_90" not in earned:
            if award_badge("avg_90"):
                new_badges.append("avg_90")

    # 6. 연속 성장
    if len(scores) >= 3:
        percentages = [s['percentage'] for s in scores]
        streak = 1
        for i in range(len(percentages) - 1, 0, -1):
            if percentages[i] > percentages[i-1]:
                streak += 1
            else:
                break
        if streak >= 3 and "streak_3" not in earned:
            if award_badge("streak_3"):
                new_badges.append("streak_3")
        if streak >= 5 and "streak_5" not in earned:
            if award_badge("streak_5"):
                new_badges.append("streak_5")

    # 7. 난이도 도전
    if difficulty == 3 and percentage >= 80 and "difficulty_3" not in earned:
        if award_badge("difficulty_3"):
            new_badges.append("difficulty_3")
    if difficulty == 4 and percentage >= 80 and "difficulty_4" not in earned:
        if award_badge("difficulty_4"):
            new_badges.append("difficulty_4")

    # 8. 속도 도전
    if timer_level == 4 and percentage >= 70 and "speed_master" not in earned:
        if award_badge("speed_master"):
            new_badges.append("speed_master")

    # 9. 시간대 업적
    if current_hour < 7 and "early_bird" not in earned:
        if award_badge("early_bird"):
            new_badges.append("early_bird")
    if current_hour >= 22 and "night_owl" not in earned:
        if award_badge("night_owl"):
            new_badges.append("night_owl")

    return new_badges

def show_new_badges(new_badges):
    """새로 획득한 뱃지를 축하 메시지와 함께 보여줍니다."""
    if not new_badges:
        return

    print("\n" + "🎊" * 20)
    print("\n   🎉 새로운 뱃지 획득! 🎉\n")
    print("🎊" * 20)

    for badge_id in new_badges:
        badge = BADGES.get(badge_id, {})
        print(f"\n   {badge.get('icon', '🏅')} {badge.get('name', badge_id)}")
        print(f"      \"{badge.get('description', '')}\"")

    print("\n" + "🎊" * 20)
    input("\n축하해요! Enter를 눌러 계속하세요...")

def show_all_badges():
    """모든 뱃지와 획득 현황을 보여줍니다."""
    badges_data = load_badges()
    earned = badges_data.get("earned", [])
    history = badges_data.get("history", [])

    clear_screen()
    print("🏅" * 20)
    print("\n      🎖️ 나의 뱃지 컬렉션 🎖️\n")
    print("🏅" * 20)

    # 통계
    total_badges = len(BADGES)
    earned_count = len(earned)
    progress = (earned_count / total_badges * 100) if total_badges > 0 else 0

    print(f"\n📊 수집 현황: {earned_count}/{total_badges}개 ({progress:.1f}%)")

    # 진행 바
    bar_length = 20
    filled = int(bar_length * earned_count / total_badges)
    bar = "█" * filled + "░" * (bar_length - filled)
    print(f"   [{bar}]")

    # 획득한 뱃지
    print("\n" + "=" * 50)
    print("✨ 획득한 뱃지:")
    print("-" * 50)

    if earned:
        for badge_id in earned:
            badge = BADGES.get(badge_id, {})
            # 획득 날짜 찾기
            date = ""
            for h in history:
                if h.get("badge_id") == badge_id:
                    date = h.get("date", "")[:10]
                    break
            print(f"   {badge.get('icon', '🏅')} {badge.get('name', badge_id)}")
            print(f"      {badge.get('description', '')} ({date})")
    else:
        print("   아직 획득한 뱃지가 없어요! 연습해서 뱃지를 모아보세요! 💪")

    # 미획득 뱃지
    print("\n" + "=" * 50)
    print("🔒 아직 획득하지 못한 뱃지:")
    print("-" * 50)

    not_earned = [b_id for b_id in BADGES if b_id not in earned]
    if not_earned:
        for badge_id in not_earned:
            badge = BADGES.get(badge_id, {})
            print(f"   🔒 {badge.get('name', badge_id)}")
            print(f"      조건: {badge.get('condition', '???')}")
    else:
        print("   🎉 모든 뱃지를 획득했어요! 대단해요! 🎉")

    print("\n" + "=" * 50)

# ==================== 일일 도전 기능 ====================

def get_today_string():
    """오늘 날짜를 문자열로 반환합니다."""
    return datetime.now().strftime("%Y-%m-%d")

def load_daily_data():
    """일일 도전 데이터를 불러옵니다."""
    if os.path.exists(DAILY_FILE):
        try:
            with open(DAILY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {"completed_dates": [], "streaks": {"current": 0, "best": 0}, "history": []}
    return {"completed_dates": [], "streaks": {"current": 0, "best": 0}, "history": []}

def save_daily_data(data):
    """일일 도전 데이터를 저장합니다."""
    with open(DAILY_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_daily_seed():
    """오늘 날짜를 기반으로 시드를 생성합니다."""
    today = get_today_string()
    # 날짜 문자열을 숫자로 변환하여 시드로 사용
    seed = sum(ord(c) * (i + 1) for i, c in enumerate(today))
    return seed

def get_daily_challenge_problems():
    """
    오늘의 도전 문제를 생성합니다.
    매일 같은 문제가 출제됩니다 (날짜 기반 시드).
    """
    seed = get_daily_seed()
    daily_random = random.Random(seed)

    # 오늘의 난이도 (요일에 따라 다름)
    weekday = datetime.now().weekday()  # 0=월, 6=일

    if weekday < 2:  # 월, 화: 쉬움
        difficulty = 1
    elif weekday < 4:  # 수, 목: 보통
        difficulty = 2
    elif weekday < 6:  # 금, 토: 어려움
        difficulty = 3
    else:  # 일: 도전
        difficulty = 4

    settings = DIFFICULTY_SETTINGS.get(difficulty, DIFFICULTY_SETTINGS[2])
    max_dividend = settings["max_dividend"]
    max_divisor = settings["max_divisor"]
    max_quotient = settings["max_quotient"]

    # 가능한 조합 찾기
    valid_combinations = []
    for divisor in range(1, max_divisor + 1):
        for quotient in range(1, max_quotient + 1):
            dividend = divisor * quotient
            if dividend <= max_dividend:
                valid_combinations.append((dividend, divisor, quotient))

    # 오늘의 문제 10개 선택
    problems = []
    for _ in range(10):
        problem = daily_random.choice(valid_combinations)
        problems.append(problem)

    return problems, difficulty

def calculate_streak(completed_dates):
    """연속 도전 일수를 계산합니다."""
    if not completed_dates:
        return 0

    # 날짜 정렬
    sorted_dates = sorted(completed_dates, reverse=True)

    today = get_today_string()
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    # 오늘 또는 어제 완료했는지 확인
    if sorted_dates[0] != today and sorted_dates[0] != yesterday:
        return 0

    streak = 1
    for i in range(len(sorted_dates) - 1):
        current = datetime.strptime(sorted_dates[i], "%Y-%m-%d")
        prev = datetime.strptime(sorted_dates[i + 1], "%Y-%m-%d")
        diff = (current - prev).days

        if diff == 1:
            streak += 1
        elif diff > 1:
            break

    return streak

def check_daily_completed():
    """오늘의 도전을 완료했는지 확인합니다."""
    data = load_daily_data()
    today = get_today_string()
    return today in data.get("completed_dates", [])

def complete_daily_challenge(correct, total, percentage):
    """일일 도전 완료를 기록합니다."""
    data = load_daily_data()
    today = get_today_string()

    if today not in data["completed_dates"]:
        data["completed_dates"].append(today)

    # 기록 추가
    data["history"].append({
        "date": today,
        "correct": correct,
        "total": total,
        "percentage": percentage
    })

    # 최근 100개만 유지
    if len(data["history"]) > 100:
        data["history"] = data["history"][-100:]

    # 연속 기록 계산
    current_streak = calculate_streak(data["completed_dates"])
    data["streaks"]["current"] = current_streak
    if current_streak > data["streaks"].get("best", 0):
        data["streaks"]["best"] = current_streak

    save_daily_data(data)

    # 일일 도전 뱃지 확인
    new_badges = []
    badges_data = load_badges()
    earned = badges_data.get("earned", [])

    # 첫 일일 도전
    if "daily_first" not in earned:
        if award_badge("daily_first"):
            new_badges.append("daily_first")

    # 7일 연속
    if current_streak >= 7 and "daily_7" not in earned:
        if award_badge("daily_7"):
            new_badges.append("daily_7")

    # 30일 연속
    if current_streak >= 30 and "daily_30" not in earned:
        if award_badge("daily_30"):
            new_badges.append("daily_30")

    # 완벽한 하루
    if percentage == 100 and "daily_perfect" not in earned:
        if award_badge("daily_perfect"):
            new_badges.append("daily_perfect")

    return current_streak, new_badges

def run_daily_challenge():
    """일일 도전을 실행합니다."""
    clear_screen()

    # 이미 완료했는지 확인
    if check_daily_completed():
        print("📅" * 20)
        print("\n   ✅ 오늘의 도전을 이미 완료했어요!\n")
        print("📅" * 20)

        data = load_daily_data()
        today = get_today_string()

        # 오늘 기록 찾기
        today_record = None
        for record in reversed(data.get("history", [])):
            if record.get("date") == today:
                today_record = record
                break

        if today_record:
            print(f"\n   📊 오늘의 기록:")
            print(f"      정답: {today_record['correct']}/{today_record['total']}")
            print(f"      정답률: {today_record['percentage']}%")

        print(f"\n   🔥 현재 연속: {data['streaks']['current']}일")
        print(f"   🏆 최고 연속: {data['streaks']['best']}일")
        print("\n   내일 다시 도전해주세요! 👋")

        return None

    # 오늘의 문제 생성
    problems, difficulty = get_daily_challenge_problems()
    difficulty_name = DIFFICULTY_SETTINGS[difficulty]["name"]
    weekday_names = ["월", "화", "수", "목", "금", "토", "일"]
    weekday = datetime.now().weekday()

    print("📅" * 20)
    print("\n   🌟 오늘의 도전 🌟\n")
    print("📅" * 20)

    print(f"\n   📆 {get_today_string()} ({weekday_names[weekday]}요일)")
    print(f"   📊 오늘의 난이도: {difficulty_name}")
    print(f"   📝 문제 수: 10문제")
    print(f"   ⏱️ 시간 제한: 없음")

    # 연속 기록 표시
    data = load_daily_data()
    current_streak = data["streaks"].get("current", 0)
    if current_streak > 0:
        print(f"\n   🔥 현재 {current_streak}일 연속 도전 중!")

    print("\n" + "-" * 40)
    print("   💡 매일 새로운 문제가 출제됩니다!")
    print("   💡 하루에 한 번만 도전할 수 있어요!")
    print("-" * 40)

    start = input("\n   도전하시겠어요? (y/n): ").strip().lower()
    if start != 'y':
        return None

    # 퀴즈 실행
    correct_count = 0
    wrong_problems = []

    print("\n" + "🌟" * 25)
    print("\n   🎓 오늘의 도전을 시작합니다! 🎓")
    print("\n" + "🌟" * 25)

    input("\n준비되면 Enter를 눌러주세요! ")

    for i, (dividend, divisor, quotient) in enumerate(problems, 1):
        clear_screen()

        fruit = random.choice(FRUITS)

        print(f"📅 오늘의 도전 {i}/10 | 난이도: {difficulty_name}")
        print(f"✅ 맞은 문제: {correct_count}개")

        display_division_visual(dividend, divisor, quotient, fruit)

        attempts = 0
        max_attempts = 2  # 일일 도전은 2번 기회

        while attempts < max_attempts:
            answer = input("🤔 정답을 입력하세요: ").strip().lower()

            if answer == 'h':
                print(f"\n💡 힌트: {fruit} {divisor}개씩 묶어보세요!")
                continue

            try:
                user_answer = int(answer)
            except ValueError:
                print("❌ 숫자를 입력해주세요!")
                continue

            attempts += 1

            if user_answer == quotient:
                print("\n🎉 정답이에요! 🎉")
                correct_count += 1
                show_answer_visual(dividend, divisor, quotient, fruit)
                break
            else:
                remaining = max_attempts - attempts
                if remaining > 0:
                    print(f"\n😅 아쉬워요! (남은 기회: {remaining}번)")
                else:
                    print(f"\n😊 정답을 알려줄게요.")
                    show_answer_visual(dividend, divisor, quotient, fruit)
                    wrong_problems.append((dividend, divisor, quotient))

        if i < 10:
            input("\n다음 문제로 가려면 Enter를 눌러주세요! ")

    # 결과 계산
    total = len(problems)
    percentage = (correct_count / total * 100) if total > 0 else 0

    # 완료 기록
    current_streak, new_badges = complete_daily_challenge(correct_count, total, percentage)

    # 결과 표시
    clear_screen()
    print("📅" * 20)
    print("\n   🎊 오늘의 도전 완료! 🎊\n")
    print("📅" * 20)

    print(f"\n   ✅ 맞은 문제: {correct_count}/10")
    print(f"   📈 정답률: {percentage:.1f}%")
    print(f"\n   🔥 연속 도전: {current_streak}일")

    # 격려 메시지
    print("\n" + "-" * 40)
    if percentage == 100:
        print("   💎 완벽해요! 최고의 하루예요! 💎")
    elif percentage >= 80:
        print("   🌟 아주 잘했어요! 🌟")
    elif percentage >= 60:
        print("   👍 잘했어요! 내일도 도전해요! 👍")
    else:
        print("   💪 내일은 더 잘할 수 있어요! 💪")
    print("-" * 40)

    # 틀린 문제
    if wrong_problems:
        print("\n📖 틀린 문제:")
        for dividend, divisor, quotient in wrong_problems:
            print(f"   {dividend} ÷ {divisor} = {quotient}")

    # 새 뱃지
    if new_badges:
        show_new_badges(new_badges)

    return percentage

def show_daily_stats():
    """일일 도전 통계를 보여줍니다."""
    data = load_daily_data()

    clear_screen()
    print("📅" * 20)
    print("\n   📊 일일 도전 통계 📊\n")
    print("📅" * 20)

    completed = data.get("completed_dates", [])
    history = data.get("history", [])
    streaks = data.get("streaks", {"current": 0, "best": 0})

    print(f"\n   📆 총 도전 일수: {len(completed)}일")
    print(f"   🔥 현재 연속: {streaks.get('current', 0)}일")
    print(f"   🏆 최고 연속: {streaks.get('best', 0)}일")

    if history:
        avg_pct = sum(h['percentage'] for h in history) / len(history)
        perfect_days = sum(1 for h in history if h['percentage'] == 100)
        print(f"\n   📈 평균 정답률: {avg_pct:.1f}%")
        print(f"   💎 완벽한 날: {perfect_days}일")

        # 최근 7일 기록
        print("\n" + "-" * 40)
        print("   📅 최근 기록:")
        print("-" * 40)

        recent = history[-7:][::-1]
        for record in recent:
            date = record.get('date', '-')
            pct = record.get('percentage', 0)
            bar_len = int(pct / 10)
            bar = "█" * bar_len + "░" * (10 - bar_len)
            print(f"   {date}: [{bar}] {pct}%")

    print("\n" + "-" * 40)

def show_score_history():
    """점수 기록을 보여줍니다."""
    scores = load_scores()

    clear_screen()
    print("📊" * 20)
    print("\n      🏆 점수 기록 🏆\n")
    print("📊" * 20)

    if not scores:
        print("\n   아직 기록이 없어요!")
        print("   문제를 풀고 점수를 저장해보세요! 🎯\n")
        return

    # 최근 10개 기록
    recent = scores[-10:][::-1]  # 최신순으로 정렬

    print("\n📅 최근 기록 (최대 10개):")
    print("-" * 50)
    print(f"{'순위':<4} {'이름':<8} {'날짜':<12} {'점수':<12} {'정답률':<8}")
    print("-" * 50)

    for i, record in enumerate(recent, 1):
        name = record.get('name', '익명')[:6]
        date = record.get('date', '-')[:10]
        score_str = f"{record['correct']}/{record['total']}"
        pct = f"{record['percentage']}%"
        print(f"{i:<4} {name:<8} {date:<12} {score_str:<12} {pct:<8}")

    print("-" * 50)

    # 최고 기록 (정답률 기준)
    if scores:
        best = max(scores, key=lambda x: (x['percentage'], x['correct']))
        print(f"\n🥇 최고 기록: {best.get('name', '익명')} - {best['percentage']}% ({best['correct']}/{best['total']})")
        print(f"   📅 {best['date']}")

    # 통계
    total_problems = sum(s['total'] for s in scores)
    total_correct = sum(s['correct'] for s in scores)
    avg_percentage = (total_correct / total_problems * 100) if total_problems > 0 else 0

    print(f"\n📈 전체 통계:")
    print(f"   총 연습 횟수: {len(scores)}회")
    print(f"   총 푼 문제: {total_problems}개")
    print(f"   총 맞은 문제: {total_correct}개")
    print(f"   평균 정답률: {avg_percentage:.1f}%")
    print()

def show_growth_graph():
    """성장 그래프를 보여줍니다."""
    scores = load_scores()

    clear_screen()
    print("📈" * 20)
    print("\n      🌱 나의 성장 그래프 🌱\n")
    print("📈" * 20)

    if len(scores) < 2:
        print("\n   📊 그래프를 그리려면 최소 2번 이상 연습해야 해요!")
        print("   열심히 연습하고 다시 와주세요! 💪\n")
        return

    # 최근 15개 기록 사용
    recent_scores = scores[-15:]

    # 그래프 높이
    graph_height = 10
    graph_width = len(recent_scores)

    # 정답률 데이터
    percentages = [s['percentage'] for s in recent_scores]

    print("\n📊 정답률 변화 그래프")
    print("-" * (graph_width * 4 + 10))

    # 그래프 그리기
    for row in range(graph_height, 0, -1):
        threshold = row * 10  # 10%, 20%, ... 100%

        if row == graph_height:
            line = "100% |"
        elif row == graph_height // 2:
            line = " 50% |"
        elif row == 1:
            line = " 10% |"
        else:
            line = "     |"

        for pct in percentages:
            if pct >= threshold:
                if pct >= 90:
                    line += " 🟢 "  # 90% 이상: 초록
                elif pct >= 70:
                    line += " 🟡 "  # 70% 이상: 노랑
                elif pct >= 50:
                    line += " 🟠 "  # 50% 이상: 주황
                else:
                    line += " 🔴 "  # 50% 미만: 빨강
            else:
                line += "    "

        print(line)

    # x축
    print("     +" + "----" * graph_width)
    print("      ", end="")
    for i in range(len(recent_scores)):
        print(f" {i+1:2} ", end="")
    print(" (회차)")

    # 범례
    print("\n" + "-" * 40)
    print("  범례: 🟢 90%↑  🟡 70%↑  🟠 50%↑  🔴 50%↓")
    print("-" * 40)

    # 통계 분석
    print("\n📊 성장 분석:")

    # 평균
    avg = sum(percentages) / len(percentages)
    print(f"   📌 평균 정답률: {avg:.1f}%")

    # 최고/최저
    best = max(percentages)
    worst = min(percentages)
    print(f"   🏆 최고 기록: {best}%")
    print(f"   📉 최저 기록: {worst}%")

    # 성장 추이
    if len(percentages) >= 3:
        first_half = percentages[:len(percentages)//2]
        second_half = percentages[len(percentages)//2:]
        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)

        diff = second_avg - first_avg
        if diff > 5:
            print(f"\n   🚀 대단해요! 실력이 {diff:.1f}% 향상됐어요!")
        elif diff > 0:
            print(f"\n   📈 조금씩 성장하고 있어요! (+{diff:.1f}%)")
        elif diff > -5:
            print(f"\n   💪 꾸준히 연습하고 있어요!")
        else:
            print(f"\n   😊 조금 어려웠나요? 천천히 다시 해봐요!")

    # 연속 기록 분석
    if len(percentages) >= 2:
        streak = 1
        improving = percentages[-1] >= percentages[-2]
        for i in range(len(percentages) - 2, -1, -1):
            if (percentages[i+1] >= percentages[i]) == improving:
                streak += 1
            else:
                break

        if improving and streak >= 3:
            print(f"   🔥 {streak}회 연속 성장 중!")
        elif percentages[-1] >= 90:
            print(f"   ⭐ 최근 성적이 아주 좋아요!")

    # 날짜별 기록
    print("\n📅 상세 기록:")
    print("-" * 50)
    for i, score in enumerate(recent_scores, 1):
        bar_length = int(score['percentage'] / 5)  # 최대 20칸
        bar = "█" * bar_length + "░" * (20 - bar_length)
        date = score.get('date', '-')[:10]
        name = score.get('name', '익명')[:4]
        print(f"  {i:2}. {date} {name}: [{bar}] {score['percentage']}%")
    print("-" * 50)

def clear_screen():
    """화면을 깔끔하게 만들기 위한 줄바꿈"""
    print("\n" + "=" * 50 + "\n")

# 난이도 설정
DIFFICULTY_SETTINGS = {
    1: {  # 쉬움
        "name": "쉬움 🌱",
        "max_dividend": 10,
        "max_divisor": 5,
        "max_quotient": 5,
        "description": "1~10 범위, 작은 수로 나누기"
    },
    2: {  # 보통
        "name": "보통 🌿",
        "max_dividend": 20,
        "max_divisor": 5,
        "max_quotient": 10,
        "description": "1~20 범위, 5 이하로 나누기"
    },
    3: {  # 어려움
        "name": "어려움 🌳",
        "max_dividend": 50,
        "max_divisor": 10,
        "max_quotient": 10,
        "description": "1~50 범위, 10 이하로 나누기"
    },
    4: {  # 도전
        "name": "도전! 🔥",
        "max_dividend": 100,
        "max_divisor": 12,
        "max_quotient": 12,
        "description": "1~100 범위, 구구단 활용"
    },
    5: {  # 혼합
        "name": "혼합 🎲",
        "max_dividend": 144,
        "max_divisor": 12,
        "max_quotient": 12,
        "description": "나누는 수를 직접 선택하여 섞어서 연습"
    }
}

def select_hybrid_divisors():
    """
    혼합 모드에서 나누는 수를 선택합니다.
    """
    print("\n" + "🎲" * 20)
    print("\n   🎲 나누는 수 선택 🎲\n")
    print("🎲" * 20)

    print("\n연습하고 싶은 나누는 수를 선택하세요!")
    print("(여러 개 선택 가능, 쉼표로 구분)\n")

    print("  2 - 2로 나누기 (2, 4, 6, 8, ...)")
    print("  3 - 3으로 나누기 (3, 6, 9, 12, ...)")
    print("  4 - 4로 나누기 (4, 8, 12, 16, ...)")
    print("  5 - 5로 나누기 (5, 10, 15, 20, ...)")
    print("  6 - 6으로 나누기 (6, 12, 18, 24, ...)")
    print("  7 - 7로 나누기 (7, 14, 21, 28, ...)")
    print("  8 - 8로 나누기 (8, 16, 24, 32, ...)")
    print("  9 - 9로 나누기 (9, 18, 27, 36, ...)")

    print("\n" + "-" * 40)
    print("💡 예시: 2,3,5 → 2, 3, 5로 나누는 문제가 섞여서 나와요!")
    print("💡 전체 선택: all 또는 a")
    print("-" * 40)

    while True:
        choice = input("\n나누는 수를 입력하세요: ").strip().lower()

        if choice in ['all', 'a', '전체']:
            selected = [2, 3, 4, 5, 6, 7, 8, 9]
            print(f"\n✅ 전체 선택: {selected}")
            return selected

        try:
            # 쉼표 또는 공백으로 구분
            parts = choice.replace(' ', ',').split(',')
            selected = []
            for p in parts:
                p = p.strip()
                if p:
                    num = int(p)
                    if 2 <= num <= 9:
                        if num not in selected:
                            selected.append(num)
                    else:
                        print(f"⚠️ {num}은(는) 2~9 범위가 아니에요. 건너뜁니다.")

            if selected:
                selected.sort()
                print(f"\n✅ 선택한 나누는 수: {selected}")
                confirm = input("이대로 진행할까요? (y/n): ").strip().lower()
                if confirm == 'y':
                    return selected
            else:
                print("❌ 최소 하나 이상 선택해주세요!")

        except ValueError:
            print("❌ 숫자를 입력해주세요! (예: 2,3,5)")

def select_difficulty():
    """
    난이도를 선택합니다.
    """
    print("\n" + "⭐" * 20)
    print("\n      🎯 난이도 선택 🎯\n")
    print("⭐" * 20)

    for level, settings in DIFFICULTY_SETTINGS.items():
        print(f"\n  {level}. {settings['name']}")
        print(f"     {settings['description']}")

    print("\n" + "-" * 40)

    while True:
        choice = input("\n난이도를 선택하세요 (1-5): ").strip()
        if choice in ['1', '2', '3', '4', '5']:
            level = int(choice)
            print(f"\n✅ '{DIFFICULTY_SETTINGS[level]['name']}' 난이도를 선택했어요!")
            return level
        else:
            print("❌ 1, 2, 3, 4, 5 중에서 선택해주세요!")

def get_division_problems(count=50, difficulty=2, divisors=None):
    """
    나머지가 없는 나눗셈 문제를 생성합니다.
    난이도에 따라 숫자 범위가 달라집니다.
    divisors: 혼합 모드에서 사용할 나누는 수 리스트 (예: [2, 3, 5])
    """
    settings = DIFFICULTY_SETTINGS.get(difficulty, DIFFICULTY_SETTINGS[2])

    max_dividend = settings["max_dividend"]
    max_divisor = settings["max_divisor"]
    max_quotient = settings["max_quotient"]

    problems = []

    # 가능한 모든 나눗셈 조합 찾기
    valid_combinations = []

    if divisors:
        # 혼합 모드: 선택한 나누는 수만 사용
        for divisor in divisors:
            for quotient in range(1, max_quotient + 1):
                dividend = divisor * quotient
                if dividend <= max_dividend:
                    valid_combinations.append((dividend, divisor, quotient))
    else:
        # 일반 모드
        for divisor in range(1, max_divisor + 1):
            for quotient in range(1, max_quotient + 1):
                dividend = divisor * quotient
                if dividend <= max_dividend:
                    valid_combinations.append((dividend, divisor, quotient))

    # 문제 생성
    for _ in range(count):
        problem = random.choice(valid_combinations)
        problems.append(problem)

    return problems

def display_division_visual(dividend, divisor, quotient, fruit):
    """
    나눗셈을 시각적으로 보여줍니다.
    과일을 그룹으로 나누어 표시합니다.
    """
    print(f"\n📚 문제: {dividend} ÷ {divisor} = ?\n")

    # 전체 과일 보여주기
    print(f"🧺 전체 {fruit} {dividend}개:")
    print(f"   {fruit * dividend}\n")

    # 나누기 설명
    print(f"👉 {dividend}개를 {divisor}명에게 똑같이 나눠주면?")
    print(f"   (한 사람당 몇 개씩 받을까요?)\n")

def show_answer_visual(dividend, divisor, quotient, fruit):
    """
    정답과 함께 시각적인 설명을 보여줍니다.
    """
    print(f"\n✨ 정답: {quotient}개 (또는 {quotient}묶음)\n")

    print(f"📦 나눠진 모습:")
    for i in range(divisor):
        group_label = f"  [{i+1}번]"
        print(f"{group_label} {fruit * quotient}")

    print(f"\n💡 설명: {fruit} {dividend}개를 {divisor}명에게 나눠주면")
    print(f"         한 사람당 {fruit} {quotient}개씩 받아요!")

def run_quiz(problems, difficulty=2, timer_level=0):
    """
    퀴즈를 실행합니다.
    """
    correct_count = 0
    wrong_problems = []
    timeout_count = 0

    difficulty_name = DIFFICULTY_SETTINGS.get(difficulty, DIFFICULTY_SETTINGS[2])["name"]
    timer_settings = TIMER_SETTINGS.get(timer_level, TIMER_SETTINGS[0])
    timer_name = timer_settings["name"]
    time_limit = timer_settings["seconds"]

    # 전체 시간 측정 시작
    total_start_time = time.time()

    print("\n" + "🌟" * 25)
    print("\n   🎓 나눗셈 연습을 시작합니다! 🎓")
    print(f"\n   📊 난이도: {difficulty_name}")
    print(f"   ⏱️ 시간: {timer_name}")
    print(f"   📝 총 {len(problems)}문제를 풀어볼 거예요.")
    print("   💡 힌트가 필요하면 'h'를 입력하세요.")
    print("   🚪 그만하고 싶으면 'q'를 입력하세요.")
    print("\n" + "🌟" * 25)

    input("\n준비되면 Enter를 눌러주세요! ")

    for i, (dividend, divisor, quotient) in enumerate(problems, 1):
        clear_screen()

        # 랜덤 과일 선택
        fruit = random.choice(FRUITS)

        print(f"📝 문제 {i}/{len(problems)} | 난이도: {difficulty_name} | ⏱️ {timer_name}")
        print(f"✅ 맞은 문제: {correct_count}개", end="")
        if timeout_count > 0:
            print(f" | ⏰ 시간초과: {timeout_count}개")
        else:
            print()

        # 문제 표시
        display_division_visual(dividend, divisor, quotient, fruit)

        # 타이머 생성
        timer = TimerInput(time_limit)

        # 답 입력 받기
        attempts = 0
        max_attempts = 3 if time_limit == 0 else 1  # 시간제한 있으면 1번만

        problem_solved = False
        while attempts < max_attempts and not problem_solved:
            if time_limit > 0:
                print(f"⏱️ 제한 시간: {time_limit}초")

            answer = timer.get_input("🤔 정답을 입력하세요: ")

            # 시간 초과
            if answer is None:
                print(f"\n⏰ 시간이 다 됐어요! 정답을 알려줄게요.")
                show_answer_visual(dividend, divisor, quotient, fruit)
                wrong_problems.append((dividend, divisor, quotient))
                timeout_count += 1
                break

            answer = answer.strip().lower()

            # 종료
            if answer == 'q':
                total_time = time.time() - total_start_time
                print("\n👋 수고했어요! 다음에 또 만나요!")
                print(f"⏱️ 총 소요 시간: {format_time(total_time)}")
                return correct_count, i - 1, wrong_problems, total_time

            # 힌트
            if answer == 'h':
                print(f"\n💡 힌트: {fruit} {divisor}개씩 묶어보세요!")
                print(f"         {fruit * divisor} ← 이게 1묶음이에요")
                continue

            # 숫자 확인
            try:
                user_answer = int(answer)
            except ValueError:
                print("❌ 숫자를 입력해주세요!")
                continue

            attempts += 1

            # 정답 확인
            if user_answer == quotient:
                print("\n🎉 정답이에요! 잘했어요! 🎉")
                correct_count += 1
                show_answer_visual(dividend, divisor, quotient, fruit)
                problem_solved = True
            else:
                remaining = max_attempts - attempts
                if remaining > 0:
                    print(f"\n😅 아쉬워요! 다시 한번 생각해봐요. (남은 기회: {remaining}번)")
                else:
                    print(f"\n😊 괜찮아요! 정답을 알려줄게요.")
                    show_answer_visual(dividend, divisor, quotient, fruit)
                    wrong_problems.append((dividend, divisor, quotient))

        if i < len(problems):
            input("\n다음 문제로 가려면 Enter를 눌러주세요! ")

    # 총 소요 시간
    total_time = time.time() - total_start_time

    return correct_count, len(problems), wrong_problems, total_time

def show_results(correct, total, wrong_problems, total_time=0, difficulty=2, timer_level=0):
    """
    최종 결과를 보여줍니다.
    틀린 문제가 있으면 다시 풀기 옵션을 제공합니다.
    """
    clear_screen()

    percentage = (correct / total * 100) if total > 0 else 0

    print("🏆" * 20)
    print("\n      📊 최종 결과 📊\n")
    print("🏆" * 20)

    print(f"\n   ✏️  푼 문제: {total}개")
    print(f"   ✅ 맞은 문제: {correct}개")
    print(f"   ❌ 틀린 문제: {total - correct}개")
    print(f"   ⏱️  소요 시간: {format_time(total_time)}")
    print(f"   📈 정답률: {percentage:.1f}%")

    # 격려 메시지
    print("\n" + "-" * 40)
    if percentage == 100:
        print("   🌟 완벽해요! 천재예요! 🌟")
    elif percentage >= 90:
        print("   🎉 아주 잘했어요! 대단해요! 🎉")
    elif percentage >= 70:
        print("   👍 잘했어요! 조금만 더 연습해요! 👍")
    elif percentage >= 50:
        print("   😊 괜찮아요! 계속 연습하면 잘할 수 있어요! 😊")
    else:
        print("   💪 힘내요! 연습하면 점점 나아질 거예요! 💪")
    print("-" * 40)

    # 틀린 문제 복습
    if wrong_problems:
        print("\n📖 틀린 문제 목록:")
        print("-" * 40)
        for dividend, divisor, quotient in wrong_problems:
            print(f"   {dividend} ÷ {divisor} = {quotient}")
        print("-" * 40)

    # 점수 저장 옵션
    print("\n" + "=" * 40)
    save_choice = input("💾 점수를 저장할까요? (y/n): ").strip().lower()

    if save_choice == 'y':
        name = input("📝 이름을 입력하세요: ").strip()
        if not name:
            name = "익명"
        saved = save_score(name, correct, total, percentage)
        print(f"\n✅ 저장되었습니다!")
        print(f"   {saved['name']} - {saved['date']} - {saved['percentage']}%")

        # 뱃지 확인 및 수여
        new_badges = check_and_award_badges(correct, total, percentage, difficulty, timer_level)
        if new_badges:
            show_new_badges(new_badges)

    # 틀린 문제 다시 풀기 옵션
    if wrong_problems:
        print("\n" + "=" * 40)
        retry_choice = input("🔄 틀린 문제를 다시 풀어볼까요? (y/n): ").strip().lower()

        if retry_choice == 'y':
            return retry_wrong_problems(wrong_problems)

    return None

def retry_wrong_problems(wrong_problems):
    """
    틀린 문제들만 다시 풀기
    """
    clear_screen()
    print("🔄" * 20)
    print(f"\n   📝 틀린 문제 다시 풀기 ({len(wrong_problems)}문제)\n")
    print("🔄" * 20)
    print("\n   💪 이번엔 꼭 맞춰봐요!")
    input("\n준비되면 Enter를 눌러주세요! ")

    correct_count = 0
    still_wrong = []

    for i, (dividend, divisor, quotient) in enumerate(wrong_problems, 1):
        clear_screen()

        fruit = random.choice(FRUITS)

        print(f"📝 다시 풀기 {i}/{len(wrong_problems)}")
        print(f"✅ 맞은 문제: {correct_count}개")

        # 문제 표시
        display_division_visual(dividend, divisor, quotient, fruit)

        # 답 입력 받기
        attempts = 0
        max_attempts = 3

        while attempts < max_attempts:
            answer = input(f"🤔 정답을 입력하세요: ").strip().lower()

            if answer == 'q':
                print("\n👋 수고했어요!")
                return still_wrong if still_wrong else None

            if answer == 'h':
                print(f"\n💡 힌트: {fruit} {divisor}개씩 묶어보세요!")
                print(f"         {fruit * divisor} ← 이게 1묶음이에요")
                continue

            try:
                user_answer = int(answer)
            except ValueError:
                print("❌ 숫자를 입력해주세요!")
                continue

            attempts += 1

            if user_answer == quotient:
                print("\n🎉 정답이에요! 이제 알겠죠? 🎉")
                correct_count += 1
                show_answer_visual(dividend, divisor, quotient, fruit)
                break
            else:
                remaining = max_attempts - attempts
                if remaining > 0:
                    print(f"\n😅 아쉬워요! 다시 한번 생각해봐요. (남은 기회: {remaining}번)")
                else:
                    print(f"\n😊 괜찮아요! 정답을 알려줄게요.")
                    show_answer_visual(dividend, divisor, quotient, fruit)
                    still_wrong.append((dividend, divisor, quotient))

        if i < len(wrong_problems):
            input("\n다음 문제로 가려면 Enter를 눌러주세요! ")

    # 다시 풀기 결과
    clear_screen()
    print("🔄" * 20)
    print("\n   📊 다시 풀기 결과 📊\n")
    print("🔄" * 20)

    retry_total = len(wrong_problems)
    retry_percentage = (correct_count / retry_total * 100) if retry_total > 0 else 0

    print(f"\n   ✏️  다시 푼 문제: {retry_total}개")
    print(f"   ✅ 맞은 문제: {correct_count}개")
    print(f"   ❌ 또 틀린 문제: {len(still_wrong)}개")
    print(f"   📈 정답률: {retry_percentage:.1f}%")

    if len(still_wrong) == 0:
        print("\n   🌟 모두 맞췄어요! 대단해요! 🌟")
    elif correct_count > 0:
        print("\n   👍 잘하고 있어요! 조금만 더 연습해요!")

    # 또 틀린 문제가 있으면 다시 풀기 옵션
    if still_wrong:
        print("\n📖 아직 틀린 문제:")
        print("-" * 40)
        for dividend, divisor, quotient in still_wrong:
            print(f"   {dividend} ÷ {divisor} = {quotient}")
        print("-" * 40)

        retry_again = input("\n🔄 또 틀린 문제를 다시 풀어볼까요? (y/n): ").strip().lower()
        if retry_again == 'y':
            return retry_wrong_problems(still_wrong)

    return still_wrong if still_wrong else None

def main():
    """
    메인 프로그램
    """
    print("\n" + "🍎" * 25)
    print("\n  🎈 초등학생 나눗셈 연습 프로그램 🎈")
    print("\n" + "🍎" * 25)

    print("""
    안녕하세요! 👋

    이 프로그램은 나눗셈을 쉽게 배울 수 있도록
    과일 그림으로 보여줄 거예요!

    예를 들어, 6 ÷ 2 = ?

    🍎🍎🍎🍎🍎🍎  (사과 6개를)

    [1번] 🍎🍎🍎  (2명에게 나눠주면)
    [2번] 🍎🍎🍎  (한 사람당 3개씩!)

    정답: 3
    """)

    while True:
        # 오늘의 도전 상태 확인
        daily_status = "✅" if check_daily_completed() else "🆕"

        print("\n" + "=" * 40)
        print("메뉴를 선택하세요:")
        print(f"  1. 📅 오늘의 도전 {daily_status}")
        print("  2. 🎯 연습 시작 (50문제)")
        print("  3. ⚡ 짧은 연습 (10문제)")
        print("  4. 💪 긴 연습 (100문제)")
        print("  5. 📊 점수 기록 보기")
        print("  6. 📈 성장 그래프 보기")
        print("  7. 🏅 뱃지 컬렉션 보기")
        print("  8. 종료")
        print("=" * 40)

        choice = input("\n선택 (1-8): ").strip()

        if choice == '1':
            # 오늘의 도전
            run_daily_challenge()
            input("\n메뉴로 돌아가려면 Enter를 누르세요...")

        elif choice in ['2', '3', '4']:
            # 난이도 선택
            difficulty = select_difficulty()

            # 혼합 모드일 경우 나누는 수 선택
            divisors = None
            if difficulty == 5:
                divisors = select_hybrid_divisors()

            # 타이머 선택
            timer_level = select_timer()

            # 문제 수 결정
            if choice == '2':
                count = 50
            elif choice == '3':
                count = 10
            else:
                count = 100

            problems = get_division_problems(count, difficulty, divisors)
            correct, total, wrong, elapsed_time = run_quiz(problems, difficulty, timer_level)
            show_results(correct, total, wrong, elapsed_time, difficulty, timer_level)

        elif choice == '5':
            show_score_history()
            input("\n메뉴로 돌아가려면 Enter를 누르세요...")
        elif choice == '6':
            show_growth_graph()
            input("\n메뉴로 돌아가려면 Enter를 누르세요...")
        elif choice == '7':
            show_all_badges()
            input("\n메뉴로 돌아가려면 Enter를 누르세요...")
        elif choice == '8':
            print("\n👋 안녕히 가세요! 다음에 또 만나요! 🎈\n")
            break
        else:
            print("❌ 1, 2, 3, 4, 5, 6, 7, 8 중에서 선택해주세요!")

        if choice in ['2', '3', '4']:
            again = input("\n다시 하시겠어요? (y/n): ").strip().lower()
            if again != 'y':
                print("\n👋 안녕히 가세요! 다음에 또 만나요! 🎈\n")
                break

if __name__ == "__main__":
    main()
