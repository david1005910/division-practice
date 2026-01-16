# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

초등학생을 위한 나눗셈/곱셈 연습 프로그램입니다. 과일 이모지를 사용하여 연산 개념을 시각적으로 이해할 수 있도록 도와줍니다.

## 실행 방법

```bash
# CLI 버전 (터미널)
python3 division_practice.py

# GUI 버전 (tkinter 윈도우)
python3 division_gui.py
```

## 아키텍처

두 개의 독립적인 Python 프로그램으로 구성되며, 외부 의존성 없이 표준 라이브러리만 사용합니다.

### 프로그램 구성
- `division_practice.py`: CLI 버전 (전체 기능 포함)
- `division_gui.py`: GUI 버전 (tkinter 사용, 나눗셈/곱셈 모드 지원)

### 데이터 저장소
- `scores.json`: CLI 버전 점수 기록 (최근 100개 유지)
- `gui_scores.json`: GUI 버전 점수 기록 (최근 100개 유지)
- `badges.json`: 획득한 뱃지 목록 및 히스토리 (CLI 전용)
- `daily.json`: 일일 도전 기록 (CLI 전용)

### CLI 버전 주요 시스템
1. **문제 생성**: 난이도별 설정(`DIFFICULTY_SETTINGS`)에 따라 나머지 없는 나눗셈 조합 생성
2. **타이머**: `TimerInput` 클래스로 시간 제한 입력 처리 (스레드 사용)
3. **뱃지 시스템**: 23개의 업적 뱃지(`BADGES`), 조건 충족 시 자동 수여
4. **일일 도전**: 날짜 기반 시드로 매일 동일 문제 출제, 연속 도전 추적
5. **혼합 모드**: 나누는 수 직접 선택하여 섞어서 연습 (난이도 5)

### GUI 버전 주요 시스템
- `DivisionPracticeGUI` 클래스: 메인 GUI 애플리케이션
- 나눗셈/곱셈 모드 선택 가능
- 분수 형태 시각화 (나눗셈 모드)
- macOS/Windows 효과음 지원 (`play_sound()`)

### 난이도 설정 (CLI - DIFFICULTY_SETTINGS)
- 1(쉬움): 1~10 범위, 5 이하로 나누기
- 2(보통): 1~20 범위, 5 이하로 나누기
- 3(어려움): 1~50 범위, 10 이하로 나누기
- 4(도전): 1~100 범위, 12 이하로 나누기
- 5(혼합): 나누는 수 직접 선택 (2~9)

## CLI 핵심 함수

- `get_division_problems(count, difficulty, divisors)`: 난이도별 나머지 없는 나눗셈 문제 생성
- `display_division_visual()`: 문제를 과일 그림으로 시각화
- `show_answer_visual()`: 정답을 그룹으로 나눠서 시각화
- `run_quiz()`: 퀴즈 진행 (힌트 'h', 종료 'q' 지원)
- `check_and_award_badges()`: 조건 확인 후 뱃지 수여
- `run_daily_challenge()`: 일일 도전 실행 (하루 1회 제한)
- `select_hybrid_divisors()`: 혼합 모드 나누는 수 선택
