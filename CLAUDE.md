# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

초등학생을 위한 나눗셈 연습 프로그램입니다. 과일 이모지를 사용하여 나눗셈 개념을 시각적으로 이해할 수 있도록 도와줍니다.

## 실행 방법

```bash
python3 division_practice.py
```

## 아키텍처

단일 파일(`division_practice.py`) Python 프로그램으로, 외부 의존성 없이 표준 라이브러리만 사용합니다.

### 데이터 저장소
- `scores.json`: 연습 점수 기록 (최근 100개 유지)
- `badges.json`: 획득한 뱃지 목록 및 히스토리
- `daily.json`: 일일 도전 기록 (연속 도전 추적)

### 주요 시스템
1. **문제 생성**: 난이도별 설정(`DIFFICULTY_SETTINGS`)에 따라 나머지 없는 나눗셈 조합 생성
2. **타이머**: `TimerInput` 클래스로 시간 제한 입력 처리 (스레드 사용)
3. **뱃지 시스템**: 20+개의 업적 뱃지(`BADGES`), 조건 충족 시 자동 수여
4. **일일 도전**: 날짜 기반 시드로 매일 동일 문제 출제, 연속 도전 추적

### 난이도 설정 (DIFFICULTY_SETTINGS)
- 1(쉬움): 1~10 범위, 5 이하로 나누기
- 2(보통): 1~20 범위, 5 이하로 나누기
- 3(어려움): 1~50 범위, 10 이하로 나누기
- 4(도전): 1~100 범위, 12 이하로 나누기

## 핵심 함수

- `get_division_problems(count, difficulty)`: 난이도별 나머지 없는 나눗셈 문제 생성
- `display_division_visual()`: 문제를 과일 그림으로 시각화
- `show_answer_visual()`: 정답을 그룹으로 나눠서 시각화
- `run_quiz()`: 퀴즈 진행 (힌트 'h', 종료 'q' 지원)
- `check_and_award_badges()`: 조건 확인 후 뱃지 수여
- `run_daily_challenge()`: 일일 도전 실행 (하루 1회 제한)
