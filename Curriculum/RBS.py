#!/usr/bin/env python3
"""
RPS.py  —  Recommended Problem Selector
────────────────────────────────────────
실행 방법:
  python RPS.py

흐름:
  1. 기업 선택
  2. 직업 선택
  3. 문제 10개 추천 (못하는 문제 우선)
  4. 문제를 하나씩 풀기 (선택)
     - 문제/선택지 표시 → 답 입력 → 정오답 판정 → 해설 표시
  5. 학습 현황 확인 (선택)
  6. 반복 또는 종료

환경 변수 (선택):
  PROBLEMS_DIR   : JobProblems/  위치  (기본: ./JobProblems)
  PATHS_DIR      : LearningPaths/ 위치 (기본: ./LearningPaths)
  USER_STATE_DIR : 사용자 상태 저장 위치 (기본: ./user_states)
"""

import os
import sys

# ── 경로 설정 ──────────────────────────────────────────────────
SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
PROBLEMS_DIR   = os.environ.get("PROBLEMS_DIR",
                                os.path.join(SCRIPT_DIR, "JobProblems"))
PATHS_DIR      = os.environ.get("PATHS_DIR",
                                os.path.join(SCRIPT_DIR, "LearningPaths"))
USER_STATE_DIR = os.environ.get("USER_STATE_DIR",
                                os.path.join(SCRIPT_DIR, "user_states"))

sys.path.insert(0, SCRIPT_DIR)
from Job_problem_recommender import JobProblemRecommender


# ── UI 유틸 ────────────────────────────────────────────────────

def clear():
    os.system("cls" if os.name == "nt" else "clear")

def hr(ch="─", w=64):
    print(ch * w)

def banner():
    hr("═")
    print("  📚  RPS — Recommended Problem Selector")
    print("  기업 공고 기반 · 약점 우선 문제 추천 시스템")
    hr("═")
    print()

def ask_yn(prompt: str) -> bool:
    while True:
        ans = input(f"{prompt} [y/n]: ").strip().lower()
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        print("  y 또는 n 을 입력해주세요.")

def pick_from_list(items: list[str], label: str) -> str:
    print(f"\n  [ {label} ]")
    for i, item in enumerate(items, 1):
        print(f"    {i:>2}. {item}")
    print()
    while True:
        raw = input(f"  선택 (번호 또는 이름): ").strip()
        if not raw:
            continue
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(items):
                return items[idx]
            print(f"  1 ~ {len(items)} 사이의 번호를 입력해주세요.")
            continue
        exact = [x for x in items if x.lower() == raw.lower()]
        if exact:
            return exact[0]
        partial = [x for x in items if raw.lower() in x.lower()]
        if len(partial) == 1:
            return partial[0]
        if len(partial) > 1:
            print(f"  여러 항목이 일치합니다: {', '.join(partial)}")
            continue
        print(f"  '{raw}'을(를) 찾을 수 없습니다.")


# ── 추천 목록 출력 ─────────────────────────────────────────────

def display_recommendations(recs: list[dict], company: str, job: str) -> None:
    print()
    hr()
    print(f"  🎯  [{company}] {job}  —  추천 문제 {len(recs)}개")
    hr()
    for r in recs:
        skills = ", ".join(r["skills_required"][:3])
        if len(r["skills_required"]) > 3:
            skills += f" 외 {len(r['skills_required']) - 3}개"
        print(
            f"\n  [{r['rank']:02d}]  ID {r['question_id']:>4}"
            f"  │  {r['difficulty']}"
            f"  │  {r['question_type']}"
        )
        print(f"        {r['category']} > {r['subcategory']}")
        if skills:
            print(f"        스킬: {skills}")
        print(f"        → {r['reason']}")
    hr()


# ── 문제 풀기 ─────────────────────────────────────────────────

CHOICE_LABELS = ["①", "②", "③", "④"]

def solve_problem(
    rec: JobProblemRecommender,
    r:   dict,
    job: str,
    user_id: str,
) -> bool:
    """
    문제 하나를 출력하고 사용자 입력을 받아 정오답을 판정한다.
    BKT 업데이트 후 정답 여부를 반환한다.
    """
    print()
    hr("─")
    print(f"  📝  문제 {r['rank']:02d}  │  ID {r['question_id']}  │  {r['difficulty']}  │  {r['question_type']}")
    print(f"  카테고리: {r['category']} > {r['subcategory']}")
    if r["scenario"]:
        print(f"\n  [상황]  {r['scenario']}")
    print(f"\n  Q. {r['question']}\n")

    choices = r["choices"]
    for i, ch in enumerate(choices):
        label = CHOICE_LABELS[i] if i < len(CHOICE_LABELS) else f"{i+1}."
        print(f"     {label}  {ch}")

    print()
    # 입력 받기
    while True:
        raw = input("  답 입력 (번호 또는 텍스트, Enter=건너뜀): ").strip()

        # 건너뜀
        if raw == "":
            print("  ⏭  건너뜀")
            return False

        # 번호 입력
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(choices):
                user_answer = choices[idx]
                break
            print(f"  1 ~ {len(choices)} 사이의 번호를 입력해주세요.")
            continue

        # 텍스트 입력 — 선택지와 대소문자 무시 비교
        matched = [ch for ch in choices if ch.lower() == raw.lower()]
        if matched:
            user_answer = matched[0]
            break

        # 부분 일치
        partial = [ch for ch in choices if raw.lower() in ch.lower()]
        if len(partial) == 1:
            user_answer = partial[0]
            break
        if len(partial) > 1:
            print(f"  여러 선택지가 일치합니다: {', '.join(partial)}")
            continue

        print("  선택지에 없는 답입니다. 번호로 입력해주세요.")

    correct_answer = r["correct_answer"]
    is_correct     = (user_answer == correct_answer)

    if is_correct:
        print(f"\n  ✅  정답!  →  {correct_answer}")
    else:
        print(f"\n  ❌  오답.  내가 선택: {user_answer}  │  정답: {correct_answer}")

    if r["explanation"]:
        print(f"\n  💡  해설: {r['explanation']}")

    # BKT 업데이트
    mastery = rec.record_answer(user_id, job, r["question_id"], is_correct)
    print(f"\n  📈  [{r['question_id']}번 문제] 숙련도 업데이트 → {mastery * 100:.1f}%")

    return is_correct


def solve_flow(
    rec: JobProblemRecommender,
    recs: list[dict],
    job: str,
    user_id: str,
) -> None:
    """추천 문제를 하나씩 순서대로 푼다."""
    print()
    if not ask_yn("  추천된 문제를 지금 풀어보시겠습니까?"):
        return

    correct_count = 0
    for r in recs:
        if solve_problem(rec, r, job, user_id):
            correct_count += 1
        print()
        if r["rank"] < len(recs):
            if not ask_yn("  다음 문제로 넘어가시겠습니까?"):
                break

    print()
    hr()
    solved = sum(1 for r in recs if rec.get_job_stats(user_id, job)["total_attempted"] > 0)
    print(f"  🏁  풀이 완료  —  정답 {correct_count} / {len(recs)}문제")
    hr()


# ── 통계 출력 ─────────────────────────────────────────────────

def show_stats_flow(rec: JobProblemRecommender, user_id: str, job: str) -> None:
    stats = rec.get_job_stats(user_id, job)
    print()
    hr()
    print(f"  📊  [{user_id}]  {job}  학습 현황")
    hr()
    if stats["total_attempted"] == 0:
        print("  아직 풀이 기록이 없습니다.")
    else:
        print(f"  총 시도 문제 수  : {stats['total_attempted']}개")
        print(f"  평균 숙련도      : {stats['avg_mastery_pct']}%")
        if stats["weakest_ids"]:
            ids = ", ".join(str(q) for q in stats["weakest_ids"])
            print(f"  가장 약한 문제들 : ID {ids}")
    hr()

    if ask_yn("  전체 직업별 통계도 보시겠습니까?"):
        all_stats = rec.get_all_stats(user_id)
        if not all_stats:
            print("  다른 직업의 기록이 없습니다.")
            return
        print()
        hr()
        print(f"  📊  [{user_id}]  전체 학습 현황")
        hr()
        for j, s in all_stats.items():
            print(
                f"  {j:<45}"
                f"  시도 {s['total_attempted']:>3}개  "
                f"  평균 숙련도 {s['avg_mastery_pct']:>5}%"
            )
        hr()


# ── 메인 ───────────────────────────────────────────────────────

def main():
    clear()
    banner()

    rec = JobProblemRecommender(
        problems_dir=PROBLEMS_DIR,
        paths_dir=PATHS_DIR,
        user_state_dir=USER_STATE_DIR,
        top_n=10,
    )

    uid_input = input("  사용자 ID를 입력하세요 (Enter = 'default'): ").strip()
    user_id   = uid_input if uid_input else "default"
    print(f"  → 사용자 ID: {user_id}\n")

    while True:
        hr()

        # ① 기업 선택
        company = pick_from_list(rec.list_companies(), "기업 선택")
        print(f"\n  ✅ 선택된 기업: {company}")

        # ② 직업 선택
        try:
            jobs = rec.list_jobs(company)
        except ValueError as e:
            print(f"\n  ⚠️  {e}")
            continue

        if not jobs:
            print(f"\n  ⚠️  '{company}'에 사용 가능한 문제 파일이 없습니다.")
            print(      "      JobProblems/ 폴더에 파일을 추가해주세요.")
            if ask_yn("\n  다른 기업을 선택하시겠습니까?"):
                continue
            break

        job = pick_from_list(jobs, "직업 선택")
        print(f"\n  ✅ 선택된 직업: {job}")

        # ③ 문제 추천
        print("\n  🔍 문제를 추천하는 중...")
        try:
            recs = rec.recommend(company=company, job=job, user_id=user_id)
        except (ValueError, FileNotFoundError) as e:
            print(f"\n  ⚠️  {e}")
            if ask_yn("\n  다시 선택하시겠습니까?"):
                continue
            break

        display_recommendations(recs, company, job)

        # ④ 문제 풀기
        solve_flow(rec, recs, job, user_id)

        # ⑤ 통계
        if ask_yn("\n  학습 현황을 보시겠습니까?"):
            show_stats_flow(rec, user_id, job)

        print()
        if not ask_yn("  계속하시겠습니까?"):
            break

    print("\n  👋 종료합니다. 열심히 공부하세요!\n")


if __name__ == "__main__":
    main()