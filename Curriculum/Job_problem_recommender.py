"""
job_problem_recommender.py
──────────────────────────
기업 공고 기반 직업 문제 추천 모델 (LLM 미사용)

[ 폴더 구조 ]
  LearningPaths/  : {직업명}_path.json   — 문제 의존성 그래프 + 가중치
  JobProblems/    : {직업명}.json        — 실제 문제 내용 (question, choices, answer)
  두 파일은 question_id 로 연결된다.

[ 파일명 규칙 ]
  직업명의 공백 → '_', '/' → '_', '(QA)' 등 특수문자 제거
  예) "AI Engineer"              → AI_Engineer.json / AI_Engineer_path.json
      "UI/UX Engineer"           → UI_UX_Engineer.json / UI_UX_Engineer_path.json
      "Quality Assurance Engineer (QA)" → Quality_Assurance_Engineer_QA.json / ...

[ 추천 알고리즘 ]
  BKT (Bayesian Knowledge Tracing) + edge combined_score 가중치
  약점 점수 = (1 - mastery) + UCB 탐색 보너스 + 선행 중요도 보정
  → 높을수록 우선 추천 (못하거나 안 풀어본 문제)

[ 레벨 매핑: 구인 데이터 → 문제 난이도 ]
  Junior    → junior_level
  Mid-level → middle_level
  Senior    → senior_level
  Staff     → senior_level  (문제 파일에 Staff 없음)
"""

from __future__ import annotations

import json
import math
import os
from typing import Optional


# ──────────────────────────────────────────────
# 상수
# ──────────────────────────────────────────────

ALL_JOBS: list[str] = [
    "AI Engineer",
    "AR/VR Engineer",
    "Backend Engineer",
    "Big Data Engineer",
    "Blockchain Engineer",
    "Cloud Infrastructure Engineer",
    "Computer Vision Engineer",
    "Data Engineer",
    "Data Scientist",
    "Database Administrator",
    "DevOps Engineer",
    "Embedded Systems Engineer",
    "Frontend Developer",
    "Full Stack Engineer",
    "Game Developer",
    "IoT Engineer",
    "Machine Learning Researcher",
    "Mobile App Developer",
    "Natural Language Processing Engineer",
    "Network Engineer",
    "Quality Assurance Engineer (QA)",
    "Research Scientist",
    "Robotics Engineer",
    "Security Engineer",
    "Site Reliability Engineer (SRE)",
    "Software Architect",
    "Software Engineer",
    "Systems Engineer",
    "Technical Program Manager",
    "UI/UX Engineer",
]

ALL_COMPANIES: list[str] = [
    "Amazon", "Apple", "Google", "Meta", "Microsoft",
    "Netflix", "Nvidia", "OpenAI", "Perplexity", "SpaceX", "Tesla",
]

DIFFICULTY_KOR: dict[str, str] = {
    "university_level": "대학 기초",
    "junior_level":     "주니어",
    "middle_level":     "미들",
    "senior_level":     "시니어",
}

QUESTION_TYPE_KOR: dict[str, str] = {
    "definition":    "개념/정의",
    "application":   "응용",
    "analysis":      "분석",
    "optimization":  "최적화",
    "debugging":     "디버깅",
    "design":        "설계",
    "implementation":"구현",
}


# ──────────────────────────────────────────────
# 파일명 변환 유틸
# ──────────────────────────────────────────────

def job_to_filename(job: str, suffix: str = "") -> str:
    """
    직업명 → 파일명 변환.
    suffix 예시: "" → "AI_Engineer.json"
                "_path" → "AI_Engineer_path.json"
    """
    name = (job
            .replace("/", "_")
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("-", "_"))
    # 연속 언더스코어 정리
    while "__" in name:
        name = name.replace("__", "_")
    return f"{name}{suffix}.json"


# ──────────────────────────────────────────────
# BKT (Bayesian Knowledge Tracing)
# ──────────────────────────────────────────────

class BKTModel:
    """
    경량 Bayesian Knowledge Tracing.

    각 question_id마다 mastery(숙련도 0~1)를 추적한다.
    약점 점수 = (1 - mastery) + UCB 탐색 보너스
    → 높을수록 우선 추천
    """

    def __init__(
        self,
        p_init:  float = 0.3,
        p_learn: float = 0.2,
        p_slip:  float = 0.1,
        p_guess: float = 0.2,
    ):
        self.p_init  = p_init
        self.p_learn = p_learn
        self.p_slip  = p_slip
        self.p_guess = p_guess
        self._mastery:  dict[int, float] = {}
        self._attempts: dict[int, int]   = {}

    def mastery(self, qid: int) -> float:
        return self._mastery.get(qid, self.p_init)

    def attempts(self, qid: int) -> int:
        return self._attempts.get(qid, 0)

    def weakness_score(self, qid: int) -> float:
        m = self.mastery(qid)
        a = self.attempts(qid)
        exploration = 0.3 / math.sqrt(1 + a)
        return (1.0 - m) + exploration

    def update(self, qid: int, correct: bool) -> float:
        p = self.mastery(qid)
        if correct:
            p_obs       = p * (1 - self.p_slip) + (1 - p) * self.p_guess
            p_posterior = p * (1 - self.p_slip) / max(p_obs, 1e-9)
        else:
            p_obs       = p * self.p_slip + (1 - p) * (1 - self.p_guess)
            p_posterior = p * self.p_slip / max(p_obs, 1e-9)
        p_new = p_posterior + (1 - p_posterior) * self.p_learn
        self._mastery[qid]  = min(max(p_new, 0.0), 1.0)
        self._attempts[qid] = self._attempts.get(qid, 0) + 1
        return self._mastery[qid]

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({
                "p_init": self.p_init, "p_learn": self.p_learn,
                "p_slip": self.p_slip, "p_guess": self.p_guess,
                "mastery": self._mastery, "attempts": self._attempts,
            }, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "BKTModel":
        with open(filepath, encoding="utf-8") as f:
            s = json.load(f)
        m = cls(s["p_init"], s["p_learn"], s["p_slip"], s["p_guess"])
        m._mastery  = {int(k): v for k, v in s.get("mastery",  {}).items()}
        m._attempts = {int(k): v for k, v in s.get("attempts", {}).items()}
        return m

    def stats(self) -> dict:
        total = len(self._attempts)
        if total == 0:
            return {"total_attempted": 0, "avg_mastery_pct": None, "weakest_ids": []}
        avg     = sum(self._mastery.values()) / len(self._mastery) if self._mastery else 0
        weakest = sorted(self._mastery.items(), key=lambda x: x[1])[:5]
        return {
            "total_attempted": total,
            "avg_mastery_pct": round(avg * 100, 1),
            "weakest_ids":     [qid for qid, _ in weakest],
        }


# ──────────────────────────────────────────────
# 추천 엔진
# ──────────────────────────────────────────────

class JobProblemRecommender:
    """
    기업 선택 → 직업 선택 → BKT 약점 기반 문제 추천

    Parameters
    ----------
    problems_dir   : JobProblems/  — 실제 문제 JSON 파일 위치
    paths_dir      : LearningPaths/ — 의존성 그래프 JSON 파일 위치
    user_state_dir : 사용자 BKT 상태 저장 위치
    top_n          : 추천 문제 수 (기본 10)
    """

    def __init__(
        self,
        problems_dir:   str = "./JobProblems",
        paths_dir:      str = "./LearningPaths",
        user_state_dir: str = "./user_states",
        top_n:          int = 10,
    ):
        self.problems_dir   = problems_dir
        self.paths_dir      = paths_dir
        self.user_state_dir = user_state_dir
        self.top_n          = top_n
        os.makedirs(user_state_dir, exist_ok=True)

    # ── 기업 / 직업 목록 ──────────────────────

    def list_companies(self) -> list[str]:
        return list(ALL_COMPANIES)

    def list_jobs(self, company: str) -> list[str]:
        """
        기업의 직업 목록.
        데이터셋상 모든 기업이 동일한 30개 직업을 가지므로
        JobProblems/ 에 파일이 실제로 존재하는 직업만 반환한다.
        """
        if company not in ALL_COMPANIES:
            raise ValueError(
                f"'{company}'은(는) 지원하지 않는 기업입니다.\n"
                f"지원 기업: {', '.join(ALL_COMPANIES)}"
            )
        return [
            job for job in ALL_JOBS
            if os.path.exists(
                os.path.join(self.problems_dir, job_to_filename(job))
            )
        ]

    # ── 파일 로드 ─────────────────────────────

    def _load_problems(self, job: str) -> dict[int, dict]:
        """JobProblems/{job}.json 로드 → {question_id: 문제데이터} 딕셔너리 반환."""
        filepath = os.path.join(self.problems_dir, job_to_filename(job))
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"문제 파일 없음: {filepath}\n"
                f"JobProblems/ 폴더에 '{job_to_filename(job)}' 파일을 추가해주세요."
            )
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        return {item["question_id"]: item for item in data}

    def _load_path(self, job: str) -> dict:
        """LearningPaths/{job}_path.json 로드."""
        filepath = os.path.join(self.paths_dir, job_to_filename(job, "_path"))
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"경로 파일 없음: {filepath}\n"
                f"LearningPaths/ 폴더에 '{job_to_filename(job, '_path')}' 파일을 추가해주세요."
            )
        with open(filepath, encoding="utf-8") as f:
            return json.load(f)

    # ── BKT 상태 관리 ─────────────────────────

    def _bkt_path(self, user_id: str, job: str) -> str:
        safe = job_to_filename(job).replace(".json", "")
        return os.path.join(self.user_state_dir, f"{user_id}__{safe}.json")

    def _get_bkt(self, user_id: str, job: str) -> BKTModel:
        p = self._bkt_path(user_id, job)
        return BKTModel.load(p) if os.path.exists(p) else BKTModel()

    def _save_bkt(self, bkt: BKTModel, user_id: str, job: str) -> None:
        bkt.save(self._bkt_path(user_id, job))

    # ── 핵심: 문제 추천 ───────────────────────

    def recommend(
        self,
        company:  str,
        job:      str,
        user_id:  str = "default",
        top_n:    Optional[int] = None,
    ) -> list[dict]:
        """
        기업 + 직업을 받아 추천 문제 리스트를 반환한다.

        반환 형식:
        [
          {
            "rank":                  1,
            "question_id":           42,
            "difficulty":            "미들",
            "question_type":         "응용",
            "category":              "Machine Learning",
            "subcategory":           "Optimization",
            "scenario":              "...",
            "question":              "...",
            "choices":               ["A", "B", "C", "D"],
            "correct_answer":        "B",
            "explanation":           "...",
            "skills_required":       [...],
            "estimated_mastery_pct": 12.5,
            "reason":                "숙련도 낮음 — 핵심 보완 필요"
          },
          ...
        ]
        """
        n = top_n or self.top_n

        available = self.list_jobs(company)
        if job not in available:
            raise ValueError(
                f"'{company}'의 직업 목록에 '{job}'이(가) 없거나 "
                f"문제 파일이 존재하지 않습니다.\n"
                f"선택 가능: {', '.join(available)}"
            )

        problems = self._load_problems(job)
        path_data = self._load_path(job)
        edges     = path_data["dependency_graph"]["edges"]
        bkt       = self._get_bkt(user_id, job)

        # edge combined_score 집계: 많은 문제가 의존하는 문제 = 핵심 기반
        edge_weight: dict[int, float] = {}
        for edge in edges:
            tid   = edge["Target_ID"]
            score = edge.get("combined_score", 0.0)
            edge_weight[tid] = edge_weight.get(tid, 0.0) + score

        # 최종 점수 계산
        candidates = []
        for qid, prob in problems.items():
            weakness   = bkt.weakness_score(qid)
            centrality = min(edge_weight.get(qid, 0.0) * 0.1, 0.2)
            candidates.append({
                "question_id": qid,
                "final_score": weakness + centrality,
                "prob":        prob,
            })

        candidates.sort(key=lambda x: -x["final_score"])
        top = candidates[:n]

        results = []
        for rank, c in enumerate(top, 1):
            prob        = c["prob"]
            mastery_pct = round(bkt.mastery(c["question_id"]) * 100, 1)
            results.append({
                "rank":                  rank,
                "question_id":           c["question_id"],
                "difficulty":            DIFFICULTY_KOR.get(prob.get("difficulty", ""), prob.get("difficulty", "")),
                "question_type":         QUESTION_TYPE_KOR.get(prob.get("question_type", ""), prob.get("question_type", "")),
                "category":              prob.get("category", ""),
                "subcategory":           prob.get("subcategory", ""),
                "scenario":              prob.get("scenario", ""),
                "question":              prob.get("question", ""),
                "choices":               prob.get("choices", []),
                "correct_answer":        prob.get("correct_answer", ""),
                "explanation":           prob.get("explanation", ""),
                "skills_required":       prob.get("skills_required", []),
                "estimated_mastery_pct": mastery_pct,
                "reason":                _make_reason(mastery_pct, bkt.attempts(c["question_id"])),
            })
        return results

    # ── 답변 기록 ─────────────────────────────

    def record_answer(
        self,
        user_id:     str,
        job:         str,
        question_id: int,
        correct:     bool,
    ) -> float:
        """풀이 결과 기록 → BKT 업데이트 → 저장. 반환: 업데이트된 mastery."""
        bkt     = self._get_bkt(user_id, job)
        mastery = bkt.update(question_id, correct)
        self._save_bkt(bkt, user_id, job)
        return mastery

    # ── 통계 ─────────────────────────────────

    def get_job_stats(self, user_id: str, job: str) -> dict:
        return self._get_bkt(user_id, job).stats()

    def get_all_stats(self, user_id: str) -> dict[str, dict]:
        result = {}
        for job in ALL_JOBS:
            p = self._bkt_path(user_id, job)
            if os.path.exists(p):
                s = BKTModel.load(p).stats()
                if s["total_attempted"] > 0:
                    result[job] = s
        return result


# ──────────────────────────────────────────────
# 헬퍼
# ──────────────────────────────────────────────

def _make_reason(mastery_pct: float, attempts: int) -> str:
    if attempts == 0:
        return "미풀이 문제 — 새롭게 도전"
    if mastery_pct < 35:
        return f"숙련도 낮음 ({mastery_pct}%) — 핵심 보완 필요"
    if mastery_pct < 65:
        return f"숙련도 중간 ({mastery_pct}%) — 반복 연습 권장"
    return f"숙련도 양호 ({mastery_pct}%) — 심화 도전"