#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_learning_paths.py
==============================================================================
직무별 문제 JSON 파일을 읽어 '학습 경로(Learning Path)' JSON을 생성한다.

입력:  Curriculum/JobProblems/*.json
출력:  Curriculum/LearningPaths/{Job_Name}_path.json

특징
- 외부 LLM API를 전혀 호출하지 않는다. (표준 라이브러리 + scikit-learn + networkx)
- 시나리오 텍스트 유사도: TF-IDF + cosine similarity (한국어/영어 혼용 토크나이저).
- 스킬 유사도: skills_required 필드의 Jaccard 유사도.
- 선행 문제 규칙: 난이도가 낮고(=먼저 풀어야 함) 스킬/시나리오가 겹치는 문제.
- 결과: category/subcategory 클러스터, 의존성 그래프(DAG), 난이도순 정렬 ordered_path.

실행 예시
    python build_learning_paths.py
    python build_learning_paths.py --input-dir Curriculum/JobProblems \\
                                   --output-dir Curriculum/LearningPaths -v
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================================ #
# 0. 상수 및 설정값
# ============================================================================ #

# 난이도 순서(숫자가 작을수록 먼저). 존재하지 않는 값은 가장 뒤로 보낸다.
DIFFICULTY_RANK: Dict[str, int] = {
    "university_level": 0,
    "junior_level":     1,
    "middle_level":     2,
    "senior_level":     3,
}

# 토큰화에 사용할 정규식: 한글/영문/숫자만 남긴다.
#   - 특수문자/이모지/문장부호 전부 제거.
#   - 언더스코어/하이픈으로 붙은 토큰은 별도 토큰으로 분리된다(의도된 동작).
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9가-힣]+")

# 파일명으로 안전한 job_name을 만들기 위한 정규식
#   - 한글/영문/숫자/밑줄/하이픈 외의 모든 문자를 '_'로 치환.
#   - "UI/UX Engineer" → "UI_UX_Engineer", "C++ Dev" → "C_Dev" 같은 직무명 대응.
JOB_NAME_SAFE = re.compile(r"[^A-Za-z0-9가-힣_\-]+")

# 한국어 불용어 — 교육 콘텐츠에 자주 나오는 기능어 위주로 최소 셋 구성.
# (KoNLPy 등 외부 의존성 없이도 TF-IDF 노이즈를 충분히 줄이는 것이 목표)
KOREAN_STOPWORDS = frozenset([
    # 조사/어미
    "이", "가", "은", "는", "을", "를", "의", "에", "에서", "에게", "한테",
    "으로", "로", "와", "과", "이나", "나", "도", "만", "까지", "부터",
    "보다", "처럼", "라고", "이라고",
    # 대명사/지시사
    "이", "그", "저", "것", "수", "등", "및",
    "이것", "그것", "저것", "여기", "거기", "저기",
    # 흔한 동사/형용사 어간
    "하다", "되다", "있다", "없다", "같다", "이다", "아니다",
    "하며", "하고", "하지만", "하여", "해서", "되어", "되었다",
    # 접속사/부사
    "그리고", "또한", "하지만", "그러나", "그래서", "따라서", "즉",
    "또", "다시", "더", "더욱", "가장", "매우", "아주", "잘",
    "잘못", "좀", "거의", "약간", "바로",
    # 지시 형용사 류
    "어떤", "모든", "각", "여러", "다른", "같은", "이러한", "저러한",
    "이번", "때문", "경우", "관련",
    # 기능어
    "대한", "대해", "위한", "위해", "통해", "있는", "없는",
    "예를", "들어",
])

# 영어 기본 불용어 — scikit-learn 'english' 대신 최소 커스텀 셋 사용.
# (scikit-learn 내장 영어 스톱워드는 AI 기술용어 일부와 충돌 가능성이 있어 직접 지정)
ENGLISH_STOPWORDS = frozenset([
    "the", "is", "are", "a", "an", "of", "to", "for", "in", "on", "by",
    "with", "and", "or", "as", "at", "be", "it", "this", "that", "from",
    "we", "you", "they", "i", "he", "she", "his", "her", "their", "its",
    "was", "were", "been", "has", "have", "had", "will", "would", "can",
    "could", "should", "may", "might", "do", "does", "did", "not", "no",
    "if", "then", "than", "so", "but", "also",
])

# 엣지 생성 하이퍼파라미터 — 필요 시 명령줄 인자로 덮어쓸 수 있다.
DEFAULT_SKILL_WEIGHT     = 0.6   # 스킬 가중치 (요구사항: 스킬 중복도 중시)
DEFAULT_SCENARIO_WEIGHT  = 0.4   # 시나리오 가중치
DEFAULT_MIN_SCORE        = 0.15  # combined_score 하한
DEFAULT_MAX_PREREQS      = 3     # 타깃 문제 한 개당 최대 선행 문제 개수
DEFAULT_SAME_CAT_BONUS   = 0.10  # 동일 카테고리 보너스

# 기본 입출력 경로 (요구사항 준수: Curriculum/JobProblems, Curriculum/LearningPaths)
_SCRIPT_DIR        = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR  = _SCRIPT_DIR / "JobProblems"
DEFAULT_OUTPUT_DIR = _SCRIPT_DIR / "LearningPaths"

logger = logging.getLogger("learning_path_builder")


# ============================================================================ #
# 1. 텍스트 전처리 & 토큰화
# ============================================================================ #

def korean_friendly_tokenize(text: str) -> List[str]:
    """
    한국어/영어가 섞인 시나리오 텍스트를 토큰 리스트로 변환.

    전략
      1) 정규식으로 한글/영문/숫자 덩어리만 추출.
      2) lower-case 통일 (영어 대소문자 구분 제거).
      3) 한/영 불용어 제거.
      4) 1글자 토큰 제거(한국어는 의미 노이즈, 영어는 'i','a' 등 이미 불용어로 컷).

    주의
      - 형태소 분석기를 쓰지 않는 단순 구현이므로, '분류기'와 '분류'는 별도 토큰이 된다.
        이는 TF-IDF 가중치로 자연스럽게 보정되며, 외부 의존성을 피하기 위한 의도적 단순화.
    """
    if not text:
        return []
    tokens = TOKEN_PATTERN.findall(text.lower())
    return [
        t for t in tokens
        if len(t) >= 2
        and t not in KOREAN_STOPWORDS
        and t not in ENGLISH_STOPWORDS
    ]


# ============================================================================ #
# 2. 입출력
# ============================================================================ #

def load_problems(json_path: Path) -> List[Dict[str, Any]]:
    """
    JSON 파일을 로드한다. 스키마는 다음 두 형태를 모두 지원한다.
      (a) [ {...}, {...}, ... ]                        ← 최상위가 리스트
      (b) {"problems": [...]} / {"questions": [...]}   ← 흔한 랩핑 dict

    반환되는 각 문제는 다음 기본값이 채워져 있음이 보장된다:
      question_id, difficulty, category, subcategory, skills_required, scenario
    """
    with json_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        problems: Optional[list] = None
        for key in ("problems", "questions", "items", "data"):
            if isinstance(raw.get(key), list):
                problems = raw[key]
                break
        if problems is None:
            raise ValueError(f"{json_path.name}: dict 최상위에서 리스트 필드를 찾지 못함.")
    elif isinstance(raw, list):
        problems = raw
    else:
        raise ValueError(f"{json_path.name}: 지원하지 않는 최상위 타입 {type(raw).__name__}")

    # 필드 보정 — 누락된 키가 있어도 뒷 단계가 깨지지 않게 기본값을 세팅한다.
    cleaned: List[Dict[str, Any]] = []
    for idx, p in enumerate(problems):
        if not isinstance(p, dict):
            logger.warning("  [%s] %d번째 항목이 dict가 아님 — 건너뜀.", json_path.name, idx)
            continue
        p.setdefault("question_id",     idx + 1)
        p.setdefault("difficulty",      "university_level")
        p.setdefault("category",        "Uncategorized")
        p.setdefault("subcategory",     "")
        p.setdefault("skills_required", [])
        p.setdefault("scenario",        "")
        # skills_required가 list가 아니면(예: 문자열) 리스트로 보정
        if not isinstance(p["skills_required"], list):
            p["skills_required"] = [str(p["skills_required"])]
        cleaned.append(p)
    return cleaned


def save_json(obj: Any, path: Path) -> None:
    """가독성 좋은 JSON으로 저장 — UTF-8, 들여쓰기 2칸, 한글 그대로."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ============================================================================ #
# 3. 유사도 계산
# ============================================================================ #

def compute_scenario_similarity(problems: List[Dict[str, Any]]) -> np.ndarray:
    """
    시나리오 텍스트 간 TF-IDF cosine similarity 행렬 (n x n).
      - 문서가 2개 미만이거나 어휘가 비어 있으면 0 행렬 반환.
      - 대각선은 0으로 설정(자기 자신 제외).
    """
    n = len(problems)
    if n < 2:
        return np.zeros((n, n), dtype=float)

    docs = [str(p.get("scenario", "")) for p in problems]
    try:
        vectorizer = TfidfVectorizer(
            tokenizer=korean_friendly_tokenize,  # 한국어 친화 커스텀 토크나이저
            lowercase=False,                      # 토크나이저에서 이미 처리
            token_pattern=None,                   # tokenizer 지정 시 경고 억제
            min_df=1,                             # 드문 용어도 기술용어일 수 있어 유지
            norm="l2",                            # 코사인 유사도를 자연스럽게
        )
        matrix = vectorizer.fit_transform(docs)
    except ValueError as exc:
        # 모든 문서가 불용어로만 이뤄진 희귀 케이스
        logger.warning("  TF-IDF 실패 (%s) — 시나리오 유사도 0 처리.", exc)
        return np.zeros((n, n), dtype=float)

    if matrix.shape[1] == 0:
        logger.warning("  TF-IDF 어휘가 비었음 — 시나리오 유사도 0 처리.")
        return np.zeros((n, n), dtype=float)

    sim = cosine_similarity(matrix).astype(float)
    np.fill_diagonal(sim, 0.0)
    return sim


def compute_skill_similarity(problems: List[Dict[str, Any]]) -> np.ndarray:
    """
    skills_required 기반 Jaccard 유사도 행렬.
      J(a,b) = |a ∩ b| / |a ∪ b|
    공통 스킬이 없으면 0, 스킬 셋이 비어 있는 문제는 0으로 남겨 둔다.
    """
    n = len(problems)
    sim = np.zeros((n, n), dtype=float)

    # 스킬 셋을 lower-case로 정규화(중복 대소문자 방지)
    skill_sets: List[set] = [
        {str(s).strip().lower() for s in (p.get("skills_required") or []) if str(s).strip()}
        for p in problems
    ]

    for i in range(n):
        a = skill_sets[i]
        if not a:
            continue
        for j in range(i + 1, n):
            b = skill_sets[j]
            if not b:
                continue
            inter = len(a & b)
            if inter == 0:
                continue
            union = len(a | b)
            sim[i, j] = sim[j, i] = inter / union
    return sim


# ============================================================================ #
# 4. 클러스터링 (category + subcategory)
# ============================================================================ #

def build_clusters(problems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    category + subcategory 조합으로 문제를 묶고, 각 클러스터 내부는 난이도 순으로 정렬.
    결과는 크기가 큰 클러스터부터 정렬하여 출력.
    """
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for p in problems:
        groups[(p["category"], p["subcategory"])].append(p)

    clusters: List[Dict[str, Any]] = []
    for (cat, sub), items in groups.items():
        items_sorted = sorted(
            items,
            key=lambda x: (
                DIFFICULTY_RANK.get(x["difficulty"], 99),
                x["question_id"],
            ),
        )
        skill_counter: Counter = Counter()
        for it in items_sorted:
            skill_counter.update(s for s in it.get("skills_required", []) if s)

        clusters.append({
            "cluster_id":   f"{cat}__{sub}" if sub else cat,
            "category":     cat,
            "subcategory":  sub,
            "size":         len(items_sorted),
            "question_ids": [it["question_id"] for it in items_sorted],
            "common_skills": [s for s, _ in skill_counter.most_common(5)],
        })

    clusters.sort(key=lambda c: (-c["size"], c["cluster_id"]))
    return clusters


# ============================================================================ #
# 5. 의존성(선행 문제) 계산
# ============================================================================ #

def get_prerequisites(
    problems: List[Dict[str, Any]],
    *,
    skill_weight:    float = DEFAULT_SKILL_WEIGHT,
    scenario_weight: float = DEFAULT_SCENARIO_WEIGHT,
    min_score:       float = DEFAULT_MIN_SCORE,
    max_prereqs:     int   = DEFAULT_MAX_PREREQS,
    same_cat_bonus:  float = DEFAULT_SAME_CAT_BONUS,
) -> List[Dict[str, Any]]:
    """
    모든 문제 쌍을 비교해 '선행 → 타깃' 관계(엣지)를 산출한다.

    엣지 조건
      (1) 선행 문제의 난이도 < 타깃 문제의 난이도   (동일 레벨은 엣지 제외)
      (2) skill_sim 또는 scenario_sim 중 하나 이상 > 0
      (3) combined_score >= min_score
      (4) 타깃 문제당 상위 max_prereqs개만 채택 (그래프가 과밀해지는 것을 방지)

    combined_score
      = skill_weight * jaccard(skills_a, skills_b)
      + scenario_weight * cosine_tfidf(scenario_a, scenario_b)
      + (동일 category 일 경우 same_cat_bonus, 상한 1.0)
    """
    n = len(problems)
    if n < 2:
        return []

    skill_sim    = compute_skill_similarity(problems)
    scenario_sim = compute_scenario_similarity(problems)

    ids   = [p["question_id"] for p in problems]
    diffs = [DIFFICULTY_RANK.get(p["difficulty"], 99) for p in problems]
    cats  = [p["category"] for p in problems]

    edges: List[Dict[str, Any]] = []

    for j in range(n):  # target 문제
        candidates: List[Tuple[float, int, float, float]] = []  # (score, i, sk, sc)
        for i in range(n):
            if i == j:
                continue
            # (1) 난이도 방향성 — 반드시 낮음 → 높음
            if diffs[i] >= diffs[j]:
                continue
            sk = float(skill_sim[i, j])
            sc = float(scenario_sim[i, j])
            if sk == 0.0 and sc == 0.0:
                continue
            score = skill_weight * sk + scenario_weight * sc
            if cats[i] == cats[j] and cats[i] and cats[i] != "Uncategorized":
                score = min(1.0, score + same_cat_bonus)
            if score < min_score:
                continue
            candidates.append((score, i, sk, sc))

        if not candidates:
            continue

        # 점수 내림차순, 동점이면 난이도 낮은 것(=더 기초) 우선
        candidates.sort(key=lambda t: (-t[0], diffs[t[1]], ids[t[1]]))

        for score, i, sk, sc in candidates[:max_prereqs]:
            edges.append({
                "Preceding_ID":        ids[i],
                "Target_ID":           ids[j],
                "skill_overlap":       round(sk, 4),
                "scenario_similarity": round(sc, 4),
                "combined_score":      round(score, 4),
                "Logic":               _build_logic_text(problems[i], problems[j], sk, sc),
            })
    return edges


def _build_logic_text(prereq: Dict[str, Any], target: Dict[str, Any],
                      skill_sim: float, scenario_sim: float) -> str:
    """사람이 읽을 수 있는 선행 관계 근거 문자열."""
    a = {str(s).lower() for s in prereq.get("skills_required", [])}
    b = {str(s).lower() for s in target.get("skills_required", [])}
    common = sorted(a & b)[:3]

    parts = []
    if common:
        parts.append(f"공통 스킬 {common} (Jaccard={skill_sim:.2f})")
    else:
        parts.append(f"스킬 Jaccard={skill_sim:.2f}")
    parts.append(f"시나리오 유사도={scenario_sim:.2f}")
    parts.append(f"난이도 {prereq.get('difficulty')} → {target.get('difficulty')}")
    if prereq.get("category") == target.get("category"):
        parts.append(f"동일 카테고리({prereq.get('category')})")
    return " | ".join(parts)


# ============================================================================ #
# 6. 그래프 & 순서 정렬
# ============================================================================ #

def build_dependency_graph(
    problems: List[Dict[str, Any]],
    edges:    List[Dict[str, Any]],
) -> nx.DiGraph:
    """
    networkx DiGraph 생성.
    엣지가 항상 '낮은 난이도 → 높은 난이도' 방향이므로 DAG가 보장되지만,
    만약을 대비해 사이클 검출 시 약한 엣지부터 끊는 방어 로직을 둔다.
    """
    g = nx.DiGraph()
    for p in problems:
        g.add_node(
            p["question_id"],
            difficulty=p["difficulty"],
            category=p["category"],
            subcategory=p["subcategory"],
        )
    for e in edges:
        g.add_edge(
            e["Preceding_ID"],
            e["Target_ID"],
            weight=e["combined_score"],
            skill_overlap=e["skill_overlap"],
            scenario_similarity=e["scenario_similarity"],
        )

    # DAG 검증 — 이론상 발생하지 않지만 안전장치
    while not nx.is_directed_acyclic_graph(g):
        try:
            cyc = nx.find_cycle(g, orientation="original")
        except nx.NetworkXNoCycle:
            break
        # 사이클 엣지 중 weight 가장 낮은 엣지를 잘라냄
        weakest = min(cyc, key=lambda e: g[e[0]][e[1]].get("weight", 0))
        logger.warning("  사이클 감지, 엣지 제거: %s → %s", weakest[0], weakest[1])
        g.remove_edge(weakest[0], weakest[1])
    return g


def build_ordered_path(
    problems: List[Dict[str, Any]],
    graph:    nx.DiGraph,
) -> List[int]:
    """
    [university → junior → middle → senior] 순서로 정렬.
    동일 난이도 내부에서는:
      - in_degree(선행 문제 수)가 적은 것 먼저 = 기초/독립 문제 먼저.
      - 그 다음 question_id 오름차순.

    (전역 topological_sort를 쓰지 않는 이유:
     '난이도 계층'을 명확한 1차 정렬 키로 유지하기 위함.
     엣지는 난이도 구분과 상관없이 추가 정보를 제공하므로,
     난이도 버킷 내부 정렬에만 in_degree 휴리스틱을 사용한다.)
    """
    buckets: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for p in problems:
        buckets[DIFFICULTY_RANK.get(p["difficulty"], 99)].append(p)

    ordered: List[int] = []
    for rank in sorted(buckets):
        group = buckets[rank]
        group.sort(
            key=lambda p: (graph.in_degree(p["question_id"]), p["question_id"])
        )
        ordered.extend(p["question_id"] for p in group)
    return ordered


# ============================================================================ #
# 7. 파일 단위 파이프라인
# ============================================================================ #

def _infer_job_name(problems: List[Dict[str, Any]], path: Path) -> str:
    """
    문제 데이터의 job_role 필드 또는 파일 이름에서 직무명 추출 후,
    파일명으로 안전한 형태로 정규화한다.

    예:
      "AI Engineer"           → "AI_Engineer"
      "UI/UX Engineer"        → "UI_UX_Engineer"   ← 슬래시로 하위 폴더가 생기는 사고 방지
      "C++ Developer"         → "C_Developer"
      "Cloud Infra Engineer"  → "Cloud_Infra_Engineer"
    """
    raw_name: Optional[str] = None
    for p in problems:
        role = p.get("job_role")
        if role:
            raw_name = str(role).strip()
            break
    if not raw_name:
        raw_name = path.stem

    # 한글/영문/숫자/밑줄/하이픈 외의 문자는 전부 '_'로 치환, 양끝 '_' 제거
    safe = JOB_NAME_SAFE.sub("_", raw_name).strip("_")
    # 치환 결과가 완전히 비어버리면(예: 이름이 "///") 파일명 stem으로 fallback
    return safe if safe else path.stem


def process_file(json_path: Path, output_dir: Path) -> Optional[Path]:
    """단일 JSON 처리. 성공 시 저장된 경로를 반환, 빈 파일이면 None."""
    logger.info("[PROCESS] %s", json_path.name)

    problems = load_problems(json_path)
    if not problems:
        logger.warning("  문제 0건 — 건너뜀.")
        return None

    job_name = _infer_job_name(problems, json_path)
    logger.info("  job_name=%s, 문항 수=%d", job_name, len(problems))

    # --- 핵심 4단계 ---
    clusters = build_clusters(problems)
    edges    = get_prerequisites(problems)
    graph    = build_dependency_graph(problems, edges)
    ordered  = build_ordered_path(problems, graph)

    # 진단 지표
    diff_counter = Counter(p["difficulty"] for p in problems)
    isolated = [
        p["question_id"] for p in problems
        if graph.in_degree(p["question_id"]) == 0
        and graph.out_degree(p["question_id"]) == 0
    ]

    output = {
        "metadata": {
            "job_role":                 job_name,
            "source_file":              json_path.name,
            "total_problems":           len(problems),
            "difficulty_distribution":  dict(diff_counter),
            "cluster_count":            len(clusters),
            "edge_count":               len(edges),
            "isolated_problem_count":   len(isolated),
            "parameters": {
                "skill_weight":           DEFAULT_SKILL_WEIGHT,
                "scenario_weight":        DEFAULT_SCENARIO_WEIGHT,
                "min_combined_score":     DEFAULT_MIN_SCORE,
                "max_prereqs_per_target": DEFAULT_MAX_PREREQS,
                "same_category_bonus":    DEFAULT_SAME_CAT_BONUS,
            },
        },
        "clusters": clusters,
        "dependency_graph": {
            "nodes": [
                {
                    "question_id":     p["question_id"],
                    "difficulty":      p["difficulty"],
                    "category":        p["category"],
                    "subcategory":     p["subcategory"],
                    "skills_required": p["skills_required"],
                }
                for p in problems
            ],
            "edges": edges,
        },
        "ordered_path":      ordered,
        "isolated_problems": isolated,
    }

    out_path = output_dir / f"{job_name}_path.json"
    save_json(output, out_path)
    logger.info("  → 저장 완료: %s", out_path)
    return out_path


# ============================================================================ #
# 8. 일괄 처리 엔트리
# ============================================================================ #

def process_all_files(input_dir: Path, output_dir: Path) -> Dict[str, Any]:
    """input_dir 내 모든 *.json을 순회. 파일별 오류는 기록하고 계속 진행."""
    if not input_dir.exists():
        raise FileNotFoundError(f"입력 폴더가 없습니다: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"입력 경로가 폴더가 아닙니다: {input_dir}")

    files = sorted(p for p in input_dir.iterdir() if p.suffix.lower() == ".json")
    if not files:
        logger.warning("입력 폴더에 JSON 파일이 없습니다: %s", input_dir)
        return {"processed": [], "failed": []}

    processed: List[str] = []
    failed:    List[Dict[str, str]] = []

    for fp in files:
        try:
            out = process_file(fp, output_dir)
            if out is not None:
                processed.append(out.name)
        except json.JSONDecodeError as exc:
            logger.error("  [FAIL] %s — JSON 파싱 오류: %s", fp.name, exc)
            failed.append({"file": fp.name, "error": f"JSONDecodeError: {exc}"})
        except Exception as exc:  # pylint: disable=broad-except
            # 개별 파일 실패로 전체 배치가 중단되지 않도록 광범위 캐치
            logger.error("  [FAIL] %s — %s: %s", fp.name, type(exc).__name__, exc)
            logger.debug(traceback.format_exc())
            failed.append({"file": fp.name, "error": f"{type(exc).__name__}: {exc}"})

    logger.info("완료: 성공 %d개, 실패 %d개", len(processed), len(failed))
    if failed:
        logger.warning("실패 목록: %s", [f["file"] for f in failed])
    return {"processed": processed, "failed": failed}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="직무별 문제 JSON → 학습 경로 JSON 변환 (LLM 미사용)"
    )
    parser.add_argument(
        "--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
        help=f"입력 폴더 (기본: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"출력 폴더 (기본: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="디버그 로그 출력",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        result = process_all_files(args.input_dir, args.output_dir)
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("치명적 오류: %s", exc)
        logger.debug(traceback.format_exc())
        return 2

    # 성공 0이거나 실패가 있으면 비정상 종료 코드를 반환
    if not result["processed"] and not result["failed"]:
        return 0
    return 0 if not result["failed"] else 1


if __name__ == "__main__":
    sys.exit(main())