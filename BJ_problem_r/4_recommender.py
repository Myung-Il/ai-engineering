import cloudscraper
import json
from collections import defaultdict
from dirControl import *

def recommend_for_user(handle):
    print(f"🎯 [4단계] 사용자 '{handle}' 맞춤형 분석 시작...\n")
    
    # 1. 학습된 모델(신경망 가중치) 로드
    try:
        with open(get_file_path('model_weights.json'), 'r', encoding='utf-8') as f:
            model = json.load(f)
    except FileNotFoundError:
        print("❌ model_weights.json 파일이 없습니다.")
        return

    # 2. 사용자 현재 상태 실시간 수집
    scraper = cloudscraper.create_scraper()
    res = scraper.get(f"https://solved.ac/api/v3/user/problem_tag_stats?handle={handle}")
    if res.status_code != 200:
        print(f"❌ 사용자 정보를 가져오지 못했습니다.")
        return
        
    items = res.json().get('items', [])
    if not items:
        print(f"⚠️ '{handle}'님은 푼 문제가 없습니다.")
        return

    my_solved = {item['tag']['key']: item['solved'] for item in items}
    
    # [강점/약점 분석]
    best_skill = max(my_solved.items(), key=lambda x: x[1])
    learning_skills = {k: v for k, v in my_solved.items() if 1 <= v < 5}
    weak_skill = min(learning_skills.items(), key=lambda x: x[1]) if learning_skills else min(my_solved.items(), key=lambda x: x[1])

    print(f"👤 [{handle}] 님의 스킬 분석 리포트")
    print(f" 💪 가장 자신 있는 분야: [{best_skill[0]}] ({best_skill[1]}문제 해결)")
    print(f" 🚧 상대적 약점 분야: [{weak_skill[0]}] ({weak_skill[1]}문제 해결)\n")

    # 3. 모델을 통한 추천 (동적 임계값 적용)
    my_mastered = [tag for tag, solved in my_solved.items() if solved >= 5]
    
    if not my_mastered:
        print(f"⚠️ 5문제 이상 푼 알고리즘이 없어 테크트리 추적이 어렵습니다.")
        return

    scores = defaultdict(float)
    reasons = defaultdict(list)
    
    # ⭐ [핵심 추가 로직] 초고수(고인물)를 위한 동적 허들(Threshold) 시스템
    # 기본 5문제 기준으로 찾고, 없으면 10, 15, 20... 최대 100문제까지 기준을 높임
    threshold = 5
    while threshold <= 100:
        # 이 기준값으로 추천할 만한 태그가 있는지 시뮬레이션
        candidates_exist = any(my_solved.get(tag, 0) < threshold for tag in model.keys())
        if candidates_exist:
            break
        threshold += 5 # 추천할 게 없으면 기준을 5문제 더 높임

    if threshold > 5:
        print(f"👑 [초고수 감지] 모든 기초 알고리즘을 마스터하셨군요!")
        print(f"📈 AI가 추천 기준을 상향 조정했습니다. (기준: {threshold}문제 미만 풀이 태그)\n")

    # 가중치 계산
    for my_tag in my_mastered:
        if my_tag in model:
            for next_tag, weight in model[my_tag].items():
                # 동적으로 맞춰진 threshold 미만으로 푼 태그만 추천
                if my_solved.get(next_tag, 0) < threshold:
                    scores[next_tag] += weight
                    reasons[next_tag].append({'base_skill': my_tag, 'weight': weight})

    # 최종 추천 결과 출력
    if scores:
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        print("📊 [분석 결과] 고랭커 실데이터 신경망 기반 추천 트리")
        
        for i, (tag, score) in enumerate(sorted_scores[:3]):
            print(f" {i+1}위 추천 알고리즘: [{tag}] (AI 적합도 점수: {score:.1f}점, 현재 {my_solved.get(tag, 0)}문제 풀이)")
            
            tag_reasons = sorted(reasons[tag], key=lambda x: x['weight'], reverse=True)
            top_reasons = [r['base_skill'] for r in tag_reasons[:3]] # 이유 3개까지만 출력
            
            print(f"    💡 추천 이유: 이미 훌륭하게 다루시는 [{'], ['.join(top_reasons)}] 역량과 강력하게 연계되기 때문입니다.\n")
    else:
        print("⚠️ 모든 알고리즘을 초고수 수준으로 마스터하셨습니다. 더 이상 AI가 가르칠 것이 없습니다!")

if __name__ == "__main__":
    my_id = input("🔍 분석할 사용자 ID를 입력하세요: ")
    recommend_for_user(my_id)