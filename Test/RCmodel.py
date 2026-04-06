import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. API 데이터 수집 함수
# ==========================================
def fetch_user_data(github_id, solvedac_id):
    print(f"[{github_id}]님의 데이터를 수집 중입니다...")
    
    # --- GitHub 데이터 수집 ---
    # 사용자의 Public 레포지토리 목록을 가져옵니다 (최대 100개)
    gh_url = f"https://api.github.com/users/{github_id}/repos?per_page=100"
    gh_res = requests.get(gh_url)
    
    py_count, js_count, topic_ai, topic_web = 0, 0, 0, 0
    total_repos = 0
    
    if gh_res.status_code == 200:
        repos = gh_res.json()
        total_repos = len(repos)
        
        for repo in repos:
            # 주력 언어 파악
            lang = repo.get('language')
            if lang == 'Python': py_count += 1
            if lang == 'JavaScript' or lang == 'TypeScript': js_count += 1
            
            # 관심 토픽 파악
            topics = repo.get('topics', [])
            if any(t in ['machine-learning', 'deep-learning', 'ai', 'data-science'] for t in topics):
                topic_ai += 1
            if any(t in ['react', 'vue', 'frontend', 'web', 'django', 'spring'] for t in topics):
                topic_web += 1
                
    py_ratio = py_count / total_repos if total_repos > 0 else 0.0
    js_ratio = js_count / total_repos if total_repos > 0 else 0.0

    # --- Solved.ac 데이터 수집 ---
    headers = {'User-Agent': 'Mozilla/5.0'} # Solved.ac API는 User-Agent 헤더가 필요함
    tier, tag_math, tag_dp, total_solved = 0, 0, 0, 1
    
    # 1. 사용자 티어 및 전체 푼 문제 수 조회
    sa_url = f"https://solved.ac/api/v3/user/show?handle={solvedac_id}"
    sa_res = requests.get(sa_url, headers=headers)
    if sa_res.status_code == 200:
        sa_data = sa_res.json()
        tier = sa_data.get('tier', 0)
        total_solved = sa_data.get('solvedCount', 1)
        if total_solved == 0: total_solved = 1

    # 2. 태그별 푼 문제 통계 조회 (수학, DP 비율 추출)
    sa_tag_url = f"https://solved.ac/api/v3/user/problem_tag_stats?handle={solvedac_id}"
    sa_tag_res = requests.get(sa_tag_url, headers=headers)
    if sa_tag_res.status_code == 200:
        tags = sa_tag_res.json().get('items', [])
        for tag_data in tags:
            tag_key = tag_data.get('tag', {}).get('key', '')
            if tag_key == 'math':
                tag_math = tag_data.get('solved', 0)
            elif tag_key == 'dp':
                tag_dp = tag_data.get('solved', 0)

    math_ratio = tag_math / total_solved
    dp_ratio = tag_dp / total_solved

    # 추출된 특성(Feature) 리스트 반환
    # [Python비율, JS비율, AI토픽수, Web토픽수, 티어, 수학비율, DP비율]
    return [[py_ratio, js_ratio, topic_ai, topic_web, tier, math_ratio, dp_ratio]]


# ==========================================
# 2. 모델 학습 (기존과 동일한 가상 데이터 기반)
# ==========================================
# 실제 서비스 시에는 많은 유저들의 데이터를 모아 아래 데이터를 교체해야 합니다.
mock_data = {
    'python_ratio': [0.8, 0.1, 0.5, 0.9, 0.0],
    'js_ratio': [0.1, 0.8, 0.4, 0.0, 0.9],
    'topic_ai': [5, 0, 2, 8, 0],
    'topic_web': [0, 6, 4, 1, 7],
    'solved_tier': [15, 10, 18, 20, 8],
    'tag_math': [0.4, 0.1, 0.2, 0.5, 0.1],
    'tag_dp': [0.2, 0.3, 0.5, 0.3, 0.1],
    'target_role': ['AI Engineer', 'Frontend Dev', 'Backend Dev', 'Data Scientist', 'Frontend Dev']
}
df = pd.DataFrame(mock_data)
X = df.drop('target_role', axis=1)
y = df['target_role']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)


# ==========================================
# 3. 실제 사용자 분석 실행
# ==========================================
if __name__ == "__main__":
    # 분석하고 싶은 사용자의 아이디를 입력하세요
    TARGET_GITHUB_ID = "myung-il"
    TARGET_SOLVEDAC_ID = "ktr040415"
    
    # 1. API를 통해 데이터 추출
    user_features = fetch_user_data(TARGET_GITHUB_ID, TARGET_SOLVEDAC_ID)
    
    # 2. 데이터 스케일링 (학습 데이터 기준)
    user_scaled = scaler.transform(user_features)
    
    # 3. 모델 예측
    predicted_role = model.predict(user_scaled)
    probabilities = model.predict_proba(user_scaled)
    
    # 4. 결과 출력
    print("\n" + "="*40)
    print("📊 사용자 분석 결과 리포트")
    print("="*40)
    print(f"추출된 원본 데이터: {user_features[0]}")
    print(f"\n💡 예측된 주력/관심 직무: **{predicted_role[0]}**\n")
    print("📈 직무별 잠재력 (확률):")
    for role, prob in zip(model.classes_, probabilities[0]):
        print(f" - {role:15s} : {prob*100:>5.1f}%")
    print("="*40)