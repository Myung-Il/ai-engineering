import pandas as pd
import joblib
import aiohttp
import asyncio
from collections import Counter
from githubToken import GITHUB_TOKEN
from dirControl import *

# 1. 모델 및 특성 목록 로드
model = joblib.load(load_data('github_role_model.pkl'))
model_features = joblib.load(load_data('model_features.pkl'))

GITHUB_TOKEN = GITHUB_TOKEN
HEADERS = {'Authorization': f'token {GITHUB_TOKEN}'}

async def fetch_user_data(username):
    url = f"https://api.github.com/users/{username}/repos?per_page=100"
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=HEADERS) as res:
            if res.status != 200:
                # 🛑 깃허브가 거절한 진짜 이유를 터미널에 출력합니다.
                error_msg = await res.json()
                print(f"\n🚨 깃허브 API 거절 (상태 코드: {res.status})")
                print(f"📝 상세 사유: {error_msg.get('message', '알 수 없음')}")
                return None
            repos = await res.json()
            
    if not repos: 
        print(f"\n⚠️ 해당 유저({username})는 공개된 레포지토리가 0개입니다.")
        return None
    
    languages = [r.get('language') for r in repos if r.get('language')]
    topics = []
    for r in repos:
        if r.get('topics'): topics.extend(r.get('topics'))
        
    return {'languages': languages, 'topics': topics, 'total_repos': len(repos)}

def get_prediction(user_data):
    # 모델이 학습했던 것과 동일한 형태의 데이터프레임 생성
    input_data = pd.DataFrame(0, index=[0], columns=model_features)
    
    # 데이터 채우기
    input_data['total_repos'] = user_data['total_repos']
    for lang, count in Counter(user_data['languages']).items():
        col = f'lang_{lang}'
        if col in input_data.columns: input_data[col] = count
            
    for topic in user_data['topics']:
        col = f'topic_{topic}'
        if col in input_data.columns: input_data[col] = 1
            
    # AI 확률 예측
    probabilities = model.predict_proba(input_data)[0]
    classes = model.classes_
    
    # 확률 순으로 정렬
    results = sorted(zip(classes, probabilities), key=lambda x: x[1], reverse=True)
    return results

async def main():
    print("🔮 [GitHub AI Profile] 직군 분석기에 오신 것을 환영합니다.")
    username = input("분석할 GitHub ID를 입력하세요: ")
    
    user_raw = await fetch_user_data(username)
    if not user_raw:
        print("❌ 유저 정보를 가져올 수 없습니다.")
        return

    predictions = get_prediction(user_raw)
    
    print(f"\n=======================================")
    print(f"👤 분석 대상: {username}")
    print(f"📂 스캔한 레포지토리: {user_raw['total_repos']}개")
    print(f"=======================================")
    
    # 본캐 (1위)
    main_role, main_prob = predictions[0]
    print(f"🎯 AI 강한 추론 직무: {main_role} ({main_prob*100:.1f}%)")
    
    # 부캐 (2위 - 확률이 5% 이상일 때만 출력)
    sub_role, sub_prob = predictions[1]
    if sub_prob > 0.05:
        print(f"🕵️  AI 약한 추론 직무: {sub_role} ({sub_prob*100:.1f}%)")
    
    print(f"=======================================")

if __name__ == "__main__":
    asyncio.run(main())