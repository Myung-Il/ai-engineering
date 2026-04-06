import asyncio
import aiohttp
import pandas as pd
from collections import Counter
import time
import sys
import os
from githubToken import GITHUB_TOKEN

# ==========================================
# 1. 환경 설정
# ==========================================
GITHUB_TOKEN = GITHUB_TOKEN
HEADERS = {'Authorization': f'token {GITHUB_TOKEN}', 'Accept': 'application/vnd.github.v3+json'}

# 동시 요청 수 제한 (서버 과부하 및 차단 방지)
MAX_CONCURRENT_REQUESTS = 50 
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# ==========================================
# 2. 휴리스틱 직군 분류 함수
# ==========================================
# 점수에 따라 레벨(Lv)과 칭호를 반환하는 헬퍼 함수
def get_level_tag(score):
    if score <= 5: return "Lv0. 추론 안됨"
    elif score <= 15: return "Lv1. 의심"
    elif score <= 75: return "Lv2. 강한 의심"
    else: return "Lv3. 확신"

def classify_role(languages, topics):
    scores = {
        'Web Frontend': 0, 'Web Backend': 0, 'Mobile App': 0, 
        'ML/AI Engineer': 0, 'Data Scientist': 0, 'Data Engineer': 0,
        'DevOps/SRE': 0, 'Systems/Embedded': 0, 'Game Dev': 0, 'Web3/Blockchain': 0
    }
    
    languages = [str(l).lower() for l in languages if l]
    topics = [str(t).lower() for t in topics if t]

    # [1] 언어 기반 스코어링 (가중치 1)
    for lang, count in Counter(languages).items():
        if lang in ['javascript', 'typescript', 'html', 'css']: scores['Web Frontend'] += count
        elif lang in ['java', 'go']: scores['Web Backend'] += count
        elif lang in ['swift', 'kotlin', 'dart']: scores['Mobile App'] += count
        elif lang in ['r']: scores['Data Scientist'] += count
        elif lang in ['scala']: scores['Data Engineer'] += count
        elif lang in ['c', 'c++', 'rust']: scores['Systems/Embedded'] += count
        elif lang in ['c#']: scores['Game Dev'] += count
        elif lang in ['solidity']: scores['Web3/Blockchain'] += count
        elif lang == 'python':
            scores['Web Backend'] += (count * 0.5)
            scores['ML/AI Engineer'] += (count * 0.5)
            scores['Data Scientist'] += (count * 0.5)

    # [2] 토픽 기반 스코어링 (가중치 3)
    topic_weight = 3
    for t in topics:
        if any(k in t for k in ['react', 'vue', 'frontend', 'tailwind']): scores['Web Frontend'] += topic_weight
        elif any(k in t for k in ['spring', 'django', 'backend', 'api']): scores['Web Backend'] += topic_weight
        elif any(k in t for k in ['ios', 'android', 'flutter']): scores['Mobile App'] += topic_weight
        elif any(k in t for k in ['machine-learning', 'pytorch', 'tensorflow']): scores['ML/AI Engineer'] += topic_weight
        elif any(k in t for k in ['data-analysis', 'pandas']): scores['Data Scientist'] += topic_weight
        elif any(k in t for k in ['hadoop', 'spark', 'docker', 'kubernetes']): scores['DevOps/SRE'] += topic_weight

    # [3] 레벨 시스템 및 본캐/부캐 로직 적용
    # 점수가 높은 순서대로 직무를 정렬
    sorted_roles = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    
    top1_role, top1_score = sorted_roles[0] # 1등 (본캐)
    top2_role, top2_score = sorted_roles[1] # 2등 (부캐)

    # 1등 점수조차 5점 이하라면 아예 추론 불가 처리
    if top1_score <= 5:
        return "X (추론 안됨)"
        
    # 본캐 텍스트 생성
    result_text = f"{top1_role} ({get_level_tag(top1_score)} / {top1_score}점)"
    
    # 2등 점수가 6점(Lv1. 의심) 이상일 경우에만 부캐 텍스트 추가
    if top2_score > 5:
        result_text += f"\n 🕵️ 부캐 스캔: {top2_role} ({get_level_tag(top2_score)} / {top2_score}점)"
        
    return result_text

# ==========================================
# 3. 비동기 데이터 수집 함수 (단일 유저 분석)
# ==========================================
async def fetch_user_data(session, username):
    url = f"https://api.github.com/users/{username}/repos?per_page=100&sort=pushed"
    
    async with semaphore:
        async with session.get(url, headers=HEADERS) as response:
            if response.status != 200:
                return None
            
            repos = await response.json()
            # 레포지토리가 2개 미만이거나 데이터 형식이 이상하면 패스 (데이터 품질 유지)
            if not isinstance(repos, list) or len(repos) < 2:
                return None 
                
            languages = [repo.get('language') for repo in repos if repo.get('language')]
            topics = []
            for repo in repos:
                if repo.get('topics'): topics.extend(repo.get('topics'))
                
            if not languages:
                return None
                
            role = classify_role(languages, topics)
            
            return {
                'github_id': username,
                'target_role': role,
                'total_repos': len(repos),
                'top_languages': str(dict(Counter(languages).most_common(3))),
                'topics': str(list(set(topics))[:10])
            }

# ==========================================
# 4. 비동기 유저 ID 검색 (연도별 1,000명씩)
# ==========================================
async def fetch_users_by_year(session, year):
    usernames = []
    # 10페이지 * 100명 = 연도별 최대 1000명 검색
    for page in range(1, 11): 
        url = f"https://api.github.com/search/users?q=type:user+created:{year}-01-01..{year}-12-31+repos:>10&per_page=100&page={page}"
        async with session.get(url, headers=HEADERS) as res:
            if res.status == 200:
                data = await res.json()
                usernames.extend([item['login'] for item in data.get('items', [])])
            else:
                break
        await asyncio.sleep(2) # 깃허브 검색 API(1분 30회 제한) 보호용 딜레이
    
    print(f"✅ [{year}년 가입자] {len(usernames)}명 ID 수집 완료")
    return usernames

# ==========================================
# 5. 메인 실행 루프
# ==========================================
async def main():
    print("🚀 초고속 5,000명 수집 파이프라인 가동...")
    start_time = time.time()
    
    async with aiohttp.ClientSession() as session:
        # [STEP 1] 유저 ID 수집 (2019~2023년)
        print("\n1️⃣ 최근 5개년(2019~2023) 분할 검색으로 유저 ID 수집 중...")
        tasks = [fetch_users_by_year(session, year) for year in range(2019, 2024)]
        results = await asyncio.gather(*tasks)
        
        all_usernames = []
        for result in results:
            all_usernames.extend(result)
            
        all_usernames = list(set(all_usernames)) # 중복 유저 제거
        print(f"🎯 총 {len(all_usernames)}명의 유저 아이디 확보 완료!")
        
        # [STEP 2] 확보된 유저의 디테일 데이터 비동기 분석
        print("\n2️⃣ 비동기 병렬 처리로 개별 유저 레포지토리 분석 시작 (최대 50명 동시 접속)...")
        dataset = []
        analyze_tasks = [fetch_user_data(session, user) for user in all_usernames]
        
        total_tasks = len(analyze_tasks)
        for i, coro in enumerate(asyncio.as_completed(analyze_tasks), 1):
            result = await coro
            if result:
                dataset.append(result)
            
            # 진행 상황을 100명 단위로 출력
            if i % 100 == 0 or i == total_tasks:
                print(f"⏳ 진행 상황: {i} / {total_tasks} 명 분석 완료...")

        # [STEP 3] 결과 저장
        print("\n3️⃣ 분석 완료, CSV 파일로 저장합니다...")
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(current_dir, 'github_dataset.csv')
        
        df = pd.DataFrame(dataset)
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
    end_time = time.time()
    print(f"\n🎉 모든 작업이 끝났습니다!")
    print(f"📁 생성된 파일: {csv_path}")
    print(f"📊 최종 수집된 유효 데이터 수: {len(df)}개")
    print(f"⏱️ 총 소요 시간: {round((end_time - start_time) / 60, 2)}분")

if __name__ == "__main__":
    # Windows 환경에서 발생하는 asyncio EventLoop 에러 방지 코드
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # 파이썬 3.7 이상 비동기 실행 표준
    asyncio.run(main())