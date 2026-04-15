import cloudscraper
import json
import time
from dirControl import *

def collect_raw_data(target_users_count=1000):
    # 최신 브라우저로 위장하여 Cloudflare 봇 방어 시스템을 우회합니다.
    scraper = cloudscraper.create_scraper(
        browser={
            'browser': 'chrome',
            'platform': 'windows',
            'desktop': True
        }
    )
    print(f"🔍 [1단계] 최상위 랭커 {target_users_count}명 목록 수집 중...")
    
    users = []
    page = 1
    
    # 1. 검색 API 대신 랭킹 API 사용 (1페이지당 상위 100명씩 가져옴)
    while len(users) < target_users_count:
        ranking_url = f"https://solved.ac/api/v3/ranking/tier?page={page}"
        
        try:
            res = scraper.get(ranking_url, timeout=10)
            
            if res.status_code == 200:
                # Cloudflare 보안 페이지가 뜰 경우를 대비한 예외 처리
                try:
                    items = res.json().get('items', [])
                except json.JSONDecodeError:
                    print("⚠️ 봇 차단 페이지가 반환되었습니다. 10초 대기 후 재시도합니다...")
                    time.sleep(10)
                    continue

                if not items:
                    print("⚠️ 더 이상 랭킹 결과가 없습니다.")
                    break
                
                for u in items:
                    if len(users) < target_users_count:
                        users.append(u['handle'])
                
                print(f"  - [페이지 {page}] 유저 {len(items)}명 추가 (현재 총 {len(users)}명)")
                page += 1
                time.sleep(1.5) # 페이지 넘길 때 딜레이
            else:
                print(f"⚠️ API 요청 실패 (상태 코드: {res.status_code}). 5초 후 재시도...")
                time.sleep(5)
        except Exception as e:
            print(f"⚠️ 네트워크 에러: {e}. 5초 후 재시도...")
            time.sleep(5)
            
    print(f"\n✅ 목록 수집 완료! 총 {len(users)}명의 유저 목록을 확보했습니다.")
    
    if not users:
        print("❌ 유저 목록을 가져오지 못해 수집을 종료합니다.")
        return

    raw_data = {}
    print(f"\n📥 [1단계] 풀이 데이터 수집 시작 (1,000명 기준 약 25분 소요됩니다)...")
    
    # 2. 확보한 유저 목록으로 태그 데이터 수집
    for i, handle in enumerate(users):
        stats_url = f"https://solved.ac/api/v3/user/problem_tag_stats?handle={handle}"
        
        retries = 3
        while retries > 0:
            try:
                stats_res = scraper.get(stats_url, timeout=10)
                if stats_res.status_code == 200:
                    raw_data[handle] = stats_res.json().get('items', [])
                    print(f"  - ({i+1}/{len(users)}) {handle} 수집 완료")
                    break
                elif stats_res.status_code == 429:
                    print(f"  - ({i+1}/{len(users)}) {handle} 너무 잦은 요청(429). 10초 대기...")
                    time.sleep(10)
                    retries -= 1
                else:
                    print(f"  - ({i+1}/{len(users)}) {handle} 수집 실패 ({stats_res.status_code})")
                    break
            except Exception as e:
                print(f"  - ({i+1}/{len(users)}) {handle} 에러, 재시도 중... (남은 횟수: {retries-1})")
                time.sleep(3)
                retries -= 1
                
        time.sleep(1.5) # 1인 수집 후 휴식 (필수)
        
        # 100명 단위 중간 저장
        if (i + 1) % 100 == 0:
            with open(get_file_path('raw_data.json'), 'w', encoding='utf-8') as f:
                json.dump(raw_data, f, ensure_ascii=False, indent=4)
            print(f"💾 [자동 저장] {i+1}명까지의 데이터를 안전하게 저장했습니다.")
            
    # 3. 최종 저장
    with open(get_file_path('raw_data.json'), 'w', encoding='utf-8') as f:
        json.dump(raw_data, f, ensure_ascii=False, indent=4)
    print("\n✅ [1단계 완료] 1,000명 분량의 진짜 데이터 수집이 완료되었습니다!\n")

if __name__ == "__main__":
    collect_raw_data(target_users_count=1000)