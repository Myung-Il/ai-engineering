import os
import requests
from dotenv import load_dotenv

load_dotenv()

# 1. .env에서 토큰들을 가져와 리스트로 만듭니다.
tokens_string = os.getenv('GITHUB_TOKENS', '')
token_list = [t.strip() for t in tokens_string.split(',') if t.strip()]

# 최종적으로 predict_user.py에 넘겨줄 변수
GITHUB_TOKEN = None

if not token_list:
    print("❌ 에러: .env 파일에서 GITHUB_TOKENS를 찾을 수 없습니다!")
else:
    print("🔍 사용 가능한 토큰을 스캔 중입니다...")
    
    # 2. 토큰 리스트를 돌면서 한도가 남은 녀석을 찾습니다.
    for index, token in enumerate(token_list):
        headers = {'Authorization': f'token {token}'}
        
        # 해당 토큰의 한도 상태를 확인합니다.
        res = requests.get('https://api.github.com/rate_limit', headers=headers)
        
        if res.status_code == 200:
            data = res.json()
            remaining = data['rate']['remaining']
            
            # 3. 남은 횟수가 0보다 크면 이 토큰으로 낙점!
            if remaining > 0:
                GITHUB_TOKEN = token
                print(f"✅ {index + 1}번째 토큰 채택 완료! (남은 횟수: {remaining}회)\n")
                break
            else:
                print(f"⚠️ {index + 1}번째 토큰 한도 초과. 다음 토큰을 확인합니다...")
        else:
            print(f"⚠️ {index + 1}번째 토큰 오류 (상태 코드: {res.status_code})")

    # 4. 모든 토큰이 다 죽어있을 경우
    if not GITHUB_TOKEN:
        print("❌ 모든 토큰의 한도가 초과되었거나 유효하지 않습니다. 1시간 뒤에 시도하세요.\n")