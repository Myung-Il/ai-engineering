import os
from dotenv import load_dotenv

# .env 파일의 내용을 로드합니다.
load_dotenv()

# 환경 변수에서 GITHUB_TOKEN을 가져옵니다.
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')

# 확인용 (토큰이 잘 불러와졌는지)
if not GITHUB_TOKEN:
    print("❌ 에러: .env 파일에서 토큰을 찾을 수 없습니다!")