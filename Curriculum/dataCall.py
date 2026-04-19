from datasets import load_dataset

# 1. 데이터 로드
print("데이터셋을 불러오는 중입니다...")
ds = load_dataset("recuse/synthetic_resume_jd_raw_dataset")
train_data = ds['train']

# 2. 3개의 빈 집합(Set) 생성 (자동으로 중복을 제거해 줍니다)
jobs = set()
levels = set()
companies = set()

# 3. 데이터를 순회하며 '_'를 기준으로 분리하여 각각의 집합에 추가
for curr_type in train_data['curr_type_str']:
    # 예: 'Software Engineer_Junior_Google' -> ['Software Engineer', 'Junior', 'Google']
    parts = curr_type.split('_')
    
    # 데이터가 정상적으로 3부분으로 나뉘었는지 확인 후 추가
    if len(parts) == 3:
        jobs.add(parts[0])
        levels.add(parts[1])
        companies.add(parts[2])

# 4. 보기 좋게 정렬하여 결과 출력
print("\n[ 고유 직업(Job) 목록 ]")
for job in sorted(jobs):
    print(f"- {job}")

print("\n[ 고유 요구 수준(Level) 목록 ]")
for level in sorted(levels):
    print(f"- {level}")

print("\n[ 고유 회사(Company) 목록 ]")
for company in sorted(companies):
    print(f"- {company}")