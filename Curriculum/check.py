from datasets import load_dataset
from collections import defaultdict, Counter

ds = load_dataset("recuse/synthetic_resume_jd_raw_dataset")
train_data = ds['train']

# 기업별 (직업, 레벨) 빈도
company_data = defaultdict(Counter)

for row in train_data:
    parts = row['curr_type_str'].split('_')
    company = parts[-1]
    level = parts[-2]
    job = ' '.join(parts[:-2])
    company_data[company][(job, level)] += 1

# 출력
for company in sorted(company_data):
    print(f"\n=== {company} ===")
    for (job, level), cnt in sorted(company_data[company].items()):
        print(f"  {job} / {level}: {cnt}건")