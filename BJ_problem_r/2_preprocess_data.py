import json
from dirControl import *

def preprocess_data():
    print("🧹 [2단계] 수집된 Raw Data 전처리 시작...")
    
    try:
        with open(get_file_path('raw_data.json'), 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print("❌ raw_data.json 파일이 없습니다.")
        return

    processed_data = {}
    
    for handle, items in raw_data.items():
        # 전처리 기준: 5문제 이상 푼 태그만 '유의미한 스킬'로 인정
        mastered_tags = []
        for item in items:
            if item.get('solved', 0) >= 5:
                mastered_tags.append(item['tag']['key'])
        
        # 유의미한 태그가 3개 이상인 유저만 데이터로 사용 (노이즈 제거)
        if len(mastered_tags) >= 3:
            processed_data[handle] = mastered_tags

    with open(get_file_path('processed_data.json'), 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)
        
    print(f"✅ [2단계 완료] {len(processed_data)}명의 유효 데이터 추출 -> processed_data.json 저장 완료\n")

if __name__ == "__main__":
    preprocess_data()