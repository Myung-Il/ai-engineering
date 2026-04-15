import json
from collections import defaultdict
from dirControl import *

def train_model():
    print("🧠 [3단계] 전처리된 데이터로 AI 가중치 모델 학습 시작...")
    
    try:
        with open(get_file_path('processed_data.json'), 'r', encoding='utf-8') as f:
            processed_data = json.load(f)
    except FileNotFoundError:
        print("❌ processed_data.json 파일이 없습니다.")
        return

    # 신경망 가중치 그래프 초기화
    graph = defaultdict(lambda: defaultdict(float))
    
    # 동시 발생(Co-occurrence) 기반 협업 필터링 학습
    for handle, tags in processed_data.items():
        # 한 유저가 마스터한 태그들끼리 서로 선(Edge)을 연결하고 두께(Weight) 증가
        for tag1 in tags:
            for tag2 in tags:
                if tag1 != tag2:
                    graph[tag1][tag2] += 1.0 # 함께 푼 빈도수 증가
                    
    # 모델 정규화 (선택사항, 가중치를 비율로 변환)
    for tag1 in graph:
        total_weight = sum(graph[tag1].values())
        for tag2 in graph[tag1]:
            graph[tag1][tag2] = round((graph[tag1][tag2] / total_weight) * 100, 2)

    with open(get_file_path('model_weights.json'), 'w', encoding='utf-8') as f:
        json.dump(graph, f, ensure_ascii=False, indent=4)
        
    print("✅ [3단계 완료] 태그 간 연관성 분석 끝 -> model_weights.json (모델) 생성 완료\n")

if __name__ == "__main__":
    train_model()