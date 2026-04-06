import pandas as pd
import ast
from dirControl import *

def preprocess_data():
    print("🧹 데이터 전처리를 시작합니다...")
    
    # 1. 수집된 데이터 불러오기
    df = pd.read_csv(load_data("github_dataset.csv"))
    original_count = len(df)
    
    # 2. 노이즈 데이터 제거 ('X (추론 안됨)' 필터링)
    # 정답지가 없는 데이터는 학습에 방해되므로 버립니다.
    df = df[~df['target_role'].str.startswith('X')]
    print(f"✔️ 유효 데이터 필터링: {original_count}개 ➔ {len(df)}개")

    # 3. 모델 학습용 '정답(Label)' 추출
    # "Web Backend (Lv2. 강한 의심..." -> "Web Backend" 만 깔끔하게 잘라냅니다.
    df['label'] = df['target_role'].apply(lambda x: x.split(' (')[0].strip())

    # 4. 문자열로 된 사전(Dict)과 리스트(List)를 파이썬 객체로 변환
    df['top_languages'] = df['top_languages'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else {})
    df['topics'] = df['topics'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])

    # 5. 언어(Languages)를 개별 숫자로 쪼개기 (특성 추출)
    # {'JavaScript': 7, 'Vue': 1} -> lang_JavaScript: 7, lang_Vue: 1
    lang_df = df['top_languages'].apply(pd.Series).fillna(0)
    lang_df = lang_df.add_prefix('lang_') # 열 이름에 lang_ 붙이기

    # 6. 토픽(Topics)을 개별 숫자로 쪼개기 (One-Hot Encoding)
    # ['react', 'api'] -> topic_react: 1, topic_api: 1
    topic_df = df['topics'].str.join('|').str.get_dummies()
    topic_df = topic_df.add_prefix('topic_') # 열 이름에 topic_ 붙이기

    # 7. 원래의 데이터프레임과 쪼개진 숫자 데이터 합치기
    # 불필요해진 원본 문자열 칼럼은 삭제합니다.
    ml_df = pd.concat([df[['github_id', 'label', 'total_repos']], lang_df, topic_df], axis=1)

    # 8. 최종 결과 저장
    ml_df.to_csv(get_file_path("ml_ready_dataset.csv"), index=False, encoding='utf-8-sig')
    print("🎉 전처리 완료! [ml_ready_dataset.csv] 파일이 생성되었습니다.")
    print(f"📊 최종 데이터 형태: {ml_df.shape[0]}행, {ml_df.shape[1]}열")

if __name__ == "__main__":
    preprocess_data()