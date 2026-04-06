import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
from dirControl import *

def train_ai_model():
    print("🤖 AI 모델 학습을 시작합니다...")
    
    # 1. 전처리된 데이터 로드
    df = pd.read_csv(load_data('ml_ready_dataset.csv'))
    
    # 2. 학습에 사용할 특성(X)과 정답(y) 분리
    # github_id와 label(정답)을 제외한 모든 숫자 열이 특성이 됩니다.
    X = df.drop(columns=['github_id', 'label'])
    y = df['label']
    
    # 3. 데이터 분할 (학습용 80%, 검증용 20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. 랜덤 포레스트 모델 생성
    # n_estimators=100: 100개의 의사결정 나무를 만들어 투표합니다.
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # 5. 모델 학습
    model.fit(X_train, y_train)
    
    # 6. 모델 검증
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n=======================================")
    print(f"✅ 학습 완료! 모델 정확도: {accuracy:.2%}")
    print("=======================================\n")
    
    # 상세 리포트 출력 (어떤 직군을 잘 맞히고 못 맞히는지 확인)
    print("📊 상세 분석 리포트:")
    print(classification_report(y_test, y_pred))
    
    # 7. 완성된 모델 저장 (나중에 바로 써먹기 위함)
    joblib.dump(model, get_file_path('github_role_model.pkl'))
    # 학습에 사용된 컬럼 순서도 저장 (중요)
    joblib.dump(X.columns.tolist(), get_file_path('model_features.pkl'))
    
    print("💾 모델 저장이 완료되었습니다: [github_role_model.pkl]")

if __name__ == "__main__":
    train_ai_model()