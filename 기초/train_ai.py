import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

# 샘플 데이터 (나중에 DB 데이터로 교체)
data = {
    "user_level": [1, 1, 2, 3, 2],
    "problem_difficulty": [1, 3, 2, 1, 4],
    "solved": [1, 0, 1, 1, 0]
}

df = pd.DataFrame(data)

X = df[["user_level", "problem_difficulty"]]
y = df["solved"]

model = LogisticRegression()
model.fit(X, y)

joblib.dump(model, "model.pkl")

print("모델 생성 완료")