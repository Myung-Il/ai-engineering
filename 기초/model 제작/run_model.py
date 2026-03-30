from my_model import MyLogisticRegression

# 학습 데이터
data = [

    # user_level, difficulty, solved
    (1, 1, 1),
    (1, 3, 0),
    (2, 2, 1),
    (3, 1, 1),
    (2, 4, 0),

]

model = MyLogisticRegression()

model.fit(data)

result = model.predict(2, 3)

print("예측 결과:", result)