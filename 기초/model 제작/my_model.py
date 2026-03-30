import math
import random

class MyLogisticRegression:

    def __init__(self, learning_rate=0.1, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs

        # 초기 가중치
        self.w1 = random.random()
        self.w2 = random.random()
        self.b = random.random()

    # sigmoid 함수
    def sigmoid(self, z):
        return 1 / (1 + math.exp(-z))

    # 예측
    def predict_proba(self, x1, x2):

        z = self.w1 * x1 + self.w2 * x2 + self.b

        return self.sigmoid(z)

    # 학습
    def fit(self, data):

        for _ in range(self.epochs):

            for x1, x2, y in data:

                # 예측
                pred = self.predict_proba(x1, x2)

                # 오차
                error = pred - y

                # 가중치 업데이트
                self.w1 -= self.lr * error * x1
                self.w2 -= self.lr * error * x2
                self.b -= self.lr * error

    # 결과 예측 (0 or 1)
    def predict(self, x1, x2):

        prob = self.predict_proba(x1, x2)

        if prob >= 0.5:
            return 1
        else:
            return 0