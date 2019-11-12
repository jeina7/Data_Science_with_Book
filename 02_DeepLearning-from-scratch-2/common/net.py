# 여러가지 종류의 Net을 딥러닝 라이브러리 없이 구현

# Net class의 구현 규칙:
# 1) 모든 Net은 함수 predict()를 가진다
# 2) 모든 Net은 그 안에 포함된 layers의 params를 모두 합친 Net의 새로운 변수 params를 가진다

from .np import *
from .layer import Sigmoid, Affine, SoftmaxWithLoss, MatMul
from .layer import Embedding, NegativeSamplingLoss


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size, self.hidden_size, self.output_size = input_size, hidden_size, output_size

        # weight, bias initialize
        W1 = np.random.randn(self.input_size, self.hidden_size)
        b1 = np.random.randn(self.hidden_size)
        W2 = np.random.randn(self.hidden_size, self.output_size)
        b2 = np.random.randn(self.output_size)

        # layers
        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2)
        ]

        # loss layer
        self.loss_layer = SoftmaxWithLoss()

        # collect params
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads


    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x


    def forward(self, x, t):
        score = self.predict(x)
        loss = self.loss_layer.forward(score, t)
        return loss


    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout


class SimpleCBOW:
    def __init__(self, vocab_size, hidden_size):
        V, H = vocab_size, hidden_size

        # 가중치 초기화
        W_in = 0.01 * np.random.randn(V, H).astype('f')
        W_out = 0.01 * np.random.randn(H, V).astype('f')

        # 계층 생성
        self.in_layer0 = MatMul(W_in)
        self.in_layer1 = MatMul(W_in)
        self.out_layer = MatMul(W_out)
        self.loss_layer = SoftmaxWithLoss()

        # 가중치와 기울기 한 리스트에 모으기
        layers = [self.in_layer0, self.in_layer1, self.out_layer]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        # 단어의 분산벡터는 in 가중치로 이용
        self.word_vecs = W_in


    def forward(self, contexts, target):
        h0 = self.in_layer0.forward(contexts[:, 0])
        h1 = self.in_layer1.forward(contexts[:, 1])
        h = (h0 + h1) / 2
        score = self.out_layer.forward(h)
        loss = self.loss_layer.forward(score, target)
        return loss


    def backward(self, dout=1):
        ds = self.loss_layer.backward(dout)
        da = self.out_layer.backward(ds)
        da *= 0.5
        self.in_layer0.backward(da)
        self.in_layer1.backward(da)
        return None


class CBOW:
    def __init__(self, vocab_size, hidden_size, window_size, corpus):
        V, H = vocab_size, hidden_size

        # 가중치 초기화
        W_in = 0.01 * np.random.randn(V, H).astype('f')
        W_out = 0.01 * np.random.randn(V, H).astype('f')

        # 계층 생성
        self.in_layers = []
        for i in range(2 * window_size):
            layer = Embedding(W_in)
            self.in_layers.append(layer)
        self.ns_loss = NegativeSamplingLoss(W_out, corpus, power=0.75, sample_size=5)
        layers = self.in_layers + [self.ns_loss]

        # 모든 가중치와 기울기 모으기
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        # 단어의 분산표현 저장
        self.word_vecs = W_in


    def forward(self, contexts, target):
        h = 0
        for i, layer in enumerate(self.in_layers):
            h += layer.forward(contexts[:, i])
        h *= 1 / len(self.in_layers)
        loss = self.ns_loss.forward(h, target)
        return loss


    def backward(self, dout=1):
        dout = self.ns_loss.backward(dout)
        dout *= 1 / len(self.in_layers)
        for layer in self.in_layers:
            layer.backward(dout)
        return None
