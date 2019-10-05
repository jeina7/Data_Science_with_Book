# 밑바닥부터 시작하는 딥러닝 2
#### 저자 : 사이토 고키 ||  출판사 : 한빛미디어


---
## Ch.1 신경망 복습
#### \# Keywords   
`#벡터와행렬` `#신경망` `#완전연결계층` `#순전파` `#역전파` `#손실함수` `#연쇄법칙` `#가중치갱신` `#GPU`   


`[밑바닥부터 시작하는 딥러닝 1]`에서 자세히 설명한 신경망 내용들을 조금 간추려서 설명하고, 무엇보다 신경망의 완전연결계층 / 활성화 함수 등을 직접 구현해볼 수 있습니다. 밑러닝 1권을 안보셨다면 이 단원에서 신경망에 대한 전체적인 내용을 간략하지만 자세히 훑어볼 수 있고, 1권을 보셨다고 해도 다시 복습 & 정리하는 데에 도움이 많이 됩니다.


- 코드 활용 & 실습 → [click!](https://github.com/jeina7/Book_studying/blob/master/02_DeepLearning-from-scratch-2/%5BChap.1%5D%20Layers%2C%20TwoLayerNet%2C%20Trainer.ipynb)   

ㅤㅤㅤ　




## Ch.2 자연어와 단어의 분산 표현
#### \# Keywords   
`#자연어처리` `#시소러스` `#통계기반기법` `#동시발생행렬` `#WordNet` `#유사도` `#PTB데이터셋` `#말뭉치`   


머신러닝/딥러닝 세계에서의 자연어에 처음으로 들어가는 장입니다. 앞으로 나올 복잡하고 어려운 자연어 세상에 들어가기 전에 가장 기본적인 개념을 이해하고 넘어갈 수 있도록 도와주는 내용들로 구성되어 있습니다. 직접 10000단어로 이루어진 **`PTB 데이터셋`** 을 이용해서 말뭉치 생성, 동시발생 행렬 생성, 차원축소 등을 모두 실습할 수 있는 코드도 있으니 여러 실험을 해 보시길 추천합니다.


- 코드 활용 & 실습 → [click!](https://github.com/jeina7/Book_studying/blob/master/02_DeepLearning-from-scratch-2/%5BChap.2%5D%20corpus%2C%20co-occurence%20matrix%2C%20similarity%2C%20ppmi%2C%20visualize_2D.ipynb)   

ㅤㅤㅤ　




## Ch.3 word2vec
#### \# Keywords   
`#추론기반기법` `#word2vec` `#CBOW` `#skip-gram` `#분산표현` `#원핫인코딩` `#음의로그가능도`   


word2vec 개념에 대해 처음 배워보는 장입니다! 앞서 배운 통계기반 기법과 추론기반 기법 두 가지 방법의 장단점에 대해 비교해보고, word2vec의 모델인 `CBOW`와 `skip-gram` 모델 두 가지를 설명합니다. 이 중 특히 CBOW 모델은 아주 간단한 데이터셋으로 직접 구현해보고, 훈련까지 시켜서 손실함수가 어떻게 떨어지는지까지 확인해봅니다.


- 코드 활용 & 실습 → [click!](https://github.com/jeina7/Book_studying/blob/master/02_DeepLearning-from-scratch-2/%5BChap.3%5D%20Training%20Simple%20CBOW%20and%20get%20WordVector.ipynb)

ㅤㅤㅤ　



ㅤ　

...이 뒤는 아직 공부중...




## Ch.4 word2vec 속도 개선


ㅤㅤㅤ　




## Ch.5 순환 신경망 (RNN)

ㅤㅤㅤ　




## Ch.6 게이트가 추가된 RNN

ㅤㅤㅤ　




## Ch.7 RNN을 사용한 문장 생성

ㅤㅤㅤ　




## Ch.8 어텐션
