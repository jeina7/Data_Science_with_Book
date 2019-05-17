# Data Science with Book
#### 책으로 공부하는 Machine Learning / Deep Learning 을 담습니다.

##  \# Book List
#### 1. [PyTorch를 활용한 머신러닝, 딥러닝 철저 입문](https://github.com/jeina7/Book_studying#1-pytorch%EB%A5%BC-%ED%99%9C%EC%9A%A9%ED%95%9C-%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-%EB%94%A5%EB%9F%AC%EB%8B%9D-%EC%B2%A0%EC%A0%80-%EC%9E%85%EB%AC%B8-1)
#### 2. [밑바닥부터 시작하는 딥러닝 2](https://github.com/jeina7/Book_studying#2-%EB%B0%91%EB%B0%94%EB%8B%A5%EB%B6%80%ED%84%B0-%EC%8B%9C%EC%9E%91%ED%95%98%EB%8A%94-%EB%94%A5%EB%9F%AC%EB%8B%9D-2-1)
#### 3. [핸즈온 머신러닝]()



---
## 1. PyTorch를 활용한 머신러닝, 딥러닝 철저 입문
(Book → [here](https://wikibook.co.kr/pytorch/))
#### \# 저자 : 코이즈미 사토시 ||  출판사 : 위키북스
#### \# 날짜 : 2019, 2월
#### \# 후기
PyTorch를 입문하면서 다양한 예제들을 가볍게 훑고 넘어가기에 참 좋은 책입니다.   
단순한 분류 / CNN 등의 예제만 있는 게 아니라, 자연어, 시계열 등의 여러가지 데이터를 다루는 예제가 있어서 좋았습니다.
예제 위주로 구성되어 있는 책인만큼, 코드를 꼭 직접 따라쳐보면서 읽어보시는 걸 추천합니다.
#### \# 예제 모델 / 성능 확인 : [Here!](https://github.com/jeina7/Book_studying/tree/master/01_PyTorch_introduction#pytorch%EB%A5%BC-%ED%99%9C%EC%9A%A9%ED%95%9C-%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-%EB%94%A5%EB%9F%AC%EB%8B%9D-%EC%B2%A0%EC%A0%80-%EC%9E%85%EB%AC%B8)
#### \# Overview
1. [[Chapter 5-5] 와인 분류하기 (1)](https://github.com/jeina7/Book_studying/blob/master/01_PyTorch_introduction/Chapter_5.5_%EC%99%80%EC%9D%B8%EB%B6%84%EB%A5%98%ED%95%98%EA%B8%B0(1).ipynb)  
sklearn의 와인 데이터의 분류 문제로 간단한 Two Layer Net을 구현하는 예제입니다.  
파이토치를 이용한 신경망 코드 구성을 직관적으로 이해할 수 있습니다.

1. [[Chapter 6-2] 와인 분류하기 (2)](https://github.com/jeina7/Book_studying/blob/master/01_PyTorch_introduction/Chapter_6.2_%EC%99%80%EC%9D%B8%EB%B6%84%EB%A5%98%ED%95%98%EA%B8%B0(2).ipynb)  
1번 예제에서 layer 개수를 6층으로 늘려서 깊은 층을 사용한 신경망의 성능을 알아보고, 두 층만 사용한 신경망과 비교합니다.

1. [[Chapter 6-3] 손글씨 이미지 분류 (1)](https://github.com/jeina7/Book_studying/blob/master/01_PyTorch_introduction/Chapter_6.3_%EC%86%90%EA%B8%80%EC%94%A8%EC%9D%B4%EB%AF%B8%EC%A7%80%EB%B6%84%EB%A5%98(1).ipynb)  
MNIST 데이터를 완전연결층으로 이루어진 신경망으로 분류해보는 예제입니다.  
CNN을 쓰지 않고 Linear한 데이터로써 신경망에 입력했을 때의 성능을 알아봅니다.

1. [[Chapter 6-4] 뉴스 기사 분류](https://github.com/jeina7/Book_studying/blob/master/01_PyTorch_introduction/Chapter_6.4_%EB%89%B4%EC%8A%A4%EA%B8%B0%EC%82%AC%EB%B6%84%EB%A5%98.ipynb)  
한국 뉴스기사 데이터를 직접 전처리해보고, 주제별로 분류해보는 예제입니다.  
개인적으로 raw한 자연어 데이터를 전처리하는 과정에서 RegEx를 사용하는 등여러가지 시도를 해 보면서 많이 배웠던 것 같습니다.  
자연어 패키지는 `Kkma` (꼬꼬마) 라이브러리를 사용하고, TF-IDF 기법을 이용합니다.

1. [[Chapter 6-5] 시계열데이터 - 이상기온 탐지](https://github.com/jeina7/Book_studying/blob/master/01_PyTorch_introduction/Chapter_6.5_%EC%8B%9C%EA%B3%84%EC%97%B4%EB%8D%B0%EC%9D%B4%ED%84%B0_%EC%9D%B4%EC%83%81%EA%B8%B0%EC%98%A8%ED%83%90%EC%A7%80.ipynb)  
시계열데이터인 기온 데이터를 이용해서 이상탐지 (Anomaly Detection)을 진행해봅니다.  
Auto Encoder (자동 부호화기) 기법을 이용합니다.

1. [[Chapter 7-2] 손글씨 이미지 분류 (2)](https://github.com/jeina7/Book_studying/blob/master/01_PyTorch_introduction/Chapter_7.2_%EC%86%90%EA%B8%80%EC%94%A8%EC%9D%B4%EB%AF%B8%EC%A7%80%EB%B6%84%EB%A5%98(2).ipynb)  
드디어 CNN을 다루는 예제입니다!  
간단한 CNN 모델을 만들어보고, 위의 Linear (완전연결계층)을 이용한 모델의 성능과 비교해봅니다.

1. [[Chapter 7-3] 옷 이미지 분류](https://github.com/jeina7/Book_studying/blob/master/01_PyTorch_introduction/Chapter_7.3_%EC%98%B7%EC%9D%B4%EB%AF%B8%EC%A7%80%EB%B6%84%EB%A5%98.ipynb)  
이번엔 Fashion-MNIST 데이터를 이용해서 CNN 분류 문제를 풀어봅니다.  

1. [[Chapter 7-4] ants & bees 이미지 분류](https://github.com/jeina7/Book_studying/blob/master/01_PyTorch_introduction/Chapter_7.4_ants_bees_%EC%9D%B4%EB%AF%B8%EC%A7%80%EB%B6%84%EB%A5%98.ipynb)  
마지막으로 데이터 크기가 큰 (128 x 128) 사진을 이용해서 분류 문제를 풀어봅니다.  
성능을 더 올리고 싶다면 더 깊은 신경망을 구성해 볼 수 있습니다!

---

## 2. 밑바닥부터 시작하는 딥러닝 2
(Book → [here](http://www.hanbit.co.kr/store/books/look.php?p_code=B8950212853))  
#### \# 저자 : 사이토 고키 ||  출판사 : 한빛미디어
#### \# 날짜 : 2019, 5월

...현재 공부중...

---

## 3. 핸즈온 머신러닝
(Book → [here](http://www.hanbit.co.kr/store/books/look.php?p_code=B9267655530))
#### \# 저자 : 오렐리앙 제롱 || 번역 : 박해선 || 출판사 : 한빛미디어
#### \# 날짜 : 2019, 5월

...현재 공부중...
