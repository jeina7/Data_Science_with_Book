# [PyTorch를 활용한 머신러닝, 딥러닝 철저 입문]
#### 저자 : 코이즈미 사토시 ||  출판사 : 위키북스



위 책을 통해 실습하며 공부한 내용을 정리하였습니다.  
PyTorch를 이용해서 신경망을 구현하며 분류 / 예측하는 실습으로 이루어져 있습니다. 주로 책의 코드를 따라했지만, 군데군데 제 임의로 수정하거나 추가한 부분도 있습니다.

---
## 1. [[Chapter 5-5] 예제: 와인 분류하기 (1)](https://nbviewer.jupyter.org/github/jeina7/Book_studying/blob/master/01_PyTorch_introduction/Chapter_5.5_%EC%99%80%EC%9D%B8%EB%B6%84%EB%A5%98%ED%95%98%EA%B8%B0%281%29.ipynb)
### \# Section : Classification

### \# Data Description
- Data Reference : `sklearn.datasets.load_wine`
- Total data : 130
- feature : 13
- train data : 104 (80%)
- test data : 26 (20%)
- Category : 2 (class_0, class_1)

### \# Model Description
- Layer : Linear(2)
- Loss Function : Cross Entropy
- Optimizer : SGD (lr=0.01)

### \# Training Description
- batch size : 16 (7 iteration = 1 epoch)
- epoch : 50 (350 iteration)
- runtime : 0.8s
- accuracy : 57.7%

---
## 2. [[Chapter 6-2] 예제: 와인 분류하기 (2)](https://nbviewer.jupyter.org/github/jeina7/Book_studying/blob/master/01_PyTorch_introduction/Chapter_6.2_%EC%99%80%EC%9D%B8%EB%B6%84%EB%A5%98%ED%95%98%EA%B8%B0%282%29.ipynb)
### \# Section : Classification

### \# Data Description
- Data Reference : `sklearn.datasets.load_wine`
- Total data : 130
- feature : 13
- train data : 104 (80%)
- test data : 26 (20%)
- Category : 2 (class_0, class_1)

### \# Model Description
- Layer : Linear(6)
- Loss Function : Cross Entropy
- Optimizer : SGD (lr=0.01)

### \# Training Description
- batch size : 16 (7 iteration = 1 epoch)
- epoch : 50 (350 iteration)
- runtime : 1.6s
- accuracy : 92.3%
> 두 층만 썼던 1번의 예제에서 6층으로 늘렸을 때 정확도가 크게 오른 것을 확인할 수 있다.

---
## 3. [[Chapter 6-3] 예제: 손글씨 이미지 분류 (1)](https://nbviewer.jupyter.org/github/jeina7/Book_studying/blob/master/01_PyTorch_introduction/Chapter_6.3_%EC%86%90%EA%B8%80%EC%94%A8%EC%9D%B4%EB%AF%B8%EC%A7%80%EB%B6%84%EB%A5%98%281%29.ipynb)
### \# Section : Classification

### \# Data Description
- Data Reference : MNIST digit data
- Total data : 70000
- feature : 784 (=28*28)
- train data : 5000 (90%)
- test data : 500 (10%)
- Category : 10 (number of 0~9)

### \# Model Description
- Layer : Linear(6)
- Loss Function : Cross Entropy
- Optimizer : SGD (lr=0.01)

### \# Training Description
- batch size : 100 (50 iteration = 1 epoch)
- epoch : 20 (1000 iteration)
- runtime : 3m 47s
- accuracy : 90.6%

---
## 4. [[Chapter 6-4] 예제: 뉴스 기사 분류](https://nbviewer.jupyter.org/github/jeina7/Book_studying/blob/master/01_PyTorch_introduction/Chapter_6.4_%EB%89%B4%EC%8A%A4%EA%B8%B0%EC%82%AC%EB%B6%84%EB%A5%98.ipynb)
### \# Section : Classification, NLP

### \# Data Description
- Data Refernce :[한국 뉴스 기사](http://kristalinfo.com/TestCollections/#hkib)
- Total data : 2521
- feature : unknown (word tokens)
- train data : 2016 (80%)
- test data : 505 (20%)
- Category : 5 (교육, 건강, 여가, 정치, 과학)

### \# Model Description
- Layer : Linear(6)
- Loss Function : Cross Entropy
- Optimizer : Adam (lr=0.005)

### \# Training Description
- batch size : 200 (100 iteration = 1 epoch)
- epoch : 0.5 (50 iteration)
- runtime : 7m 27s
- accuracy : 72.3%

---
## 5. [[Chapter 6-5] 예제: 시계열데이터 - 이상기온 탐지](https://nbviewer.jupyter.org/github/jeina7/Book_studying/blob/master/01_PyTorch_introduction/Chapter_6.5_%EC%8B%9C%EA%B3%84%EC%97%B4%EB%8D%B0%EC%9D%B4%ED%84%B0_%EC%9D%B4%EC%83%81%EA%B8%B0%EC%98%A8%ED%83%90%EC%A7%80.ipynb)
### \# Section : Regression, Time series

### \# Data Description
- Data Refernce :[기상청 - 서울기온](http://kristalinfo.com/TestCollections/#hkib)
- Total data : 2191 (6 years)
- feature : 1 (temperature)
- train data : 1461 (4 years)
- test data : 730 (2 years)

### \# Model Description
- Model : Auto Encoder
- Layer : Linear(4)
- Loss Function : MSE Loss
- Optimizer : Adam (lr=0.001)

### \# Training Description
- batch size : 100 (12 iteration = 1 epoch)
- epoch : 500 (60000 iteration)
- runtime : 12.6s
- Final loss (mse) : 1.8

---
## 6. [[Chapter 7-2] 예제: 손글씨 이미지 분류 (2)](https://nbviewer.jupyter.org/github/jeina7/Book_studying/blob/master/01_PyTorch_introduction/Chapter_7.2_%EC%86%90%EA%B8%80%EC%94%A8%EC%9D%B4%EB%AF%B8%EC%A7%80%EB%B6%84%EB%A5%98%282%29.ipynb)
### \# Section : Classification, CNN

### \# Data Description
- Data Reference : MNIST digit data
- Total data : 70000
- feature : 784 (=28*28)
- train data : 5000 (90%)
- test data : 500 (10%)
- Category : 10 (number of 0~9)

### \# Model Description
- Layer : Conv2d(2), Linear(2)
- Loss Function : Cross Entropy
- Optimizer : SGD (lr=0.01)

### \# Training Description
- batch size : 100 (50 iteration = 1 epoch)
- epoch : 20 (1000 iteration)
- runtime : 15m 47s
- accuracy : 96.6%
> CNN을 사용하지 않은 손글씨이미지분류 (1) 에서보다 약 6% 오른 것을 확인할 수 있다.

---
## 7. [[Chapter 7-3] 예제: 옷 이미지 분류](https://nbviewer.jupyter.org/github/jeina7/Book_studying/blob/master/01_PyTorch_introduction/Chapter_7.3_%EC%98%B7%EC%9D%B4%EB%AF%B8%EC%A7%80%EB%B6%84%EB%A5%98.ipynb)
### \# Section : Classification, CNN

### \# Data Description
- Data Reference : FASION-MNIST digit data
- Total data : 60000
- feature : 784 (=28*28)
- train data : 5000 (90%)
- test data : 500 (10%)
- Category : 10 (T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)

### \# Model Description
- Layer : Conv2d(2), Linear(2)
- Loss Function : Cross Entropy
- Optimizer : Adam (lr=0.001)

### \# Training Description
- batch size : 100 (50 iteration = 1 epoch)
- epoch : 20 (1000 iteration)
- runtime : 16m 32s
- accuracy : 83.6%

---
## 8. [[Chapter 7-4] 예제: ants & bees 이미지 분류](https://nbviewer.jupyter.org/github/jeina7/Book_studying/blob/master/01_PyTorch_introduction/Chapter_7.4_ants_bees_%EC%9D%B4%EB%AF%B8%EC%A7%80%EB%B6%84%EB%A5%98.ipynb)
### \# Section : Classification, CNN

### \# Data Description
- Data Refernce :[Pytorch tutorial image](https://download.pytorch.org/tutorial/hymenoptera.zip)
- Total data : 397
- feature : 128 * 128 * 3 (rgb)
- train data : 357 (90%)
- test data : 40 (10%)

### \# Model Description
- Layer : Conv2d(2), Linear(2)
- Loss Function : Cross Entropy
- Optimizer : Adam (lr=0.001)

### \# Training Description
- batch size : 32 (10 iteration = 1 epoch)
- epoch : 30 (300 iteration)
- runtime : 18m 44s
- accuracy : 70%
