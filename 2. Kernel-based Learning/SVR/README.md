# SVR

# Support Vector Regression

### svr이란

svr은 svm의 regression 버전으로 svm에서의 hyperplane은 data를 가장 잘 분류하는 역할을 하지만, svr에서의 hyperplane은 data를 가장 잘 표현하는 역할을 한다는 차이가 있다. classification과 regression의 차이이다. 

![image](https://user-images.githubusercontent.com/79893946/197459417-38b609ca-9990-41cc-9a47-cb622d72198b.png)

이번 tutorial에서는 가장 많이 사용되는 svr 방법 중에 하나인 epsilon-svr에 대해서 살펴보도록 하겠다.

![image](https://user-images.githubusercontent.com/79893946/197459457-7b890774-8469-4d94-80a3-aab52b62d016.png)

(C)가 epsilon-svr의 그림이다. epsilon-svr은 기본적으로 데이터에 노이즈가 존재한다고 가정하기 때문에 적합된 회귀선 위아래에 epsilon만큼을 더하고 뺀 부분에서는 penelty를 부과하지 않는다. 적합된 회귀선에 epsilon을 더하고 뺀 부분을 epsilon-tube라고 한다. 다시 말하면 epsilon-tube 내의 데이터에는 penelty를 부과하지 않고 epsilon-tube 바깥에 있는 데이터에만 tube에서 데이터의 거리만큼 penelty를 부과한다. 이는 (D)의 loss function에서도 확인할 수 있다.

![image](https://user-images.githubusercontent.com/79893946/197459502-4ef938a8-979a-4f3d-9ee9-625c6c47371a.png)

epsilon-svr에서의 목적함수는 선형 회귀식이 loss function은 w를 최소화함으로써 general한 함수를 만들고, hinge loss라고도 불리는 오차의 합을 최소화함으로써 fitting의 적합도를 올리는 것을 목적으로 한다.

![image](https://user-images.githubusercontent.com/79893946/197459531-0ffb5499-9b8c-4a21-ae62-d36662adf5bb.png)

이는 ridge regression의 목적식과도 유사하다. svr에서는 w를 최소화하는 것으로써 simple한 함수를 fitting한다면 ridge regression에서는 회귀계수들에 penelty를 줌으로써 general한 함수를 fitting한다. 또한 svr에서는 hinge loss의 c를 통해 예측오차를 줄인다면 ridge regression에서는 적합된 회귀식과 실제값과의 차이를 최소화하는 것을 통해 예측오차를 줄인다. 

### svr optimization

![image](https://user-images.githubusercontent.com/79893946/197459572-58bf06d7-c0df-48e4-b2f8-34a8e4565c8b.png)

먼저 다음의 loss function에서 lagrangian multiplier를 사용해서 lagrangian primal problem을 만든다.

![image](https://user-images.githubusercontent.com/79893946/197459748-c81674d0-5e9c-4132-ba5d-a41a154cebc8.png)

다음으로 편미분을 통해 (1), (2), (3)의 식을 얻는다.

![image](https://user-images.githubusercontent.com/79893946/197459800-c563576b-f6cd-462f-a0a8-3c7776489896.png)

앞서 구한 식을  lagrangian primal problem에 대입함으로써 dual problem으로 바꾸어 준다,

![image](https://user-images.githubusercontent.com/79893946/197459822-43f9ca65-cb98-48f1-92cc-e4d0d14b6f65.png)

이를 통해 decision function, 즉 회귀식을 적합할 수 있다. 

### kernel trick

![image](https://user-images.githubusercontent.com/79893946/197459845-951be530-6159-4786-99d5-b71d37bfe0be.png)

svm에서와 마찬가지로 비선형적인 데이터 분포를 fitting하기 위해 저차원의 데이터를 고차원의 데이터로 kernel function을 통해 mapping하여 svr을 진행할 수 있다. 이때 사용하는 방법이 kernel trick이다

![image](https://user-images.githubusercontent.com/79893946/197459868-9fc8a2fd-cf0b-4f65-bd58-39c18b3cab7e.png)

kernel trick을 사용하여 optimization을 하면 앞서 설명한 svr에서 kernel function을 사용하는 것 이외에는 매우 유사한 방법을 사용한다.

![image](https://user-images.githubusercontent.com/79893946/197459895-8ae2d059-c1a0-41b5-9b14-8639fb0d93aa.png)

svr은 loss function에 따라 그 종류가 정의되며 다양하다. 

![image](https://user-images.githubusercontent.com/79893946/197459930-7d750533-70eb-4eb0-8939-d52b9f9e44e7.png)

다음은 svr의 loss functon의 따른 회귀선인데 비선형적인 데이터들을 kernel svr들이 더 잘 fitting하는 것을 확인할 수 있다. 

### 코드

[SVR.py](http://SVR.py) - epsilon-svr을 cvxopt library를 통해 구현한 것이다. 

SVR.ipynb - data를 생성하고 svr의 종류와 hyperparameter에 따라 어떻게 fitting하는지를 보여준다. 

reference

[https://leejiyoon52.github.io/Support-Vecter-Regression/](https://leejiyoon52.github.io/Support-Vecter-Regression/)
