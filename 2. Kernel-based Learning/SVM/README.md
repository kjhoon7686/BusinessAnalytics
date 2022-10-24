# SVM

# Support Vector Machine

### svm이란

svm은 서로 다른 class의 값을 분류하기 위해 사용하는 알고리즘으로 데이터의 값을 가장 잘 분류하는 hyperplane을 찾는 것을 목적으로 한다.

![image](https://user-images.githubusercontent.com/79893946/197450826-110fb8fc-b64e-4490-926b-97b0c04fd089.png)

위 그림에서 빨간색과 파란색이 각 데이터들의 class라고 하자. 이때

- decision boundary : 실선
- hyperplane : 점선
- margin : decision boundary와 한 hyperplane간의 거리
- support vector : 각 hyperplane에 존재하는 data

라고 할 수 있다.

svm의 목적은 data를 class에 따라 잘 분류하는 것이기 때문에 결정 경계와 hyperplane간의 거리인 margin을 최대화하는 것이 목적이 된다. 

### svm의 종류

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/31a610ac-31b1-47e6-9bdb-30b84052ccb4/Untitled.png)

다음 표에서 살펴볼 수 있듯이 3가지 경우를 살펴보게 될텐데 현실에서 모든 데이터가 정확하게 분류되는 것은 흔치 않기 때문에 hard margin을 사용하는 svm은 잘 사용하지 않고 case2와 case3를 많이 사용하게 된다. 

### case 1 : Linear and Hard margin

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/304605d7-0807-4629-812f-bfb714a89a8c/Untitled.png)

case1은 위 그림처럼 모든 데이터를 분류할 수 있는 hyperplane을 찾고 그 margin을 최대화하는 것을 목적으로 하는 svm이다.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/a271c958-7794-48cd-b2f0-a792caef7b7f/Untitled.png)

 

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/34e595cf-b9a1-460a-b6c7-dcd31b5fdace/Untitled.png)

margin이 w와 반비례하기 때문에  w를 최소화하는 것이 margin을 최대화하는 것이다. 따라서 목적식은 w를 최소화하는 것이고 제약조건은 위 그림과 같다.

위 목적식과 제약식을 사용하여 lagrangian problem으로 변형하고 kkt condition을 적용하여 dual problem으로 변환하면 다음의 분류함수를 얻을 수 있다.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/2235141f-49b2-4a81-8ac4-fd8d462eea0a/Untitled.png)

(자세한 증명은 고려대학교 산업공학과 강필성 교수님의 Business Analysis 강의를 참고)

### case 2 : Linear and Soft Margin

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f13f2c73-e6f2-45d0-8078-db363fab34a2/Untitled.png)

case 2는 case 1과 다르게 각각의 class 분류에 오차를 적용한 것을 위 그림을 통해 알 수 있다. 이때 $\xi$는 각 data가 margin을 벗어난 정도를 의미한다. 

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d28406d1-baf8-41b5-8570-c40434d3d1ee/Untitled.png)

따라서 목적함수는 case1의 목적함수에 penelty인  $\xi$와 이를 조절하는 hyperparameter인 C로 이루어진다. 또한 제약식에도 페널티가 부과된다.

목적함수 최적화를 통해 분류함수를 얻는 방법은 case1과 매우 비슷하다. 

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/25775cb5-3a43-4937-8d57-c17d825b6032/Untitled.png)

case2의 svm에서 penelty를 조절하는 c값에 따른 차이를 보면 c가 클수록 마진이 좁은 것을 알 수 있다.

### case 3 : Nonlinear and Soft Margin

case1과 case2는 모두 분류 경계면이 선형이기 때문에 분류 경계면이 비선형인 것이 적합할 경우 잘 분류할 수 없게 된다. 

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b831e417-20e7-4692-9bbe-c495651ed07e/Untitled.png)

따라서 case3은 위 그림과 같이 저차원의 데이터를 고차원으로 매핑하여 비선형 분류 경계면을 생성함으로써 유연성을 확보하고, margin을 최대화하여 일반화 성능을 확보하는 것을 목적으로 한다.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/bf39b8bd-0266-484a-8c47-8fae12568867/Untitled.png)

목적식과 제약식은 case2와 매우 유사하지만 데이터들을 바로 사용하는 것이 아니라 kernel 함수를 통해 고차원으로 mapping하여 사용한다. 따라서 크게보면 case2가 case3에 포함된다고 할 수 있다.

목적함수를 최적화하는 방법은 case2와 비슷하지만 kernel trick을 사용하여 kernel function을 찾아낸다는 점에서 차이가 있다. 

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/4c5d70b4-63f1-4156-a609-cbc5b31b24ed/Untitled.png)

다음의 함수들이 대표적으로 사용되는 kernel 함수들이다. 

## 코드

[SVM.py](http://SVM.py) - cvxopt 라이브러리를 사용해서 각 case의 svm을 파라미터에 따라 사용할 수 있도록 만든 함수이다. 

SVM.ipynb - c-svm과 nu-svm의 공통점과 차이점, kernel function과 hyperparameter의 변화에 따른 svm의 결과를 확인 할 수 있다.
