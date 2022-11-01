# Kernel-based Learning

# Support Vector Machine

## 개념

### SVM이란

svm은 서로 다른 class의 값을 분류하기 위해 사용하는 알고리즘으로 데이터의 값을 가장 잘 분류하는 hyperplane을 찾는 것을 목적으로 한다.

![image](https://user-images.githubusercontent.com/79893946/199164082-89827120-ebe0-48c8-8b3a-6fe23c51af62.png)

위 그림에서 빨간색과 파란색이 각 데이터들의 class라고 하자. 이때

- decision boundary : 실선
- hyperplane : 점선
- margin : decision boundary와 한 hyperplane간의 거리
- support vector : 각 hyperplane에 존재하는 data

라고 할 수 있다.

svm의 목적은 data를 class에 따라 잘 분류하는 것이기 때문에 결정 경계와 hyperplane간의 거리인 margin을 최대화하는 것이 목적이 된다. 

### SVM의 종류

![image](https://user-images.githubusercontent.com/79893946/199164112-3a262d7d-26c0-4bbf-b194-d91ea04e06a8.png)

다음 표에서 살펴볼 수 있듯이 3가지 경우를 살펴보게 될텐데 현실에서 모든 데이터가 정확하게 분류되는 것은 흔치 않기 때문에 hard margin을 사용하는 svm은 잘 사용하지 않고 case2와 case3를 많이 사용하게 된다. 

### Case 1 : Linear and Hard margin

![image](https://user-images.githubusercontent.com/79893946/199164155-f743a01a-7adb-4436-92e9-71ca52d15bc2.png)

case1은 위 그림처럼 모든 데이터를 분류할 수 있는 hyperplane을 찾고 그 margin을 최대화하는 것을 목적으로 하는 svm이다.

![image](https://user-images.githubusercontent.com/79893946/199164188-93bcfb5c-5e50-4ecb-a1d7-49e03ba00d70.png)

 

![image](https://user-images.githubusercontent.com/79893946/199164217-51b9ce00-f71d-40a9-b797-6bab482c573d.png)

margin이 w와 반비례하기 때문에  w를 최소화하는 것이 margin을 최대화하는 것이다. 따라서 목적식은 w를 최소화하는 것이고 제약조건은 위 그림과 같다.

위 목적식과 제약식을 사용하여 lagrangian problem으로 변형하고 kkt condition을 적용하여 dual problem으로 변환하면 다음의 분류함수를 얻을 수 있다.

![image](https://user-images.githubusercontent.com/79893946/199164238-d67b69b2-179e-4fdc-aaf2-a6943f960c3d.png)

(자세한 증명은 고려대학교 산업공학과 강필성 교수님의 Business Analysis 강의를 참고)

### Case 2 : Linear and Soft Margin

![image](https://user-images.githubusercontent.com/79893946/199164260-99b5ebb2-4cff-4c11-b92b-4bdcaeca3b51.png)

case 2는 case 1과 다르게 각각의 class 분류에 오차를 적용한 것을 위 그림을 통해 알 수 있다. 이때 $\xi$는 각 data가 margin을 벗어난 정도를 의미한다. 

![image](https://user-images.githubusercontent.com/79893946/199164281-cd077a0f-9f42-4380-a7a4-d99f79b1e45b.png)

따라서 목적함수는 case1의 목적함수에 penelty인  $\xi$와 이를 조절하는 hyperparameter인 C로 이루어진다. 또한 제약식에도 페널티가 부과된다.

목적함수 최적화를 통해 분류함수를 얻는 방법은 case1과 매우 비슷하다. 

![image](https://user-images.githubusercontent.com/79893946/199164311-de163943-2a64-45fe-a161-c8f4950eae10.png)

case2의 svm에서 penelty를 조절하는 c값에 따른 차이를 보면 c가 클수록 마진이 좁은 것을 알 수 있다.

### Case 3 : Nonlinear and Soft Margin

case1과 case2는 모두 분류 경계면이 선형이기 때문에 분류 경계면이 비선형인 것이 적합할 경우 잘 분류할 수 없게 된다. 

![image](https://user-images.githubusercontent.com/79893946/199164352-d0e4eb73-dd6f-4e34-9751-b7f5b2fca743.png)

따라서 case3은 위 그림과 같이 저차원의 데이터를 고차원으로 매핑하여 비선형 분류 경계면을 생성함으로써 유연성을 확보하고, margin을 최대화하여 일반화 성능을 확보하는 것을 목적으로 한다.

![image](https://user-images.githubusercontent.com/79893946/199164372-7c744da7-3b31-4489-8186-6a3f3c40cef8.png)

목적식과 제약식은 case2와 매우 유사하지만 데이터들을 바로 사용하는 것이 아니라 kernel 함수를 통해 고차원으로 mapping하여 사용한다. 따라서 크게보면 case2가 case3에 포함된다고 할 수 있다.

목적함수를 최적화하는 방법은 case2와 비슷하지만 kernel trick을 사용하여 kernel function을 찾아낸다는 점에서 차이가 있다. 

![image](https://user-images.githubusercontent.com/79893946/199164387-fd304b44-285d-4cf5-9afe-2b27d7d17171.png)

다음의 함수들이 대표적으로 사용되는 kernel 함수들이다. 

## 실험

1. C-SVM vs $\nu$-SVM
    1. Linear Case
    2. Nonlinear Case
2. SVM의 kernel function과 그 hyperparameter에 따른 결과 비교
    1. In C-SVM, result by hyperparameter C
    2. In RBF Kernel Function SVM, result by hyperparameter C, $\gamma$
    3. In Polynomial Kernel Function, result by hyperparameter degree, r 

### SVM 구현 코드

svm의 kernel function의 종류와 하이퍼파라미터 등을 지정하는 코드이다.

```python
def __init__(
            self,
            kernel: str = "linear",
            gamma: float | None = None,
            deg: int = 3,
            r: float = 0.,
            c: float = 1.
    ):
        # Lagrangian's multipliers, hyperparameters and support vectors 초기화
        self._lambdas = None
        self._sv_x = None
        self._sv_y = None
        self._w = None
        self._b = None

        # gamma 인자가 없으면 자동으로 계산됨
        self._gamma = gamma

        # function 종류에 따른 식 할당
        self._kernel = kernel
        if kernel == "linear":
            self._kernel_fn = lambda x_i, x_j: np.dot(x_i, x_j)
        elif kernel == "rbf":
            self._kernel_fn = lambda x_i, x_j: np.exp(-self._gamma * np.dot(x_i - x_j, x_i - x_j))
        elif kernel == "poly":
            self._kernel_fn = lambda x_i, x_j: (self._gamma * np.dot(x_i, x_j) + r) ** deg
        elif kernel == "sigmoid":
            self._kernel_fn = lambda x_i, x_j: np.tanh(np.dot(x_i, x_j) + r)

        # Soft margin svm의 penelty term parameter
        self._c = c

        self._is_fit = False
```

이 부분에서는 각 변수와 kernel function 및 hyperparameter를 지정해준다. 

- kernel : kernel function의 종류를 지정하는 인자로 kernel function의 기본값은 linear로 지정되어 있고 rbf, polynomial, sigmoid kernel function을 지정할 수 있다.
- gamma : rbf kernel과 polynomial kernel의 하이퍼파라미터인 gamma를 지정하는 인자이다.
- deg : polynomial kernel의 차수를 지정하는 인자이다.
- r : polynomial kernel과 sigmoid kernel의 하이퍼파라미터인 r을 지정하는 인자이다.
- c : penalty term c를 지정하는 인자이다.

cvxopt library를 사용하여 Lagrange problem의 optimization을 통해 data들을 fit하는 코드이다.

```python
def fit(self, x: np.ndarray, y: np.ndarray, verbosity: int = 1) -> None:
        # "verbosity"가 [0, 3]의 범위를 벗어나면 1 할당
        if verbosity not in {0, 1, 2}:
            verbosity = 1

        n_samples, n_features = x.shape
        # gamma가 지정되지 않으면 gamma = 1/sigma^2로 할당
        if not self._gamma:
            self._gamma = 1 / (n_features * x.var())

        # max{L_D(Lambda)} 다음의 식으로 다시 적을 수 있음
        #   min{1/2 Lambda^T H Lambda - 1^T Lambda}
        #       s.t. -lambda_i <= 0
        #       s.t. lambda_i <= c
        #       s.t. y^t Lambda = 0
        # where H[i, j] = y_i y_j K(x_i, x_j)
        k = np.zeros(shape=(n_samples, n_samples))
        for i, j in itertools.product(range(n_samples), range(n_samples)):
            k[i, j] = self._kernel_fn(x[i], x[j])
        p = cvxopt.matrix(np.outer(y, y) * k)
        q = cvxopt.matrix(-np.ones(n_samples))
        # 사용되는 margin의 종류에 따라 g, h matrix 계산 
        if self._c:
            g = cvxopt.matrix(np.vstack((
                -np.eye(n_samples),
                np.eye(n_samples)
            )))
            h = cvxopt.matrix(np.hstack((
                np.zeros(n_samples),
                np.ones(n_samples) * self._c
            )))
        else:
            g = cvxopt.matrix(-np.eye(n_samples))
            h = cvxopt.matrix(np.zeros(n_samples))
        a = cvxopt.matrix(y.reshape(1, -1).astype(np.double))
        b = cvxopt.matrix(np.zeros(1))

        # CVXOPT option 설정
        cvxopt.solvers.options["show_progress"] = False
        cvxopt.solvers.options["maxiters"] = 200

        # quadratic solver 사용해서 solution 계산
        try:
            sol = cvxopt.solvers.qp(p, q, g, h, a, b)
        except ValueError as e:
            print(f"Impossible to fit, try to change kernel parameters; CVXOPT raised Value Error: {e:s}")
            return

        # Lagrange multipliers 추출
        # lambdas = Lagrange multipliers
        lambdas = np.ravel(sol["x"])

        # non-zero Lagrange multipliers를 갖는 support vector들의 indice를 찾고,
        # support vector들 저장
        is_sv = lambdas > 1e-5
        self._sv_x = x[is_sv]
        self._sv_y = y[is_sv]
        self._lambdas = lambdas[is_sv]

        # b 계산: (1/N_s sum_i{y_i - sum_sv{lambdas_sv * y_sv * K(x_sv, x_i}})식 이용
        sv_index = np.arange(len(lambdas))[is_sv]
        self._b = 0
        for i in range(len(self._lambdas)):
            self._b += self._sv_y[i]
            self._b -= np.sum(self._lambdas * self._sv_y * k[sv_index[i], is_sv])
        self._b /= len(self._lambdas)

        # kernel이 linear라면 w 계산
        if self._kernel == "linear":
            self._w = np.zeros(n_features)
            for i in range(len(self._lambdas)):
                self._w += self._lambdas[i] * self._sv_x[i] * self._sv_y[i]
        else:
            self._w = None
        self._is_fit = True

        # verbosity에 따라 결과 출력
        if verbosity in {1, 2}:
            print(f"{len(self._lambdas):d} support vectors found out of {n_samples:d} data points")
            if verbosity == 2:
                for i in range(len(self._lambdas)):
                    print(f"{i + 1:d}) X: {self._sv_x[i]}\ty: {self._sv_y[i]}\tlambda: {self._lambdas[i]:.2f}")
            print(f"Bias of the hyper-plane: {self._b:.3f}")
            print("Weights of the hyper-plane:", self._w)
```

data의 fitting 후에 hyperplain에 data들을 projection하는 코드이다.

```python
def project(
            self,
            x: np.ndarray,
            i: int | None = None,
            j: int | None = None
    ) -> np.ndarray:

        if not self.is_fit:
            raise SVMNotFitError
        # kernel이 linear이고 w가 정의된다면, f(x)는 f(x) = X * w + b를 통해 결정됨
        if self._w is not None:
            return np.dot(x, self._w) + self._b
        else:
            # kernel이 linear가 아닌 경우에는
            # f(x) = sum_i{sum_sv{lambda_sv y_sv K(x_i, x_sv)}}를 통해 f(x) 결정됨 
            y_predict = np.zeros(len(x))
            for k in range(len(x)):
                for lda, sv_x, sv_y in zip(self._lambdas, self._sv_x, self._sv_y):
                    if i or j:
                        sv_x = np.array([sv_x[i], sv_x[j]])

                    y_predict[k] += lda * sv_y * self._kernel_fn(x[k], sv_x)
            return y_predict + self._b
```

projection을 통해 data들의 class를 predict할 수 있게 된다. 

data를 projection한 후의 point들의 부호를 통해 class를 예측하는 코드이다

```python
def predict(self, x: np.ndarray) -> np.ndarray:
        # point들의 label을 예측하기 위해서는 f(x)의 부호만 고려하면 됨
        return np.sign(self.project(x))
```

### 1. C-SVM vs NU-SVM

C-SVM과 $\nu$-SVM은 같은 parameter를 다르게 representation하고 수학적 표현이 다른 유사한 방식의 svm이다. 각각의 표현식을 살펴보았을 때는 어느 부분이 비슷하고, 어느 부분이 차이가 있는지 알기 힘든데, 이를 데이터에 각각 적용해보면서 알아보자.

1. **Linear Case**

dataset은 scikitlearn의 make_blobs 함수를 통해 선형으로 분리가능한 1000개의 데이터 샘플을 만들어서 실험하였다. 각 class에 따른 데이터의 분포는 다음과 같다.

![image](https://user-images.githubusercontent.com/79893946/199164495-1ec97428-b42f-43a5-a38b-64d07d90dfa3.png)

1) C-SVM

![image](https://user-images.githubusercontent.com/79893946/199164513-a4dd3ef0-ec94-4452-87cd-69d6036fdb4a.png)

2) $\nu$-SVM

![image](https://user-images.githubusercontent.com/79893946/199164531-a4dd6292-35b5-43c0-b578-68fe3f11e32e.png)

결과표

|  | C-SVM |  nu-SVM |
| --- | --- | --- |
| Accuracy | 0.893 | 0.897 |
| Precision | 0.903 | 0.908 |
| Recall | 0.891 | 0.891 |

먼저, fitting된 margin을 살펴보면 $\nu$-SVM의 margin이 C-SVM에 비해 더 넓은 것을 확인할 수 있다.

하지만, 결과표의 성능지표를 확인해보면 Accuracy, Precision, Recall 모두 거의 차이가 없는 것을 확인할 수 있다. 따라서 linear case에서는 margin의 차이는 있지만 각 기법의 성능 차이는 없다고 결론지을 수 있을 것 같다.

**b. Nonlinear Case**

dataset은 scikitlearn의 make_circles 함수를 통해 원형으로 분포하는 1000개의 데이터 샘플을 만들어서 실험하였다. 각 class에 따른 데이터의 분포는 다음과 같다.

![image](https://user-images.githubusercontent.com/79893946/199164574-235e8fed-e3c5-4bc6-b43e-63c250a6846b.png)

1) C-SVM

![image](https://user-images.githubusercontent.com/79893946/199164593-ee4ddebd-bb77-4cfe-ad3e-756f4aa11ee9.png)

2) $\nu$-SVM

![image](https://user-images.githubusercontent.com/79893946/199164608-fd8a882a-c38a-475f-a989-f14ec1951ec6.png)

결과표

|  | C-SVM |  nu-SVM |
| --- | --- | --- |
| Accuracy | 1.000 | 0.997 |
| Precision | 1.000 | 0.993 |
| Recall | 1.000 | 1.000 |

Nonlinear한 data 분포의 경우에도 linear한 경우와 거의 비슷한 결과를 보인다.

먼저, fitting된 margin을 살펴보면 $\nu$-SVM의 margin이 C-SVM에 비해 더 넓은 것을 확인할 수 있다.

하지만, 결과표의 성능지표를 확인해보면 Accuracy, Precision, Recall 모두 거의 차이가 없는 것을 확인할 수 있다. 따라서 nonlinear case에서도 margin의 차이는 있지만 각 기법의 성능 차이는 없다고 결론지을 수 있을 것 같다.

결론적으로, C-SVM과 $\nu$-SVM은 margin의 넓이와 parameter의 표현이외에는 굉장히 비슷한 방법이라고 할 수 있다.

### 2. SVM의 kernel function과 그 hyperparameter에 따른 결과 비교

**Data**

iris data를 활용하여 실험을 진행하였다. iris data 중에서도 class가 두 가지만 있는 것이 이해가 쉽기 때문에 setosa와 versicolor label만 사용하였다.

![image](https://user-images.githubusercontent.com/79893946/199164641-7d2f83fc-5713-4ff0-bf60-7a67317d75b7.png)

1. **In C-SVM, result by hyperparameter C**

![image](https://user-images.githubusercontent.com/79893946/199164651-7628932f-54c7-47d7-846a-e7f03451c810.png)

결과표

| C | Accuracy |
| --- | --- |
| 0.005 | 0.70 |
| 0.01 | 0.97 |
| 0.05 | 0.97 |
| 0.1 | 0.97 |
| 0.5 | 0.97 |
| 1 | 0.97 |
| 3 | 0.97 |
| 5 | 1.00 |
| 10 | 0.97 |

먼저 그림을 살펴보면 C-SVM에서 C가 커질수록 penalty를 많이 주기 때문에 margin이 작아지는 것을 확인할 수 있다.  

C와 Accuracy를 살펴보면 너무 작거나 큰 C가 아닌 적당한 C를 선택해야 성능이 좋다는 것을 알 수 있다. 

**b. In RBF Kernel Function SVM, result by hyperparameter C, $\gamma$**

![image](https://user-images.githubusercontent.com/79893946/199164680-c2dcfbdb-d8d2-42a3-812c-f14918322fcb.png)

결과표

| C | gamma | Accuracy |
| --- | --- | --- |
| 0.1 | 0.1 | 0.97 |
| 0.1 | 1 | 0.97 |
| 0.1 | 10 | 0.47 |
| 1 | 0.1 | 0.97 |
| 1 | 1 | 0.97 |
| 1 | 10 | 0.97 |
| 10 | 0.1 | 0.97 |
| 10 | 1 | 0.97 |
| 10 | 10 | 0.97 |

먼저 그림을 살펴보면 RBF Kernel Function SVM에서 C값에 따라 차이는 있지만 C가 같은 경우에 $\gamma$가 작을수록 선형에 가까운 결정경계를 만들고 $\gamma$가 클수록 복잡한 결정경계를 만드는 것을 확인할 수 있다.

C, $\gamma$와 Accuracy를 살펴보면 C와 $\gamma$의 적절한 조합을 선택해야 모델의 성능이 좋은 것을 확인할 수 있다.

**c. In Polynomial Kernel Function, result by hyperparameter degree, r**

![image](https://user-images.githubusercontent.com/79893946/199164706-774be6f2-8911-4c95-a8f0-50c953360351.png)

결과표

| C | gamma | d | r | Accuracy |
| --- | --- | --- | --- | --- |
| 1 | 0.1 | 2 | -50 | 0 |
| 1 | 0.1 | 2 | -1 | 0 |
| 1 | 0.1 | 2 | 0 | 0.47 |
| 1 | 0.1 | 2 | 1 | 0.97 |
| 1 | 0.1 | 2 | 50 | 1.00 |
| 1 | 1 | 2 | -50 | 0 |
| 1 | 1 | 2 | -1 | 0 |
| 1 | 1 | 2 | 0 | 0.53 |
| 1 | 1 | 2 | 1 | 0.97 |
| 1 | 1 | 2 | 50 | 1.00 |
| 1 | 0.1 | 3 | -50 | 0.97 |
| 1 | 0.1 | 3 | -1 | 0.97 |
| 1 | 0.1 | 3 | 0 | 0.73 |
| 1 | 0.1 | 3 | 1 | 0.97 |
| 1 | 0.1 | 3 | 50 | 1.00 |
| 1 | 1 | 3 | -50 | 0.97 |
| 1 | 1 | 3 | -1 | 0.93 |
| 1 | 1 | 3 | 0 | 1.00 |
| 1 | 1 | 3 | 1 | 1.00 |
| 1 | 1 | 3 | 50 | 1.00 |

먼저 그림을 살펴보면 Polynomial Kernel Function SVM에서 다른 하이퍼파라미터가 같을 때, r이 작아질수록 더 복잡한 결정경계를 만드는 것을 확인할 수 있다. 또한 다른 하이퍼파라미터가 같을 때, 차수가 높을수록 더 복잡한 결정경계를 만드는 것을 확인할 수 있다. 

하이퍼파라미터들과 Accuracy를 살펴보면 이렇게 하이퍼파라미터가 많을 때에는 하나의 하이퍼파라미터와 뚜렷한 상관관계를 보이기보다는 하이퍼파라미터 조합에 따라 성능이 결정되기 때문에 하이퍼파라미터 검색 알고리즘 등을 통해 가장 적절한 하이퍼파라미터를 찾는 것이 중요하다는 결론을 내릴 수 있다. 

reference

[https://zernes.github.io/SVM/](https://zernes.github.io/SVM/)

[https://www.kaggle.com/code/carlosdg/effect-of-hyperparameters-and-kernels-on-svms/notebook](https://www.kaggle.com/code/carlosdg/effect-of-hyperparameters-and-kernels-on-svms/notebook)
