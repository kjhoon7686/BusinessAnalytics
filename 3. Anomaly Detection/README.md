# Anomaly Detection

# 목차

## 개념

1. Local Outlier Factor (LOF)
2. Auto-Encoder (AE)
3. Support Vector-based Anomaly Detection (OCSVM)
4. Isolation Forests (IF)

## 코드 구현

1. Local Outlier Factor (LOF)
2. Auto-Encoder (AE)
3. Support Vector-based Anomaly Detection (OCSVM)
4. Isolation Forests (IF)

## 실험

1. Outlier 비율별 생성 데이터셋의 성능 비교 [1%, 5%, 10%, 30%] 
2. Outlier 비율별 실제 데이터셋의 성능 비교 [1.2%, 6.25%, 9.6%, 32%]
3. Outlier 비율별 실제 데이터셋에 대한 hyperparameter tuning 후 성능 비교
4. 데이터셋/모델별 최종 성능 비교
5. 결론

## 개념

### 1. Local Outlier Factor (LOF)

LOF는 주변부 데이터의 상대적인 밀도를 고려하는 이상치 탐지 방법이다.

LOF에서는 k-distance와 reachability distance라는 개념이 중요하다.

먼저, k-distance(p)는 데이터 A로부터 k번째로 가장 가까운 데이터까지의 거리를 의미한다. $N_k(p)$는 k-distance 범위 내에 들어오는 데이터의 집합을 의미한다.

다음으로, reachability distance는 k-distance 범위 안에 있는 데이터까지의 거리는 distance(p, o)와 k-distance 중에 큰 값을 사용하는 개념이다.

![image](https://user-images.githubusercontent.com/79893946/201839257-775b1c0e-c4ad-4fd3-aa76-d415012cedb9.png)

![image](https://user-images.githubusercontent.com/79893946/201839300-9d3186a4-1389-44f7-82a6-42ccd711c7e5.png)

이 개념들을 사용하면 LOF알고리즘은 다음과 같이 정의될 수 있다.

![image](https://user-images.githubusercontent.com/79893946/201839325-dd23f8d3-2cb6-47dd-95bd-5ede9f5706b1.png)

![image](https://user-images.githubusercontent.com/79893946/201839360-e18966bf-5954-41b7-9622-8119ec97a281.png)

위의 그림을 보면 주변부 밀도와 데이터 간 거리에 따라 이상치 score가 어떻게 변화하는지 알 수 있다. 

다음 그림은 LOF의 예시이다.

![image](https://user-images.githubusercontent.com/79893946/201839390-1250f0cf-2322-4978-94b1-e4306cab0197.png)

![image](https://user-images.githubusercontent.com/79893946/201839429-6e58063f-6065-45ed-bb54-b720b656d1cf.png)

### 2. Auto-Encoder (AE)

Auto-Encoder란 input과 output이 동일한 neural network이다.

![image](https://user-images.githubusercontent.com/79893946/201839473-b425af76-af18-4bc3-ac8c-fe62307b794e.png)

위의 그림을 보면 auto-encoder는 고양이 사진을 input으로 했을 때, encoder와 decoder를 지나 다시 원래의 고양이 사진을 복원하는 것을 목적으로 함을 알 수 있다. 이 때, latent space는 원본 데이터의 크기보다 작아야한다. 

![image](https://user-images.githubusercontent.com/79893946/201839499-de7bd928-3ad1-46cc-b627-efbf86d6293e.png)

위의 예시를 neural network의 형태로 나타낸 그림이다.

input을 동일하게 복원한 output을 산출하는 auto-encoder모델은 정상 데이터에 대해서 학습되기 때문에 정상 데이터는 잘 복원할 수 있지만, 이상치 데이터는 학습되지 않았기 때문에 복원력이 떨어진다는 점을 활용한 모델이다. 

auto-encoder는 크게 두 가지 방식으로 활용될 수 있는데 첫째는 latent space의 크기가 input data보다 작은 것을 활용하여 **차원축소**를 하는 것이고, 둘째는 앞서 언급했듯이 **이상치 탐치**에 사용하는 것이다.  

앞서 설명한 일반적인 auto-encoder이외에도 이미지 데이터를 처리하는 모델인 cnn을 활용한 Convolutional Auto-Encoder(CAE)라는 모델도 있다.

![image](https://user-images.githubusercontent.com/79893946/201839567-13cacae5-e37f-4c89-9147-bdbbc24cace1.png)

![image](https://user-images.githubusercontent.com/79893946/201839588-07dc4776-5815-47f7-92be-b7c523e1d7e5.png)

CAE의 핵심은 일반적인 cnn과는 다르게 데이터가 축소되었다가 다시 input size와 동일한 output size로 확대되어야 한다는 점이다. 이를 위해 unpooling과 transpose convolution이라는 방법을 사용한다.

unpooling은 max pooling의 위치를 기억하고 그 위치에 값을 복원하는 방법이다.

![image](https://user-images.githubusercontent.com/79893946/201839637-277b323b-3242-4781-9928-62cdb3ed0bea.png)

transpose convolution은 convolution 연산을 통해 feature map의 크기를 키우는 방법이다.

![image](https://user-images.githubusercontent.com/79893946/201839684-bd3c4746-aa75-4929-b676-380ed5537eea.png)

### 3. Support Vector-based Anomaly Detection (OCSVM)

SVM 방법론 중에서 가장 많이 사용되는 방법론 중 하나인 One-class Support Vector Machine은 다음 그림과 같이 정상 데이터를 최대한 떨어뜨리는 hyperplane을 찾는 SVM 방법론이다. 다음 그림에서 확인할 수 있듯이 hyperplane 아래이 있는 데이터는 outlier로, hyperplane 위에 있는 데이터는 정상 데이터로 분류된다.

![image](https://user-images.githubusercontent.com/79893946/201839731-f9e6b09a-2dae-4e5a-afbd-9113c9524bdd.png)

One-class Support Vector Machine의 증명 과정을 간단하게 살펴보자.

![image](https://user-images.githubusercontent.com/79893946/201839875-594e7d0c-d114-4d89-bbda-15e7ad60e167.png)

다음은 one-class svm의 목적식인데, 두 부분으로 이루어져 있다. 먼저 앞의 수식은 margin을 최대화하는 목적을 달성하기 위한 식이고, 두번째 식은 원점에서 최대한 멀어져야 하는 목적을 달성하기 위한 식이다. 또한 가운데 식은 클래스로 인정받지 못하는 샘플을 최소화하는 목적을 달성하기 위한 식이다. 마지막으로 라그랑지 제약조건을 보면 모든 데이터는 결정평면 위쪽에 위치해야 한다는 제약식을 확인할 수 있다. 

위의 제약식에서 라그랑지와 듀얼 문제를 거치면 최종적으로 다음과 같은 최적화 문제로 수렴된다. 

![image](https://user-images.githubusercontent.com/79893946/201839928-b4a18693-54bb-4522-a356-32cef0d820e3.png)

또한 일반적인 svm과 같이 kernel trick을 사용할 수 있다.

### 4. Isolation Forests (IF)

isolation forest는 이상치 데이터는 개체수가 적고 정상 데이터와 특정 속성의 값이 다를 가능성이 높다는 점에서 시작된 모델이다. 

![image](https://user-images.githubusercontent.com/79893946/201839960-7b34cace-65a5-439e-a31c-2d78fc7c803a.png)

다음과 같이 하나의 객체를 고립시키는 tree를 생성하는 상황을 가정해보자. 정상 데이터라면 고립시키는데에 split이 많이 필요하고, 이상치 데이터라면 상대적으로 적은 split으로 고립시킬 수 있는 것을 알 수 있다.

isolation forest의 알고리즘은 다음과 같다.

![image](https://user-images.githubusercontent.com/79893946/201839984-91fbd030-4e68-4f55-b387-5bb52ad286d1.png)

먼저, 전체 데이터에서 n개의 데이터 집합을 샘플링한다. 이 떼 tree하나당 256개 정도 샘플링하면 충분하다고 알려져 있다. 다음으로 이렇게 랜덤하게 선택된 관측치에 대해 임의의 변수와 분할점을 사용하여 이진 분할을 한다. 이 분할은 위 그림의 3가지 조건을 만족시켜야 한다. 그리고 1,2의 과정을 반복하여 여러 개의 tree를 생성하고 tree마다 각 관측치의 path length를 저장한다. 마지막으로 각 관측치의 평균 path length를 기반으로 이상치 스코어 계산 및 이상치를 판별한다. 

이때, 이상치 스코어는 다음과 같이 정의될 수 있다. 

![image](https://user-images.githubusercontent.com/79893946/201840027-cf7be067-45ae-4ec6-886b-ae5cd01c8ecf.png)

![image](https://user-images.githubusercontent.com/79893946/201840054-5716f538-3d99-44e7-a6b1-82422941df41.png)

즉, tree에서 path length가 짧을수록 이상치 스코어는 1에 가까워지고, path length가 길수록 이상치 스코어는 0에 가까워지는 것을 확인할 수 있다.

## 코드 구현

### 1. Local Outlier Factor (LOF)

```python
def k_distance(k, instance, instances, distance_function=distance_euclidean):
    distances = {}
    for instance2 in instances:
        distance_value = distance_function(instance, instance2)
        if distance_value in distances:
            distances[distance_value].append(instance2)
        else:
            distances[distance_value] = [instance2]
    distances = sorted(distances.items())
    neighbours = []
    [neighbours.extend(n[1]) for n in distances[:k]]
    k_distance_value = distances[k - 1][0] if len(distances) >= k else distances[-1][0]
    return k_distance_value, neighbours
```

k-distance를 구하는 함수이다. 하나의 instance에 대해 다른 instance들과의 거리를 계산하고 이를 크기별로 정렬한 뒤 k개의 value를 저장한다.

```python
def reachability_distance(k, instance1, instance2, instances, distance_function=distance_euclidean):
    (k_distance_value, neighbours) = k_distance(k, instance2, instances, distance_function=distance_function)
    return max([k_distance_value, distance_function(instance1, instance2)])
```

reachability_distance를 구하는 함수이다. k-distance값과 두 instance의 distance 중 최댓값을 사용하는 것을 확인할 수 있다. 

```python
def local_reachability_density(min_pts, instance, instances, **kwargs):
    (k_distance_value, neighbours) = k_distance(min_pts, instance, instances, **kwargs)
    reachability_distances_array = [0]*len(neighbours) #n.zeros(len(neighbours))
    for i, neighbour in enumerate(neighbours):
        reachability_distances_array[i] = reachability_distance(min_pts, instance, neighbour, instances, **kwargs)
    if not any(reachability_distances_array):
        warnings.warn("Instance %s (could be normalized) is identical to all the neighbors. Setting local reachability density to inf." % repr(instance))
        return float("inf")
    else:
        return len(neighbours) / sum(reachability_distances_array)
```

local_reachability_density를 구하는 함수이다. 

```python
def local_outlier_factor(min_pts, instance, instances, **kwargs):
    (k_distance_value, neighbours) = k_distance(min_pts, instance, instances, **kwargs)
    instance_lrd = local_reachability_density(min_pts, instance, instances, **kwargs)
    lrd_ratios_array = [0]* len(neighbours)
    for i, neighbour in enumerate(neighbours):
        instances_without_instance = set(instances)
        instances_without_instance.discard(neighbour)
        neighbour_lrd = local_reachability_density(min_pts, neighbour, instances_without_instance, **kwargs)
        lrd_ratios_array[i] = neighbour_lrd / instance_lrd
    return sum(lrd_ratios_array) / len(neighbours)
```

local_reachability_density를 사용해서 최종적으로 LOF를 구하는 함수이다. 

```python
def outliers(k, instances, **kwargs):
    instances_value_backup = instances
    outliers = []
    for i, instance in enumerate(instances_value_backup):
        instances = list(instances_value_backup)
        instances.remove(instance)
        l = LOF(instances, **kwargs)
        value = l.local_outlier_factor(k, instance)
        if value > 1:
            outliers.append({"lof": value, "instance": instance, "index": i})
    outliers.sort(key=lambda o: o["lof"], reverse=True)
    return
```

위에서 설명한 함수들을 사용하여 최종적으로 outlier를 판별하는 함수이다. 

### 2. Auto-Encoder (AE)

```python
class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True), nn.Linear(64, 12), nn.ReLU(True), nn.Linear(12, 3))
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True), nn.Linear(128, 28 * 28), nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return
```

auto-encoder의 구현은 간단하다. 일반적인 neural network들의 module을 사용한다. 이 때 주의해야할 점은 input과 output의 크기가 같아야 하고, latent space를 의미하는 encoder의 마지막 크기와 decoder의 처음 크기가 같고 input size보다 작아야 한다. 

```python
class conv_autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return
```

convolutional auto-encoder의 구현이다. 기본적인 auto-encoder와 구조가 매우 유사한데, neural network의 모듈만 convolution과 pooling으로 바뀐 것을 확인할 수 있다.

### 3. Support Vector-based Anomaly Detection (OCSVM)

```python
class SVM(object):
    def __init__(self):

    def fit(self,data):
        #train with data
        self.data = data
        # { |\w\|:{w,b}}
        opt_dict = {}
        
        transforms = [[1,1],[-1,1],[-1,-1],[1,-1]]
        
        all_data = np.array([])
        for yi in self.data:
            all_data = np.append(all_data,self.data[yi])
                    
        self.max_feature_value = max(all_data)         
        self.min_feature_value = min(all_data)
        all_data = None
        
        #with smaller steps our margins and db will be more precise
        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      #point of expense
                      self.max_feature_value * 0.001,]
        
        #extremly expensise
        b_range_multiple = 5
        #we dont need to take as small step as w
        b_multiple = 5
        
        latest_optimum = self.max_feature_value*10
        
        #making step smaller and smaller to get precise value
        for step in step_sizes:
            w = np.array([latest_optimum,latest_optimum])
            
            #we can do this because convex
            optimized = False
            while not optimized:
                for b in np.arange(-1*self.max_feature_value*b_range_multiple,
                                   self.max_feature_value*b_range_multiple,
                                   step*b_multiple):
                    for transformation in transforms:
                        w_t = w*transformation
                        found_option = True
                        
                        #weakest link in SVM fundamentally
                        #SMO attempts to fix this a bit
                        # ti(xi.w+b) >=1
                        for i in self.data:
                            for xi in self.data[i]:
                                yi=i
                                if not yi*(np.dot(w_t,xi)+b)>=1:
                                    found_option=False
                if w[0]<0:
                    optimized=True
                    print("optimized a step")
                else:
                    w = w-step
                    
            # sorting ||w|| to put the smallest ||w|| at poition 0 
            norms = sorted([n for n in opt_dict])
            #optimal values of w,b
            opt_choice = opt_dict[norms[0]]

            self.w=opt_choice[0]
            self.b=opt_choice[1]
            
            latest_optimum = opt_choice[0][0]+step*2
```

다음은 one-class svm 구현코드이다. 

### 4. Isolation Forests (IF)

```python
class IsolationTree:
    def __init__(self, height, height_limit):
        self.height = height
        self.height_limit = height_limit

    def fit(self, X: np.ndarray, improved=False):
        if improved:
            self.improved_fit(X)
        else:
            if self.height >= self.height_limit or X.shape[0] <= 2:
                self.root = LeafNode(X.shape[0], X)
                return self.root

            # Choose Random Split Attributes and Value
            num_features = X.shape[1]
            splitAtt = np.random.randint(0, num_features)
            splitVal = np.random.uniform(min(X[:, splitAtt]), max(X[:, splitAtt]))

            X_left = X[X[:, splitAtt] < splitVal]
            X_right = X[X[:, splitAtt] >= splitVal]

            left = IsolationTree(self.height + 1, self.height_limit)
            right = IsolationTree(self.height + 1, self.height_limit)
            left.fit(X_left)
            right.fit(X_right)
            self.root = DecisionNode(left.root, right.root, splitAtt, splitVal)
            self.n_nodes = self.count_nodes(self.root)
            return self.root

    def improved_fit(self, X: np.ndarray):
        if self.height >= self.height_limit or X.shape[0] <= 2:
            self.root = LeafNode(X.shape[0], X)
            return self.root

        # Choose Best (The Most unbalanced) Random Split Attributes and Value
        num_features = X.shape[1]
        ratio_imp = 0.5 # Intialize the samples ratio after split as 0.5

        for i in range(num_features):
            splitAtt = i
            for _ in range(10):
                splitVal = np.random.uniform(min(X[:, splitAtt]), max(X[:, splitAtt]))
                X_left = X[X[:, splitAtt] < splitVal]
                X_right = X[X[:, splitAtt] >= splitVal]
                ratio = min(X_left.shape[0] / (X_left.shape[0] + X_right.shape[0]),
                            X_right.shape[0] / (X_left.shape[0] + X_right.shape[0]))
                if ratio < ratio_imp:
                    splitAtt_imp = splitAtt
                    splitVal_imp = splitVal
                    X_left_imp = X_left
                    X_right_imp = X_right
                    ratio_imp = ratio

        left = IsolationTree(self.height + 1, self.height_limit)
        right = IsolationTree(self.height + 1, self.height_limit)
        left.fit(X_left_imp)
        right.fit(X_right_imp)
        self.root = DecisionNode(left.root, right.root, splitAtt_imp, splitVal_imp)
        self.n_nodes = self.count_nodes(self.root)
        return self.root

    def count_nodes(self, root):
        count = 0
        stack = [root]
        while stack:
            node = stack.pop()
            count += 1
            if isinstance(node, DecisionNode):
                stack.append(node.right)
                stack.append(node.left)
        return count

class IsolationTreeEnsemble:
    def __init__(self, sample_size, n_trees=10):
        self.sample_size = sample_size
        self.n_trees = n_trees

    def fit(self, X: np.ndarray, improved=False):
        self.trees = []
        if isinstance(X, pd.DataFrame):
            X = X.values
        n_rows = X.shape[0]
        height_limit = np.ceil(np.log2(self.sample_size))
        for i in range(self.n_trees):
            # data_index = np.random.choice(range(n_rows), size=self.sample_size, replace=False)
            data_index = np.random.randint(0, n_rows, self.sample_size)
            X_sub = X[data_index]
            tree = IsolationTree(0, height_limit)
            tree.fit(X_sub)
            self.trees.append(tree)
        return self

    def path_length(self, X:np.ndarray) -> np.ndarray:
        paths = []
        for row in X:
            path = []
            for tree in self.trees:
                node = tree.root
                length = 0
                while isinstance(node, DecisionNode):
                    if row[node.splitAtt] < node.splitVal:
                        node = node.left
                    else:
                        node = node.right
                    length += 1
                leaf_size = node.size
                pathLength = length + c(leaf_size)
                path.append(pathLength)
            paths.append(path)
        paths = np.array(paths)
        return np.mean(paths, axis=1)

    def anomaly_score(self, X:pd.DataFrame) -> np.ndarray:

        if isinstance(X, pd.DataFrame):
            X = X.values
        avg_length = self.path_length(X)
        scores = np.array([np.power(2, -l/c(self.sample_size))for l in avg_length])
        return scores

    def predict_from_anomaly_scores(self, scores:np.ndarray, threshold:float) -> np.ndarray:
        return np.array([1 if s >= threshold else 0 for s in scores])

    def predict(self, X:np.ndarray, threshold:float) -> np.ndarray:
        scores = self.anomaly_score(X)
        prediction = self.predict_from_anomaly_scores(scores, threshold)
        return prediction
```

다음은 isolation forest 구현 코드이다. 먼저 tree를 통해 node를 생성하고, 이를 이용해 path length의 길이를 구한 후 anomaly score를 산출하는 것을 확인할 수 있다.

## 실험

### 1. Outlier 비율별 생성 데이터셋의 성능 비교[ 1%, 5%, 10%, 30%]

데이터셋의 outlier 비율에 따른 모델별 성능을 확인해보기 실험을 진행하였다. 모델은 LOF, IF, AE, OCSVM 이렇게 4 종류를 사용하였다. 먼저 gaussian distribution에서 data를 sampling한 후 uniform distribution을 통해 outlier를 추가하여 인위적인 데이터를 만들어보았다. outlier 비율은 전체 데이터의 1%, 5%, 10%, 30% 네 가지로 하여 데이터를 구성하였다. 전체 데이터 수는 1000개로 고정하였다. 하이퍼파라미터는 default로 설정하였다.

**1) outlier 1%**

![image](https://user-images.githubusercontent.com/79893946/201840129-63e0518f-91b1-453c-8f29-afbc9d503487.png)

total outlier = 10

| model | number of error |
| --- | --- |
| LOF | 73 |
| IF | 90 |
| AE | 90 |
| OCSVM | 90 |

outlier 1% 데이터에서는 LOF가 가장 성능이 좋고 나머지 세 모델은 성능이 비슷한 것을 확인할 수 있다. 

**2) outlier 5%**

![image](https://user-images.githubusercontent.com/79893946/201840166-8f08c972-5bb2-4fbe-978b-f6307088b923.png)

total outlier = 50

| model | number of error |
| --- | --- |
| LOF | 47 |
| IF | 54 |
| AE | 56 |
| OCSVM | 56 |

outlier 5% 데이터에서는 전반적으로 모든 모델의 성능이 향상된 것을 확인할 수 있다. 또한 여전히 LOF의 성능이 가장 좋고 나머지 세 모델은 비슷한 성능을 보이는 것을 알 수 있다. Outlier가 너무 적은 데이터는 모델이 학습하기 어렵다는 것을 추론할 수 있었디.

**3) outlier 10%**

![image](https://user-images.githubusercontent.com/79893946/201840202-e5ff63c2-052a-432c-bf13-8975262abbb3.png)

total outlier = 100

| model | number of error |
| --- | --- |
| LOF | 86 |
| IF | 6 |
| AE | 6 |
| OCSVM | 6 |

outlier 10% 데이터에서는 IF, AE, OCSVM의 성능은 대폭 상승했지만 LOF의 성능은 오히려 하락한 것을 확인할 수 있다. 여전히 세 모델은 비슷한 성능을 보여주었다. 위의 결과를 해석해보면 LOF 모델은 outlier가 5-10%일 때, 가장 성능이 좋고 나머지 모델들은 10%까지는 outlier의 비율이 높아질수록 성능이 좋아지는 것으로 추론할 수 있을 것 같다. 

**4) outlier 30%**

![image](https://user-images.githubusercontent.com/79893946/201840241-9eee8595-f998-4878-8fcc-55df91a41edc.png)

total outlier = 300

| model | number of error |
| --- | --- |
| LOF | 298 |
| IF | 200 |
| AE | 200 |
| OCSVM | 200 |

outlier 30% 데이터에서는 모든 모델의 성능이 악화된 것을 확인할 수 있다. 또한 LOF의 성능은 여전히 나머지 세 모델들에 비해 좋지 않았고, 세 모델은 비슷한 성능을 보였다. 

이를 통해 종합적으로 결론을 내보면 LOF는 outlier가 적을 때 model-based 모델들보다 성능이 좋고 outlier가 일정 비율[이 실험에서는 5-10%]이 넘어가면 성능이 악화되는 것을 알 수 있었다. 또한 나머지 세 모델은 outlier의 비율이 너무 적을 때는 성능이 좋지 않고 outlier 비율이 높아질수록 성능이 좋아졌다. 하지만 outlier가 30% 정도 되는 이상치가 너무 많은 상황에서는 성능이 악화되는 것을 확인할 수 있었다. 마지막으로 모든 outlier 비율 데이터에서 model-based 모델의 성능이 비슷한 것이 인상적이었다.

### 2. Outlier 비율별 실제 데이터셋의 성능 비교

실험 1의 데이터셋은 인위적으로 만든 데이터셋이었다. 인위적으로 만든 데이터셋에 대한 실험 결과가 실제 데이터셋에서의 결과와 같은 방향성을 보이는지를 확인하기 위해 위의 outlier 비율과 최대한 비슷한 실제 데이터셋을 사용해서 성능 비교 실험을 해보았다. 성능지표는 AUROC를 사용하였다.

실험에 사용한 데이터셋은 다음과 같다.

- Outlier 1.2% : [Satimage-2](http://odds.cs.stonybrook.edu/satimage-2-dataset/) [number of data = 5803]
- Outlier 6.25% : [Letter Recognition](http://odds.cs.stonybrook.edu/letter-recognition-dataset/) [number of data = 1600]
- Outlier 9.6% : [Cardio](http://odds.cs.stonybrook.edu/cardiotocogrpahy-dataset/)[number of data = 1831]
- Outlier 32% : [Satellite](http://odds.cs.stonybrook.edu/satellite-dataset/)[number of data = 6435]

데이터의 수가 1000개 이상인 데이터를 사용하였다. 또한 모델들은 실험 1에서와 같이 default hyperparameter로 진행하였다. 

**1) 학습 시간 [Time]**

| Data | #Samples | # Dimensions | Outlier Perc | LOF | IF | AE | OCSVM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| satimage-2 | 5803 | 36 | 1.2235 | 0.4838 | 0.5587 | 5.3474 | 0.7752 |
| letter | 1600 | 32 | 6.25 | 0.0467 | 0.3339 | 2.7424 | 0.0539 |
| cardio | 1831 | 21 | 9.6122 | 0.0506 | 0.3393 | 2.9235 | 0.0835 |
| satellite | 6435 | 36 | 31.6395 | 0.487 | 0.5407 | 5.5657 | 0.9477 |

LOF, IF, OCSVM, AE 순으로 학습시간이 오래 소요되었고, LOF는 밀도 기반이기 때문에 가장 시간이 적게 걸리고, 다른 모델들은 이에 비해 시간이 오래 걸리는 것을 알 수 있었다. AE는 neural network 모델이기 때문에 시간이 가장 오래 걸리는 것을 알 수 있었다.

**2) AUROC** 

| Data | #Samples | # Dimensions | Outlier Perc | LOF | IF | AE | OCSVM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| satimage-2 | 5803 | 36 | 1.2235 | 0.5699 | 0.9969 | 0.9827 | 0.9997 |
| letter | 1600 | 32 | 6.25 | 0.8631 | 0.5551 | 0.4845 | 0.5918 |
| cardio | 1831 | 21 | 9.6122 | 0.5898 | 0.93 | 0.9476 | 0.9271 |
| satellite | 6435 | 36 | 31.6395 | 0.5605 | 0.701 | 0.6123 | 0.6629 |

satimage-2 데이터부터 살펴보면 실험 1의 결과와는 다르게 LOF를 제외한 모델의 성능이 아주 높게 나왔으며, LOF의 성능은 낮게 나왔다. 하지만 letter 데이터를 살펴보면 LOF의 성능이 나머지 세 모델의 성능보다 높게 나온 것이 실험1의 결과와 일치했다. 여기서 실험 1과의 가장 큰 차이점이라면 데이터의 수인데 satimage-2의 데이터 수는 5803개로 letter보다 훨씬 작기 크기 때문에 model-based 모델들의 성능이 높은 것으로 예상되고 task가 아주 쉬웠을 것으로 예상된다. 따라서 outlier 비율 못지 않게 **데이터의 수**가 모델 성능에 큰 영향을 주는 것을 알 수 있다.

cardio와 satellite 데이터의 결과를 보면 실험 1과 같은 결과를 보이는 것을 알 수 있다. outlier 비율이 10% 정도일 때 model-based 모델들의 성능이 매우 높고 LOF의 성능이 낮았다. 또한 outlier 비율이 30% 이상인 satellite의 결과를 보면 데이터의 수가 많음에도 불구하고 성능이 다 떨어진 것을 확인할 수 있었다.

결론적으로 outlier 비율이 약 5% 이하일 때는 데이터의 수가 모델 성능에 더 큰 영향을 주고, outlier 비율이 약 10% 일때는 model-based 모델들의 성능이 LOF보다 높은 것을 확인할 수 있었다. 또한 outlier 비율이 30% 이상인 outlier가 많은 데이터의 경우에는 데이터 수와 관계 없이 모델 성능이 모두 악화되었지만 상대적으로 model-based 모델들의 성능이 더 높은 것을 알 수 있었다.

### 3. Outlier 비율별 실제 데이터셋에 대한 hyperparameter tuning 후 성능 비교

실험 1, 2의 결과가 hyperparameter setting 때문이었을 수 있기 때문에 hyperparameter에 따른 성능 변화와 tuning 후 데이터셋별 모델 성능을 다시 비교해보기 위해 데이터셋/모델별 hyperparameter에 따른 성능을 비교해보았다. 평가지표는 마찬가지로 AUROC를 사용했다. 각 모델별로 중요하다고 생각하는 hyperparameter를 두 개씩 정해서 실험을 진행하였다.

1) LOF

**Default setting** 

- n_neighbors = 20
- leaf_size =30

**Hyperparameter setting**

- n_neighbors = [10,20,30]
- leaf_size = [20,30,40]

n_neighbors는 k-distance의 k를 의미하고 leaf_size는 tree의 leaf_size이다.

**결과**

| Data | satimage-2 | letter | cardio | satellite |
| --- | --- | --- | --- | --- |
| # Samples | 5803 | 1600 | 1831 | 6435 |
| # Dimensions | 36 | 32 | 21 | 36 |
| Outlier Perc | 1.22 | 6.25 | 9.61 | 31.64 |
| Default | 0.5699 | 0.8631 | 0.5898 | 0.5605 |
| n_neighbors=10/leaf_size=20 | 0.57 | 0.9 | 0.48 | 0.54 |
| n_neighbors=10/leaf_size=30 | 0.57 | 0.9 | 0.48 | 0.54 |
| n_neighbors=10/leaf_size=40 | 0.57 | 0.9 | 0.48 | 0.54 |
| n_neighbors=20/leaf_size=20 | 0.53 | 0.89 | 0.52 | 0.56 |
| n_neighbors=20/leaf_size=30 | 0.53 | 0.89 | 0.52 | 0.56 |
| n_neighbors=20/leaf_size=40 | 0.53 | 0.89 | 0.52 | 0.56 |
| n_neighbors=30/leaf_size=20 | 0.44 | 0.86 | 0.59 | 0.57 |
| n_neighbors=30/leaf_size=30 | 0.44 | 0.86 | 0.59 | 0.57 |
| n_neighbors=30/leaf_size=40 | 0.44 | 0.86 | 0.59 | 0.57 |

outlier 비율이 작은 데이터셋의 경우에는 n_neighbors가 10일 때 성능이 가장 좋았고, outlier 비율이 높은 경우에는 n_neighbors가 30인 경우의 성능이 좋았다. 또한 leaf_size는 모델 성능에 큰 영향을 주지 못하는 것을 알 수 있었다. default setting과 비교했을 때, letter를 제외하면 큰 차이는 없었다.

2) IF

**Default setting** 

- n_estimators = 100
- max_samples = min(256, n_samples)

**Hyperparameter setting**

- n_estimators = [50,100,200]
- max_samples = [128,256,512]

n_estimatiors는 base estimators의 수를 의미하고, max_samples은 sample의 수를 의미한다. 강의 자료에서 샘플은 256개면 충분하다는 내용이 있었기에 이를 바탕으로 default가 256으로 설정된 것 같았고 이를 넘으면 성능이 좋아지는지 아닌지를 확인해보고자 했다.

**결과**

| Data | satimage-2 | letter | cardio | satellite |
| --- | --- | --- | --- | --- |
| # Samples | 5803 | 1600 | 1831 | 6435 |
| # Dimensions | 36 | 32 | 21 | 36 |
| Outlier Perc | 1.22 | 6.25 | 9.61 | 31.64 |
| Default | 0.9969 | 0.5551 | 0.93 | 0.701 |
| n_estimators=50/max_samples=128 | 1.0 | 0.43 | 0.89 | 0.66 |
| n_estimators=50/max_samples=256 | 1.0 | 0.56 | 0.93 | 0.73 |
| n_estimators=50/max_samples=512 | 1.0 | 0.59 | 0.88 | 0.69 |
| n_estimators=100/max_samples=128 | 1.0 | 0.53 | 0.93 | 0.67 |
| n_estimators=100/max_samples=256 | 1.0 | 0.57 | 0.91 | 0.67 |
| n_estimators=100/max_samples=512 | 1.0 | 0.57 | 0.91 | 0.66 |
| n_estimators=200/max_samples=128 | 1.0 | 0.49 | 0.92 | 0.65 |
| n_estimators=200/max_samples=256 | 1.0 | 0.51 | 0.92 | 0.67 |
| n_estimators=200/max_samples=512 | 1.0 | 0.56 | 0.89 | 0.67 |

satimage-2 데이터의 경우에는 hyperparameter와 관계없이 성능이 최대를 기록했다. 나머지 세 데이터셋에 대해서는 모두 n_estimators가 50일 때 가장 좋은 성능을 보였다. 또한 max_samples의 경우에는 뚜렷한 특징을 보이지는 않았는데 sample의 수에 따라 성능이 드라마틱하게 변화하지는 않았다. 하지만 sample의 수에 따라 학습 소요 시간이 다르기 때문에 작은 sample을 사용하는 것이 좋고, 이러한 기준과 실험에 따라서 논문에서 256개로 설정한 것으로 생각된다. 데이터의 크기를 더욱 늘리면 어떤 결과를 보일지 실험해보는 것도 좋은 실험이 될 것 같다. 마찬가지로 default setting과 비교했을 때 letter를 제외하면 큰 차이는 없었다.

3) AE

**Default setting** 

- hidden_neurons = [32, 16, 2, 2, 16, 32]
- hidden_activation='relu’

**Hyperparameter setting**

- hidden_neurons = [[64,16,16,64],[64,8,8,64],[64,4,4,64]]
- hidden_activation = ['relu', 'tanh']

hidden_neurons는 encoder와 decoder의 층별 크기를 의미하고, hidden_activation은 각 hidden layer별 활성화 함수를 의미한다. 층별 node size에 따라 어떤 변화를 보이는지를 중점적으로 비교해보려 했다.

**결과**

| Data | satimage-2 | letter | cardio | satellite |
| --- | --- | --- | --- | --- |
| # Samples | 5803 | 1600 | 1831 | 6435 |
| # Dimensions | 36 | 32 | 21 | 36 |
| Outlier Perc | 1.22 | 6.25 | 9.61 | 31.64 |
| Default | 0.9827 | 0.4845 | 0.9476 | 0.6123 |
| hidden_neurons=[64, 16, 16, 64]/hidden_activation=relu | 0.99 | 0.66 | 0.94 | 0.62 |
| hidden_neurons=[64, 16, 16, 64]/hidden_activation=tanh | 0.99 | 0.66 | 0.94 | 0.62 |
| hidden_neurons=[64, 8, 8, 64]/hidden_activation=relu | 0.99 | 0.66 | 0.94 | 0.62 |
| hidden_neurons=[64, 8, 8, 64]/hidden_activation=tanh | 0.99 | 0.66 | 0.94 | 0.63 |
| hidden_neurons=[64, 4, 4, 64]/hidden_activation=relu | 0.99 | 0.66 | 0.94 | 0.62 |
| hidden_neurons=[64, 4, 4, 64]/hidden_activation=tanh | 0.99 | 0.66 | 0.94 | 0.62 |

모든 데이터셋에 대해 hyperparameter별로 성능이 거의 변화하지 않았다. layer의 수나 다른 parameter들이 더 중요한 역할을 할 것으로 예상된다. 마찬가지로 default setting과 비교했을 때 letter를 제외하면 큰 차이는 없었다.

4) OCSVM

**Default setting** 

- kernel='rbf’
- nu=0.5

**Hyperparameter setting**

- kernel = ['poly','rbf','sigmoid']
- nu = [0.25,0.5,0.75]

kernel은 svm의 kernel function을 의미하고, nu는 svm의 hyperparameter이다. kernel과 nu가 가장 핵심 hyperparameter라고 판단하여 실험을 진행하였다

**결과**

| Data | satimage-2 | letter | cardio | satellite |
| --- | --- | --- | --- | --- |
| # Samples | 5803 | 1600 | 1831 | 6435 |
| # Dimensions | 36 | 32 | 21 | 36 |
| Outlier Perc | 1.22 | 6.25 | 9.61 | 31.64 |
| Default | 0.9997 | 0.5918 | 0.9271 | 0.6629 |
| kernel=poly/nu=0.25 | 0.02 | 0.79 | 0.18 | 0.48 |
| kernel=poly/nu=0.5 | 0.08 | 0.83 | 0.15 | 0.52 |
| kernel=poly/nu=0.75 | 0.03 | 0.82 | 0.13 | 0.54 |
| kernel=rbf/nu=0.25 | 0.99 | 0.68 | 0.88 | 0.62 |
| kernel=rbf/nu=0.5 | 0.99 | 0.69 | 0.91 | 0.68 |
| kernel=rbf/nu=0.75 | 0.99 | 0.68 | 0.92 | 0.66 |
| kernel=sigmoid/nu=0.25 | 0.92 | 0.41 | 0.85 | 0.47 |
| kernel=sigmoid/nu=0.5 | 0.94 | 0.32 | 0.9 | 0.47 |
| kernel=sigmoid/nu=0.75 | 0.96 | 0.24 | 0.93 | 0.51 |

kernel에 따라서 성능 변화가 아주 컸고, nu에 따라서도 변화를 보였다. 또한 전반적으로 rbf kernel의 성능이 nu에 관계없이 가장 안정적이고 좋았다. 마찬가지로 default setting과 비교했을 때 letter를 제외하면 큰 차이는 없었다.

결론적으로 일관적인 특징은 letter 데이터셋을 제외하면 성능 변화가 크지 않는 것을 보아 역시 hyperparameter보다는 데이터의 특성이 더 중요하고 letter의 데이터셋의 특성은 다른 데이터셋과 다를 것으로 예상된다.

### 4. 데이터셋/모델별 최종 성능 비교

- Default setting

| Data | #Samples | # Dimensions | Outlier Perc | LOF | IF | AE | OCSVM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| satimage-2 | 5803 | 36 | 1.2235 | 0.5699 | 0.9969 | 0.9827 | 0.9997 |
| letter | 1600 | 32 | 6.25 | 0.8631 | 0.5551 | 0.4845 | 0.5918 |
| cardio | 1831 | 21 | 9.6122 | 0.5898 | 0.93 | 0.9476 | 0.9271 |
| satellite | 6435 | 36 | 31.6395 | 0.5605 | 0.701 | 0.6123 | 0.6629 |
- Hyperparameter tuning

| Data | #Samples | # Dimensions | #outlier perc | LOF | IF | AE | OCSVM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| satimage-2 | 5803 | 36 | 1.22 | 0.56 | 1.0 | 0.99 | 0.99 |
| letter | 1600 | 32 | 6.25 | 0.9 | 0.59 | 0.66 | 0.83 |
| cardio | 1831 | 21 | 9.61 | 0.59 | 0.93 | 0.94 | 0.93 |
| satellite | 6435 | 36 | 31.64 | 0.57 | 0.73 | 0.63 | 0.68 |

letter 데이터를 제외하고는 hyperparameter tuning후에도 성능이 크게 변화하지 않았다.

### 5. 결론

실험 2의 결론을 최종 결론으로 실험을 마무리 하였다. 

[실험 2 결론 내용]

> satimage-2 데이터부터 살펴보면 실험 1의 결과와는 다르게 LOF를 제외한 모델의 성능이 아주 높게 나왔으며, LOF의 성능은 낮게 나왔다. 하지만 letter 데이터를 살펴보면 LOF의 성능이 나머지 세 모델의 성능보다 높게 나온 것이 실험1의 결과와 일치했다. 여기서 실험 1과의 가장 큰 차이점이라면 데이터의 수인데 satimage-2의 데이터 수는 5803개로 letter보다 훨씬 작기 크기 때문에 model-based 모델들의 성능이 높은 것으로 예상되고 task가 아주 쉬웠을 것으로 예상된다. 따라서 outlier 비율 못지 않게 **데이터의 수**가 모델 성능에 큰 영향을 주는 것을 알 수 있다. cardio와 satellite 데이터의 결과를 보면 실험 1과 같은 결과를 보이는 것을 알 수 있다. outlier 비율이 10% 정도일 때 model-based 모델들의 성능이 매우 높고 LOF의 성능이 낮았다. 또한 outlier 비율이 30% 이상인 satellite의 결과를 보면 데이터의 수가 많음에도 불구하고 성능이 다 떨어진 것을 확인할 수 있었다. 결론적으로 outlier 비율이 약 5% 이하일 때는 데이터의 수가 모델 성능에 더 큰 영향을 주고, outlier 비율이 약 10% 일때는 model-based 모델들의 성능이 LOF보다 높은 것을 확인할 수 있었다. 또한 outlier 비율이 30% 이상인 outlier가 많은 데이터의 경우에는 데이터 수와 관계 없이 모델 성능이 모두 악화되었지만 상대적으로 model-based 모델들의 성능이 더 높은 것을 알 수 있었다.
> 

더 많은 모델과 데이터셋 그리고 데이터의 수에 따라서도 실험을 진행하면 더 흥비로운 실험 결과를 얻을 수 있을 것 같다. 

reference

[https://journalofbigdata.springeropen.com/articles/10.1186/s40537-017-0101-8/figures/4](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-017-0101-8/figures/4)

[https://tp46.github.io/general/2018/11/27/model-based-novelty-detection/](https://tp46.github.io/general/2018/11/27/model-based-novelty-detection/)

[https://github.com/damjankuznar/pylof/blob/master/lof.py](https://github.com/damjankuznar/pylof/blob/master/lof.py)

[https://github.com/L1aoXingyu/pytorch-beginner/blob/master/08-AutoEncoder/conv_autoencoder.py](https://github.com/L1aoXingyu/pytorch-beginner/blob/master/08-AutoEncoder/conv_autoencoder.py)

[https://github.com/tianqwang/isolation-forest-from-scratch/blob/4dc10d7d9f8a9633d1582c864ee094b9204e362b/iforest.py#L27](https://github.com/tianqwang/isolation-forest-from-scratch/blob/4dc10d7d9f8a9633d1582c864ee094b9204e362b/iforest.py#L27)
