# ****ISOMAP, LLE, t-SNE****

# ISOMAP(Isometric Feature Mapping)

![Untitled](https://github.com/kjhoon7686/BusinessAnalytics/blob/main/1.%20Dimensionality%20Reduction/ISOMAP%2C%20LLE%2C%20t-SNE/images/Untitled.png)

Isomap은 pca와 mds의 특징을 결합한 알고리즘이다. 좌측 그림에서의 euclidean distance는 짧지만 데이터의 실제 특성을 반영하는 manifold에서는 우측 그림에서처럼 거리가 멀다. isomap은 neighborhood graph를 통해 고차원에서의 intrinsic distance를 찾고 이를 축소된 차원에서 표현하는 방법론이다.

Isomap procedure

- Step 1 : neighborhood graph를 만든다.

neighborhood graph를 만드는 방식은 ϵ-isomap과 k-isomap 두 가지 방식이 있다. ϵ-isomap은 고정된 ϵ의 값보다 두 점이 가까우면 두 점이 neighborhood라고 생각하고 연결하는 방법이다. 반면 k-isomap은 j에서 가까운 k개의 포인트에 i가 포함되면 연결하는 방식이다.

```python
class Graph(object):
    def __init__(self):
        self.nodes = set()
        self.edges = {}
        self.distances = {}
    def add_node(self, value):
        self.nodes.add(value)
    def add_edge(self, from_node, to_node, distance):
        self._add_edge(from_node, to_node, distance)
        self._add_edge(to_node, from_node, distance)
    def _add_edge(self, from_node, to_node, distance):
        self.edges.setdefault(from_node, [])
        self.edges[from_node].append(to_node)
        self.distances[(from_node, to_node)] = distance
```

- Step2 : 포인트들 간의 최단 거리를 계산한다.

![Untitled](https://github.com/kjhoon7686/BusinessAnalytics/blob/main/1.%20Dimensionality%20Reduction/ISOMAP%2C%20LLE%2C%20t-SNE/images/Untitled%201.png)

step 1에서 만든 neighborhood graph를 바탕으로 distance matrix를 계산한다. 이때 모든 데이터 포인트들 간의 거리를 구해야 하기 때문에 계산 복잡도가 높다.

```python
def dijkstra(graph, initial_node):
    visited_dist = {initial_node: 0}
    nodes = set(graph.nodes)
    while nodes:
        connected_node = None
        for node in nodes:
            if node in visited_dist:
                if connected_node is None:
                    connected_node = node
                elif visited_dist[node] < visited_dist[connected_node]:
                    connected_node = node
        if connected_node is None:
            break
        nodes.remove(connected_node)
        cur_wt = visited_dist[connected_node]
        for edge in graph.edges[connected_node]:
            wt = cur_wt + graph.distances[(connected_node, edge)]
            if edge not in visited_dist or wt < visited_dist[edge]:
                visited_dist[edge] = wt
    return visited_dist
```

- Step 3 : MDS embedding

distance matrix에 MDS를 사용하여 차원을 줄인다. 

다음 코드에서는 kernel pca를 사용하였다

```python
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.decomposition import KernelPCA
from dijkstra import Graph, dijkstra
import numpy as np
import pickle

def isomap(input, n_neighbors, n_components, n_jobs):
    distance_matrix = pickle.load(open('./isomap_distance_matrix.p', 'rb'))
    kernel_pca_ = KernelPCA(n_components=n_components,
                                 kernel="precomputed",
                                 eigen_solver='arpack', max_iter=None,
                                 n_jobs=n_jobs)
    Z = distance_matrix ** 2
    Z *= -0.5
    embedding = kernel_pca_.fit_transform(Z)
    return(embedding)
```

![Untitled](https://github.com/kjhoon7686/BusinessAnalytics/blob/main/1.%20Dimensionality%20Reduction/ISOMAP%2C%20LLE%2C%20t-SNE/images/Untitled%202.png)

![Untitled](https://github.com/kjhoon7686/BusinessAnalytics/blob/main/1.%20Dimensionality%20Reduction/ISOMAP%2C%20LLE%2C%20t-SNE/images/Untitled%203.png)

reference

[https://woosikyang.github.io/first-post.html](https://woosikyang.github.io/first-post.html)

[https://gyubin.github.io/ml/2018/10/26/non-linear-embedding](https://gyubin.github.io/ml/2018/10/26/non-linear-embedding)

# Locally Linear Embedding(LLE)

LLE는 고차원 공간의 인접 데이터들을 선형적 구조를 유지한 채 저차원으로 임베딩하는 방법이다. 매니폴드 학습의 한 종류이다. LLE의 특징은 다음과 같다.

- 구현이 간단하다.
- local optima에 빠지지 않는다.
- 매우 비선형적인 임베딩을 생성할 수 있다.
- 고차원의 데이터를 저차원 좌표계에 표현할 수 있다.

LLE Procedure

![Untitled](https://github.com/kjhoon7686/BusinessAnalytics/blob/main/1.%20Dimensionality%20Reduction/ISOMAP%2C%20LLE%2C%20t-SNE/images/Untitled%204.png)

- Step 1 : 각 데이터 포인트에서 k개의 이웃 계산
- Step 2 : 가중치 행렬 구성

neighborhood로부터 각 데이터를 가장 잘 표현할 수 있는 가중치 행렬을 계산한다.

![Untitled](https://github.com/kjhoon7686/BusinessAnalytics/blob/main/1.%20Dimensionality%20Reduction/ISOMAP%2C%20LLE%2C%20t-SNE/images/Untitled%205.png)

- Step 3 : 가중치 행렬 W를 가장 잘 보존할 수 있는 임베딩 벡터 계산

![Untitled](https://github.com/kjhoon7686/BusinessAnalytics/blob/main/1.%20Dimensionality%20Reduction/ISOMAP%2C%20LLE%2C%20t-SNE/images/Untitled%206.png)

- 예시

![Untitled](https://github.com/kjhoon7686/BusinessAnalytics/blob/main/1.%20Dimensionality%20Reduction/ISOMAP%2C%20LLE%2C%20t-SNE/images/Untitled%207.png)

## t-SNE

1) SNE

t-SNE가 무엇인지 알기 위해서는 SNE부터 알아보아야 한다. SNE(Stochastic Neighbor Embedding)는 local pairwise distance를 확률적으로 정의하여 차원을 축소한다. 두 객체 간의 거리는 고차원에서 이웃이 될 확률과 저차원에서 이웃이 될 확률 두 종류로 가우시안 분포를 기준으로 하여 정의한다. 이 두 종류의 거리가 일치하도록 하는 것이 t-SNE의 목표이다. 

![Untitled](https://github.com/kjhoon7686/BusinessAnalytics/blob/main/1.%20Dimensionality%20Reduction/ISOMAP%2C%20LLE%2C%20t-SNE/images/Untitled%208.png)

2) KL-divergence

![Untitled](https://github.com/kjhoon7686/BusinessAnalytics/blob/main/1.%20Dimensionality%20Reduction/ISOMAP%2C%20LLE%2C%20t-SNE/images/Untitled%209.png)

KL-divergence는 두 확률 분포 사이의 차이를 계산해주는 비대칭 지표이다. SNE에서 언급했던 두 종류의 거리를 KL-divergence를 통해 비교하여 이를 최소화하는 방향으로 학습이 진행된다. 학습은 gradient descent를 통해 수행된다.

![Untitled](https://github.com/kjhoon7686/BusinessAnalytics/blob/main/1.%20Dimensionality%20Reduction/ISOMAP%2C%20LLE%2C%20t-SNE/images/Untitled%210.png)

3) t-SNE

가우시안 분포는 평균에 가까운 부분의 확률값이 높고 꼬리쪽으로 갈수록 확률이 급격하게 낮아진다는 특성을 보인다. 그래서 일반적인 SNE를 사용하면 일정 거리를 넘어가면 거리에 상관없이 선택될 확률의 차이가 매우 적어진다. 이를 Crowding problem이라고 부른다. 따라서 t-SNE는 이를 보완하기 위해 만들어진 방식으로 가우시안 분포 대신 꼬리쪽의 분포가 두꺼운 t 분포를 사용한다. t-SNE는 고차원에서의 거리는 가우시안 분포를 사용하고 저차원에서의 거리는 t 분포를 사용한다. 

![Untitled](https://github.com/kjhoon7686/BusinessAnalytics/blob/main/1.%20Dimensionality%20Reduction/ISOMAP%2C%20LLE%2C%20t-SNE/images/Untitled%2011.png)

- 예시

![Untitled](https://github.com/kjhoon7686/BusinessAnalytics/blob/main/1.%20Dimensionality%20Reduction/ISOMAP%2C%20LLE%2C%20t-SNE/images/Untitled%2012.png)

![Untitled](https://github.com/kjhoon7686/BusinessAnalytics/blob/main/1.%20Dimensionality%20Reduction/ISOMAP%2C%20LLE%2C%20t-SNE/images/Untitled%2013.png)

- reference

[https://gyubin.github.io/ml/2018/10/26/non-linear-embedding](https://gyubin.github.io/ml/2018/10/26/non-linear-embedding)

[https://woosikyang.github.io/first-post.html](https://woosikyang.github.io/first-post.html)
