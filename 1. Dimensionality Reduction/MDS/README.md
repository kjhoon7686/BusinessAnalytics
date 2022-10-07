# Multidimensional Scaling(MDS)

# Multidimensional Scaling

: 관측치 간의 거리를 기반으로 고차원에서의 거리를 저차원의 거리로 축소하여 표현하는 방법

## PCA vs MDS

![Untitled](https://github.com/kjhoon7686/BusinessAnalytics/blob/main/1.%20Dimensionality%20Reduction/MDS/images/Untitled.png)

## Step 1 : Distance Matrix 만들기

- distance의 성질

<img src="https://github.com/kjhoon7686/BusinessAnalytics/blob/main/1.%20Dimensionality%20Reduction/MDS/images/Untitled%201.png" width="50%" height="50%"/>

mds의 과정은 간단하다.

주어진 데이터 행렬 (nxr)을 이용하여

거리 행렬(nxn)을 만들고 이를 사용하여 원하는 차원의

mds행렬(nxp)을 만든다.

즉, O(original matrix) → D(distance matrix) → B(inner product matrix) → X(mds matrix)의 과정으로 이루어진다. 

예제 코드는 iris 데이터를 사용하였다.

관측치 x의 feature value를 사용하여 유사도 행렬을 만든다.

```python
# step 1
d_mat = np.zeros((len(X),len(X)))
print(d_mat,'\n')

for i in range(len(X)):
    for j in range(len(X)):
        d_mat[i,j] = (np.sum((X[i,:]-X[j,:])**2)) 
print(d_mat)
```

## Step 2 : inner product matrix B 계산

- [https://www.dropbox.com/s/sgg7d9s6mxxtu41/01_3_Dimensionality Reduction_PCA and MDS.pdf?dl=0](https://www.dropbox.com/s/sgg7d9s6mxxtu41/01_3_Dimensionality%20Reduction_PCA%20and%20MDS.pdf?dl=0) (자세한 증명)

<img src="https://github.com/kjhoon7686/BusinessAnalytics/blob/main/1.%20Dimensionality%20Reduction/MDS/images/Untitled%202.png" width="20%" height="20%"/>

<img src="https://github.com/kjhoon7686/BusinessAnalytics/blob/main/1.%20Dimensionality%20Reduction/MDS/images/Untitled%203.png" width="20%" height="20%"/>

<img src="https://github.com/kjhoon7686/BusinessAnalytics/blob/main/1.%20Dimensionality%20Reduction/MDS/images/Untitled%204.png" width="20%" height="20%"/>

<img src="https://github.com/kjhoon7686/BusinessAnalytics/blob/main/1.%20Dimensionality%20Reduction/MDS/images/Untitled%205.png" width="20%" height="20%"/>

거리 행렬 D를 사용하여 내적 행렬 B를 만든다.

```python
# step 2
n = len(d_mat)
H = np.eye(n)-(1/n)*(np.ones((n,n)))
B = -H.dot(d_mat).dot(H)/2
```

## Step 3 : B의 **eigenvector, eigenvalue를 통해 coordinate X(nxp) 구하기**

![Untitled](https://github.com/kjhoon7686/BusinessAnalytics/blob/main/1.%20Dimensionality%20Reduction/MDS/images/Untitled%206.png)

B는 rank p의 psd이기 때문에 p개의 non-negative eigenvalue와 n-p개의 zero eigenvalue를 갖는다.

또한 B는 n-p개의 zero eigenvalue를 갖기 때문에 B를 pxp matrix로 다시 표현할 수 있다.

이를 이용하여 X를 구할 수 있게 된다. 

```python
# step 3
eigen_value, eigen_vector = np.linalg.eig(B)
inverseEigenVectors = np.linalg.inv(eigen_vector) 
diagonal = inverseEigenVectors.dot(B).dot(eigen_vector)
```

```python
dimension = 2 # 축소할 차원 수 지정

B_1 = eigen_vector[:,0:dimension].dot(diagonal[0:dimension,0:dimension]).dot(eigen_vector[:,0:dimension].T)
diagonal[diagonal<1] = 0
coordinate_X = eigen_vector[:,0:dimension].dot(np.sqrt(diagonal[0:dimension,0:dimension]))
print(coordinate_X.shape)
```

## 시각화

```python
# 시각화
y = df['class'].values 

with plt.style.context("seaborn-darkgrid"):
    for l in np.unique(y):
        plt.scatter(coordinate_X[y==l,0], coordinate_X[y==l,1],label=l)
    plt.xlabel("dimension 1")
    plt.ylabel("dimension 2") 
    plt.legend()
    plt.show()
```

![Untitled](https://github.com/kjhoon7686/BusinessAnalytics/blob/main/1.%20Dimensionality%20Reduction/MDS/images/Untitled%207.png)

- reference

[https://ljhz123.github.io/2018/10/22/PCA+MDS.html](https://ljhz123.github.io/2018/10/22/PCA+MDS.html)

[https://towardsdatascience.com/mds-multidimensional-scaling-smart-way-to-reduce-dimensionality-in-python-7c126984e60b](https://towardsdatascience.com/mds-multidimensional-scaling-smart-way-to-reduce-dimensionality-in-python-7c126984e60b)
