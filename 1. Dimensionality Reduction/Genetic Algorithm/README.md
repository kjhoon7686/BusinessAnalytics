# Genetic Algorithm

![Untitled](https://github.com/kjhoon7686/BusinessAnalytics/blob/main/1.%20Dimensionality%20Reduction/Genetic%20Algorithm/images/Untitled.png)

전진선택법, 후진소거법, 단계적선택법 등의 휴리스틱 기반 변수 선택 기법들은 소요시간이 적게 들지만 모델 성능의 최대치를 달성하기 힘들다는 단점이 있고, 전역 탐색법은 모델 성능의 최대치를 달성할 수 있지만 소요시간이 많이 든다는 단점이 있다. 휴리스틱 기반 변수 선택 기법들보다 소요시간은 더 많이 들지만 모델 성능을 전역 탐색법에 더 가깝게 달성하기 위한 방법이 유전 알고리즘이다.

- 유전 알고리즘 : 생명체의 생식 과정을 모사한 진화 알고리즘의 일종으로 자연선택법에 기반을 두고 있음.
- 유전 알고리즘의 세가지 핵심 단계
    1. 선택(selection) : 현재 가능 해집합에서 우수한 해들을 선택하여 다음 세대를 생성하기 위한 부모 세대로 지정
    2. 교배(crossover) : 선택된 부모 세대들의 유전자를 교환하여 새로운 세대를 생성
    3. 돌연변이(mutation) : 낮은 확률로 변이를 발생시켜 Local Optimaum에서 탈출할 수 있는 기회 제공

- 변수 선택을 위한 유전 알고리즘 절차

![Untitled](https://github.com/kjhoon7686/BusinessAnalytics/blob/main/1.%20Dimensionality%20Reduction/Genetic%20Algorithm/images/Untitled%201.png)

## ****Step1: 염색체 초기화 및 하이퍼파라미터 설정(initialization)****

먼저 유전알고리즘에 대한 기본적인 용어를 살펴보자

![Untitled](https://github.com/kjhoon7686/BusinessAnalytics/blob/main/1.%20Dimensionality%20Reduction/Genetic%20Algorithm/images/Untitled%202.png)

- 해집단(population) : 가능한 염색체들의 조합으로 이루어지며, 정해진 수의 염색체 집단을 의미한다.
- 염색체(chromosome) : 한 세대 혹은 해집단에서 가능한 하나의 유전적 표현을 의미한다. 염색체는 유전자들의 집합이다.
- 유전자(gene) : 유전자는 염색체의 구성요소이다. 유전 알고리즘에서의 최소 단위이며 0과 1로 인코딩된다.
- 적합도(fitness) : 염색체의 우열을 가릴 수 있는 정량적 지표이다. 적합도는 두 염색체가 동일한 예측 성능을 나타낼 경우에는 적은 수의 변수를 사용한 염색체를 선호해야 하고, 두 염색체가 동일한 변수를 사용했을 경우에는 우수한 예측 성능을 나타내는 염색체를 선호해야 한다. 선형회귀분석에서는 adjusted R-square, AIC, BIC 등이 사용될 수 있고, 분류 문제에서는 정확도 등을 사용할 수 있다.

![Untitled](https://github.com/kjhoon7686/BusinessAnalytics/blob/main/1.%20Dimensionality%20Reduction/Genetic%20Algorithm/images/Untitled%203.png)

먼저 하이퍼파라미터를 설정한 후 random하게 chromosome을 초기화해주어야 한다.

- size : chromosome의 수
- n_feat : data의 변수 개수
- cut_off : 1과 0의 비율

```python
# population 초기화 함수
# size : chromosome의 수(population size), n_feat : feature의 수, cutoff : 이진값 변환 기준값
def initilization_of_population(size,n_feat,cutoff): 
    population = []
    for i in range(size):
        chromosome = np.ones(n_feat,dtype=np.bool) # n_feat 수만큼의 array 생성 
        chromosome[:int(cutoff*n_feat)]=False # default : 0.3 = 전체 feature에서 0의 비율         
        np.random.shuffle(chromosome) # 생성된 chromosome 무작위 shuffle
        population.append(chromosome) # population에 chromosome 추가
    return population
```

## **Step2: 각 염색체 선택 변수별 모델 학습**

각 염색체 선택 변수별 학습할 모델을 선택해주어야 한다. 여기서는 logistic regression 모델을 사용했다. 

```python
logmodel = LogisticRegression(max_iter = 1000) # 선택한 model
```

## **Step3: 각 염색체 적합도 평가(Fitness evaluation)**

선택한 모델을 바탕으로 학습된 모형의 적합도를 평가한다.

- logmodel : 선택한 모델

```python
# 적합도 평가 함수
def fitness_score(population):
    scores = []
    for chromosome in population:
        logmodel.fit(X_train.iloc[:,chromosome],Y_train) # 생성된 chromosome에 해당하는 feature들로 모델 학습       
        predictions = logmodel.predict(X_test.iloc[:,chromosome]) # 생성된 chromosome에 해당하는 feature들로 예측
        scores.append(accuracy_score(Y_test,predictions)) # accuracy 계산 및 저장
    scores, population = np.array(scores), np.array(population) 
    inds = np.argsort(scores)                                    
    return list(scores[inds][::-1]), list(population[inds,:][::-1])
```

## **Step4: 우수 염색체 선택(Selection)**

현재 세대에서 높은 적합도를 나타내는 염색체들을 부모로 선택하여 다음 세대의 염색체를 생성하는데 사용한다. 선택 방법에는 확정적 선택법과 확률적 선택법이 있다.

- 확정적 선택
    - 적합도 기준 상위 N%(혹은 N개)에 해당하는 염색체만 부모 염색체로 사용한다.
    - 하위에 해당하는 염색체는 부모 염색체로 사용되지 않는다.
- 확률적 선택
    - 적합도 함수에 비례하는 가중치를 사용하여 부모 염색체를 선택한다.
    - 모든 염색체가 부모 염색체로 선택될 가능성이 있지만, 적합도 함수가 낮을수록 선택 가능성 역시 낮아진다.

다음 코드에서는 확정적 선택법을 이용하였다.

- pop_after_fit : fitness_score 평가 후 score가 높은 순으로 정렬된 population
- n_parents : 부모 염색체로 사용할 염색체의 수

```python
# 확정적 선택법
# pop_after_fit : fitness_score 평가 후 score 높은 순대로 정렬된 population, n_parents : population에서 샘플링할 염색체의 수
def selection(pop_after_fit,n_parents): 
    population_nextgen = []
    for i in range(n_parents):
        population_nextgen.append(pop_after_fit[i])
    return population_nextgen
```

## **Step5: 다음 세대 염색체 생성(Crossover & Mutation)**

step5는 crossover와 mutation을 활용해 다음 세대 염색체를 생성하는 방법이다.

- 교배(crossover)
    - 두 부모 염색체가 있다고 할 때,
    - one point cross over는 한 지점에서만 유전자를 교배하고
    - multi point cross over는 여러 지점에서 유전자를 교배한다.

![Untitled](https://github.com/kjhoon7686/BusinessAnalytics/blob/main/1.%20Dimensionality%20Reduction/Genetic%20Algorithm/images/Untitled%204.png)

다음 코드에서는 one point cross over를 사용하였다.

- selection 후의 population에서 순서대로 두 염색체를 절반씩 결합하여 교배시킨 염색체를 population에 추가한다.

```python
# 교배
# selection후 population에 순서대로 두 염색체를 절반씩 결합하여 crossover된 chromosome추가
def crossover(pop_after_sel):
    pop_nextgen = pop_after_sel
    for i in range(0,len(pop_after_sel),2):
        new_par = []
        child_1 , child_2 = pop_nextgen[i] , pop_nextgen[i+1]
        new_par = np.concatenate((child_1[:len(child_1)//2],child_2[len(child_1)//2:]))
        pop_nextgen.append(new_par)
    return pop_nextgen
```

- 돌연변이(mutation)
    - 돌연변이는 세대가 진화해가는 과정에서 다양성을 확보하기 위한 장치이다.
    - 특정 유전자의 정보를 낮은 확률로 반대값으로 변환하다.
    - 현재 해가 local optima에서 탈출할 수 있는 기회를 제공한다.

![Untitled](https://github.com/kjhoon7686/BusinessAnalytics/blob/main/1.%20Dimensionality%20Reduction/Genetic%20Algorithm/images/Untitled%205.png)

- mutation_rate : 돌연변이의 비율

```python
# 돌연변이
# mutaion rate만큼 유전자를 랜덤으로 바꿈
def mutation(pop_after_cross,mutation_rate,n_feat):   
    mutation_range = int(mutation_rate*n_feat)
    pop_next_gen = []
    for n in range(0,len(pop_after_cross)):
        chromo = pop_after_cross[n]
        rand_posi = [] 
        for i in range(0,mutation_range):
            pos = randint(0,n_feat-1)
            rand_posi.append(pos)
        for j in rand_posi:
            chromo[j] = not chromo[j]  
        pop_next_gen.append(chromo)
    return pop_next_gen
```

## **Step6: 최종 변수 집합 선택**

step2-step5의 과정을 원하는 만큼 반복 수행하고, 가장 좋은 성능을 낸 population을 선택한다. 해당 population이 최종적인 염색체 조합이 된다.

- n_gen : 반복 수

```python
# step1 to step 6 실행
# n_gen : 반복 수
def generations(df,label,size,n_feat,cutoff,n_parents,mutation_rate,n_gen,X_train,
                                   X_test, Y_train, Y_test):
    best_chromo= []
    best_score= []
    population_nextgen=initilization_of_population(size,n_feat,cutoff)
    for i in range(n_gen):
        scores, pop_after_fit = fitness_score(population_nextgen)
        print('Best score in generation',i+1,':',scores[:1])  
        pop_after_sel = selection(pop_after_fit,n_parents)
        pop_after_cross = crossover(pop_after_sel)
        population_nextgen = mutation(pop_after_cross,mutation_rate,n_feat)
        best_chromo.append(pop_after_fit[0])
        best_score.append(scores[0])
    return best_chromo,best_score
```
