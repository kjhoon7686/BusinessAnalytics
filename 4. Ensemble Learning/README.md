# Ensemble Learning

# 목차

## 개념

1. Random Forest (RF)
2. Gradient Boosting Machine (GBM)
3. Extreme Gradient Boosting Machine (XGBM)
4. Light Gradient Boosting Machine (LGBM)

## 실험

1. 데이터수에 따른 데이터셋별 모델 성능 비교 [성능, 시간] 
2. 데이터셋별 하이퍼파라미터에 따른 모델 성능 비교
3. 데이터셋별 feature importance 비교
4. 데이터셋별 hyperparameter tuning 전후 결과 비교
5. 결론

## 개념

이론에 대한 자세한 내용은 고려대학교 산업공학과 강필성 교수님의 “Business Analytics”강의를 참고하시길 바랍니다.

### 1. Random Forest (RF)

Random Forest는 의사결정나무 기반 앙상블의 특수한 형태로서 앙상블의 핵심인 Diversity 확보를 위해 두 가지 방법을 사용한다.

- Bagging
- Randomly chosen predictor variable

![image](https://user-images.githubusercontent.com/79893946/204269390-702fa0d4-c752-4d02-bdc9-3fad847652e0.png)

![image](https://user-images.githubusercontent.com/79893946/204269429-71e90e40-217f-4a8a-aa56-59c3b58125b4.png)

### 2. Gradient Boosting Machine (GBM)

Gradient Boosting Machine은 gradient descent 방법을 boosting에 적용한 모델이다. 개별 모델을 sequential한 방식으로 학습시키고, 각 단계의 Base learner들이 이전 단계의 Base learner의 약점을 보완한다는 특징을 가지고 있다. 이전 단계의 Base learner의 약점은 손실 함수의 gradient에 반영된다.

![image](https://user-images.githubusercontent.com/79893946/204269480-ceea89f6-331e-46fd-9e7f-c3336dbf8f17.png)

Gradient Boosting Machine의 알고리즘은 다음과 같다.

![image](https://user-images.githubusercontent.com/79893946/204269531-d314a1e0-4b43-49f3-8c90-a64105858918.png)

![image](https://user-images.githubusercontent.com/79893946/204269574-105c149d-2bc4-4d64-8a19-9e1f45f3bc1f.png)

![image](https://user-images.githubusercontent.com/79893946/204269608-1b2123de-b76d-458d-9358-2341468f8af4.png)

### 3. Extreme Gradient Boosting Machine (XGBM)

XGBoost는 GBM에 overfitting regularization효과를 중심으로 해서 여러 장점을 추가한 모델이다.

![image](https://user-images.githubusercontent.com/79893946/204269656-85a1fd38-3d6d-4ad5-9c98-53d0fe24e473.png)

다음의 두 가지 split finding algorithm을 통해 모델이 구성된다.

- Approximation algorithm : 이 알고리즘은 미리 정해놓은 분할 수로 data를 분할한 후 split이 이루어지기 때문에 기존 모델에 대해 속도가 빨라진다는 장점이 있다.
- Sparsity-Aware algorithm : 이 알고리즘은 데이터가 sparse한 경우에 어떤 방법으로 처리할지를 미리 설정해놓는 방식으로 이루어진다.

![image](https://user-images.githubusercontent.com/79893946/204269696-19a95202-dcef-4e08-be82-c132e58ee2e4.png)

### 4. Light Gradient Boosting Machine (LGBM)

LGBM은 GBM이 모든 feature에 대해 모든 data instance를 scan해야 한다는 한계점을 해결하기 위해 등장한 모델이다.

이를 위해 다음의 두 가지 방법을 사용한다.

![image](https://user-images.githubusercontent.com/79893946/204269738-8a75ce74-cf8f-4b46-a090-3476c510fb63.png)

또한 아래 그림과 같이 일반적인 Boosting알고리즘과 달리 Leaf-wise tree의 형태를 보이는데 이로 인해 속도가 매우 빠르다고 알려져 있다.

![image](https://user-images.githubusercontent.com/79893946/204269768-d7e38530-9534-49f8-b9b4-8fb580cb2bfd.png)

## 실험

### 1. 데이터 수에 따른 데이터셋별 모델 성능 비교 [성능, 시간]

**데이터셋 설명**

데이터셋은 데이터 수에 따른 모델별 성능에 차이가 있는지를 알아보기 위해 데이터 수가 다른 5가지 binary classification 데이터셋을 사용했다.

| Dataset | # data | # feature |
| --- | --- | --- |
| titanic | 891 | 11 |
| nba | 1340 | 21 |
| wine | 6498 | 13 |
| electrical_grid | 10000 | 14 |
| employee | 14999 | 10 |
- titanic : 승객의 정보를 feature로 하여 생존 여부를 예측하는 데이터셋
- nba : 선수의 지표를 feature로 하여 5년 후에도 리그에 남아있을지를 예측하는 데이터셋
- wine : wine의 특성을 feature로 하여 wine의 type을 예측하는 데이터셋
- electrical_grid : 시스템의 속성값들을 feature로 하여 system의 안정성을 예측하는 데이터셋
- employee : employee의 특성을 feature로 하여 퇴사여부를 예측하는 데이터셋

**모델 선정 이유**

모델은 bagging계열 모델인 RF와 boosting 계열 모델인 GBM, XGBM, LGBM을 사용했다. bagging과 boosting의 차이를 비교하고 GBM의 문제점을 해결한 모델인 XGBM과 LGBM을 비교하기 위해 위의 4가지 모델을 사용하였다.

**실험 세팅**

실험은 stratifiedkfold 방법을 사용하였고 10 fold로 진행하였다. 같은 데이터셋에서 test까지 나누는 것은 의미가 없다고 생각하여 kfold 방법으로 아래 모든 실험을 진행하였다.

**hyperparameter default value**

| Model | n_estimators | max_depth |
| --- | --- | --- |
| RF | 100 | None |
| GBM | 100 | 3 |
| XGBM | 100 | 6 |
| LGBM | 100 | -1 |

**데이터셋별 모델 성능 비교 [default setting] [Accuracy 높은 순 정렬]**

- **titanic [#data : 891]**

| Model | Accuracy | AUC | Recall | Prec. | F1 | Kappa | MCC | TT (Sec) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GBM | 0.8154 | 0.8575 | 0.6923 | 0.8140 | 0.7470 | 0.6030 | 0.6091 | 0.1210 |
| RF | 0.7994 | 0.8653 | 0.6763 | 0.7829 | 0.7241 | 0.5681 | 0.5731 | 0.4840 |
| LGBM | 0.7881 | 0.8487 | 0.6925 | 0.7540 | 0.7171 | 0.5489 | 0.5539 | 0.0340 |
| XGBM | 0.7752 | 0.8407 | 0.6882 | 0.7360 | 0.7061 | 0.5250 | 0.5303 | 0.5070 |
- **nba [#data : 1340]**

| Model | Accuracy | AUC | Recall | Prec. | F1 | Kappa | MCC | TT (Sec) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GBM | 0.6970 | 0.7465 | 0.8105 | 0.7374 | 0.7714 | 0.3243 | 0.3287 | 0.2380 |
| RF | 0.6927 | 0.7376 | 0.8036 | 0.7345 | 0.7668 | 0.3179 | 0.3224 | 0.1450 |
| LGBM | 0.6884 | 0.7124 | 0.7936 | 0.7347 | 0.7619 | 0.3112 | 0.3155 | 0.0500 |
| XGBM | 0.6552 | 0.6939 | 0.7580 | 0.7138 | 0.7343 | 0.2431 | 0.2459 | 0.9120 |
- **wine [#data : 6498]**

| Model | Accuracy | AUC | Recall | Prec. | F1 | Kappa | MCC | TT (Sec) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LGBM | 0.9956 | 0.9969 | 0.9988 | 0.9954 | 0.9971 | 0.9880 | 0.9881 | 0.0390 |
| XGBM | 0.9954 | 0.9977 | 0.9988 | 0.9951 | 0.9970 | 0.9874 | 0.9875 | 0.1010 |
| RF | 0.9943 | 0.9973 | 0.9974 | 0.9951 | 0.9962 | 0.9844 | 0.9845 | 0.1420 |
| GBM | 0.9936 | 0.9971 | 0.9977 | 0.9939 | 0.9958 | 0.9826 | 0.9827 | 0.1610 |
- **electrical_grid [#data : 10000]**

| Model | Accuracy | AUC | Recall | Prec. | F1 | Kappa | MCC | TT (Sec) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LGBM | 0.9427 | 0.9871 | 0.9662 | 0.9459 | 0.9559 | 0.8741 | 0.8745 | 0.0670 |
| XGBM | 0.9424 | 0.9883 | 0.9656 | 0.9461 | 0.9557 | 0.8735 | 0.8740 | 0.4520 |
| GBM | 0.9193 | 0.9767 | 0.9629 | 0.9160 | 0.9388 | 0.8204 | 0.8229 | 0.5330 |
| RF | 0.9171 | 0.9768 | 0.9536 | 0.9206 | 0.9367 | 0.8167 | 0.8181 | 0.2870 |
- **employee [#data : 14999]**

| Model | Accuracy | AUC | Recall | Prec. | F1 | Kappa | MCC | TT (Sec) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RF | 0.9888 | 0.9919 | 0.9612 | 0.9914 | 0.9760 | 0.9687 | 0.9689 | 0.1990 |
| XGBM | 0.9869 | 0.9921 | 0.9604 | 0.9840 | 0.9720 | 0.9634 | 0.9636 | 0.3800 |
| LGBM | 0.9859 | 0.9929 | 0.9536 | 0.9868 | 0.9698 | 0.9606 | 0.9609 | 0.0560 |
| GBM | 0.9765 | 0.9882 | 0.9292 | 0.9708 | 0.9495 | 0.9342 | 0.9346 | 0.2350 |

결과를 분석해보면 성능 평가 지표와 모델 훈련 시간 측면에서 어느 정도 일관성을 보인다고 할 수 있을 것 같다.

- 데이터셋별 성능 비교 : 가장 뚜렷하게 확인할 수 있는 것은 dataset이 일반적으로 적다고 말할 수 잇는 titanic과 nba데이터셋에서 GBM과 RF 모델이 다른 두 모델에 비해 성능이 높다는 점이다. XGBM과 LGBM은 데이터의 수가 일정 수준 이상일 때 성능이 좋다는 경험적 직관과 일치하는 결과이다. 또한 GBM은 overfitting이 되는 경향이 있어서 데이터셋이 적은 경우에 더 좋은 성능을 보이는 것으로 추론할 수 있을 것 같다.
- 모델 훈련 시간 비교 : 일관적인 결과는 거의 없지만 LGBM이 모든 경우에 훈련 시간이 압도적으로 짧다는 것을 확인할 수 있었다. 이는 수업시간에 배운 내용과 일치하는 결과이다. 다만 XGBM은 prallel한 연산이 가능하기 때문에 GBM보다 데이터수가 많은 경우에 훈련 시간이 짧을 것으로 예상했었는데, 꼭 그렇지는 않아서 의외인 결과였다. 데이터셋에 따라 달라지는 것 같다고 추론할 수 있을 것 같다.

### 2. 데이터셋별 하이퍼파라미터에 따른 모델 성능 비교

**hyperparameter 선정 이유**

hyperparameter로는 n_estimators와 max_depth를 선정했다. n_estimators는 tree 계열 모델에서는 tree의 수를 의미하고, boosting 계열 모델에서는 week learner를 의미한다. 의미가 조금은 다르지만 결국 base learner의 수라는 점에서는 같은 의미의 parameter라고 볼 수 있다. max_depth는 개별 tree의 최대 depth를 의미하고 none으로 설정되면 제한없이 split된다. n_estimator와 max_depth를 선정한 이유는 base learner를 쓰는 방법론에서 base learner의 수와 tree의 depth를 조절하는 hyperparameter이기 때문에 성능에 직접적인 연관성이 있을 것이라고 판단했을 뿐만 아니라 bagging과 boosting 방법론에서 공통적인 hyperparameter이기 때문이다. 따라서 다른 중요한 하이퍼파라미터들도 많지만 위의 두 hyperparameter를 선정하였다.

**hyperparameter setting**

- n_estimators : [50, 100, 200]
- max_depth : [None, 2, 4, 8, 16]

**실험 세팅**

hyperparamter tuning은 시간이 오래 걸리기 때문에 1번 실험과 달리 fold를 5로 설정하여 실험을 진행하였다.

**데이터셋별 hyperparameter에 따른 성능 비교**

- titanic
    
    
    | **RF** | max_depth : None | max_depth : 2 | max_depth : 4 | max_depth : 8 | max_depth : 16 |
    | --- | --- | --- | --- | --- | --- |
    | n_estimators : 50 | 0.7978 | 0.6196 | 0.6983 | 0.7464 | 0.7785 |
    | n_estimators : 100 | 0.8043 | 0.6132 | 0.6982 | 0.7432 | 0.7705 |
    | n_estimators : 200 | 0.8026 | 0.6131 | 0.6950 | 0.7544 | 0.7737 |
    
    | **GBM** | max_depth : None | max_depth : 2 | max_depth : 4 | max_depth : 8 | max_depth : 16 |
    | --- | --- | --- | --- | --- | --- |
    | n_estimators : 50 | 0.8009 | 0.7962 | 0.8315 | 0.8315 | 0.8251 |
    | n_estimators : 100 | 0.8042 | 0.7978 | 0.8347 | 0.8363 | 0.8283 |
    | n_estimators : 200 | 0.8009 | 0.8059 | 0.8347 | 0.8396 | 0.8331 |
    
    | **XGBM** | max_depth : None | max_depth : 2 | max_depth : 4 | max_depth : 8 | max_depth : 16 |
    | --- | --- | --- | --- | --- | --- |
    | n_estimators : 50 | 0.7801 | 0.8155 | 0.8026 | 0.7833 | 0.7801 |
    | n_estimators : 100 | 0.7721 | 0.8074 | 0.7865 | 0.7672 | 0.7753 |
    | n_estimators : 200 | 0.7640 | 0.7994 | 0.7672 | 0.7592 | 0.7656 |
    
    | **LGBM** | max_depth : None | max_depth : 2 | max_depth : 4 | max_depth : 8 | max_depth : 16 |
    | --- | --- | --- | --- | --- | --- |
    | n_estimators : 50 | 0.8106 | 0.8171 | 0.8122 | 0.8058 | 0.8106 |
    | n_estimators : 100 | 0.7881 | 0.8218 | 0.8058 | 0.7993 | 0.7897 |
    | n_estimators : 200 | 0.7752 | 0.8138 | 0.7961 | 0.7864 | 0.7736 |
    
- nba
    
    
    | **RF** | max_depth : None | max_depth : 2 | max_depth : 4 | max_depth : 8 | max_depth : 16 |
    | --- | --- | --- | --- | --- | --- |
    | n_estimators : 50 | 0.6969 | 0.6307 | 0.7012 | 0.7022 | 0.6958 |
    | n_estimators : 100 | 0.7065 | 0.6307 | 0.6884 | 0.7054 | 0.6958 |
    | n_estimators : 200 | 0.6937 | 0.6307 | 0.6841 | 0.7065 | 0.6991 |
    
    | **GBM** | max_depth : None | max_depth : 2 | max_depth : 4 | max_depth : 8 | max_depth : 16 |
    | --- | --- | --- | --- | --- | --- |
    | n_estimators : 50 | 0.6489 | 0.6969 | 0.6948 | 0.6841 | 0.6660 |
    | n_estimators : 100 | 0.6500 | 0.6991 | 0.6937 | 0.6820 | 0.6670 |
    | n_estimators : 200 | 0.6564 | 0.6937 | 0.6873 | 0.6884 | 0.6724 |
    
    | **XGBM** | max_depth : None | max_depth : 2 | max_depth : 4 | max_depth : 8 | max_depth : 16 |
    | --- | --- | --- | --- | --- | --- |
    | n_estimators : 50 | 0.6841 | 0.6649 | 0.6510 | 0.6702 | 0.6862 |
    | n_estimators : 100 | 0.6734 | 0.6670 | 0.6478 | 0.6819 | 0.6873 |
    | n_estimators : 200 | 0.6755 | 0.6521 | 0.6617 | 0.6809 | 0.6883 |
    
    | **LGBM** | max_depth : None | max_depth : 2 | max_depth : 4 | max_depth : 8 | max_depth : 16 |
    | --- | --- | --- | --- | --- | --- |
    | n_estimators : 50 | 0.6852 | 0.7002 | 0.6842 | 0.6831 | 0.6809 |
    | n_estimators : 100 | 0.6862 | 0.6830 | 0.6703 | 0.6724 | 0.6820 |
    | n_estimators : 200 | 0.6670 | 0.6767 | 0.6628 | 0.6766 | 0.6734 |

- wine
    
    
    | **RF** | max_depth : None | max_depth : 2 | max_depth : 4 | max_depth : 8 | max_depth : 16 |
    | --- | --- | --- | --- | --- | --- |
    | n_estimators : 50 | 0.9949 | 0.9743 | 0.9903 | 0.9947 | 0.9949 |
    | n_estimators : 100 | 0.9952 | 0.9719 | 0.9897 | 0.9947 | 0.9952 |
    | n_estimators : 200 | 0.9952 | 0.9710 | 0.9905 | 0.9947 | 0.9952 |
    
    | **GBM** | max_depth : None | max_depth : 2 | max_depth : 4 | max_depth : 8 | max_depth : 16 |
    | --- | --- | --- | --- | --- | --- |
    | n_estimators : 50 | 0.9848 | 0.9877 | 0.9919 | 0.9872 | 0.9855 |
    | n_estimators : 100 | 0.9853 | 0.9923 | 0.9932 | 0.9890 | 0.9855 |
    | n_estimators : 200 | 0.9861 | 0.9943 | 0.9945 | 0.9892 | 0.9855 |
    
    | **XGBM** | max_depth : None | max_depth : 2 | max_depth : 4 | max_depth : 8 | max_depth : 16 |
    | --- | --- | --- | --- | --- | --- |
    | n_estimators : 50 | 0.9956 | 0.9945 | 0.9956 | 0.9954 | 0.9956 |
    | n_estimators : 100 | 0.9960 | 0.9965 | 0.9960 | 0.9956 | 0.9958 |
    | n_estimators : 200 | 0.9965 | 0.9958 | 0.9960 | 0.9956 | 0.9960 |
    
    | **LGBM** | max_depth : None | max_depth : 2 | max_depth : 4 | max_depth : 8 | max_depth : 16 |
    | --- | --- | --- | --- | --- | --- |
    | n_estimators : 50 | 0.9943 | 0.9908 | 0.9936 | 0.9941 | 0.9943 |
    | n_estimators : 100 | 0.9956 | 0.9936 | 0.9947 | 0.9949 | 0.9956 |
    | n_estimators : 200 | 0.9960 | 0.9956 | 0.9963 | 0.9956 | 0.9960 |

- electrical_grid
    
    
    | **RF** | max_depth : None | max_depth : 2 | max_depth : 4 | max_depth : 8 | max_depth : 16 |
    | --- | --- | --- | --- | --- | --- |
    | n_estimators : 50 | 0.9118 | 0.7080 | 0.8203 | 0.8961 | 0.9101 |
    | n_estimators : 100 | 0.9171 | 0.7084 | 0.8205 | 0.8956 | 0.9150 |
    | n_estimators : 200 | 0.9180 | 0.7088 | 0.8165 | 0.8998 | 0.9156 |
    
    | **GBM** | max_depth : None | max_depth : 2 | max_depth : 4 | max_depth : 8 | max_depth : 16 |
    | --- | --- | --- | --- | --- | --- |
    | n_estimators : 50 | 0.8415 | 0.8713 | 0.9084 | 0.9150 | 0.8405 |
    | n_estimators : 100 | 0.8407 | 0.9011 | 0.9244 | 0.9201 | 0.8388 |
    | n_estimators : 200 | 0.8395 | 0.9190 | 0.9357 | 0.9247 | 0.8395 |
    
    | **XGBM** | max_depth : None | max_depth : 2 | max_depth : 4 | max_depth : 8 | max_depth : 16 |
    | --- | --- | --- | --- | --- | --- |
    | n_estimators : 50 | 0.9348 | 0.9120 | 0.9313 | 0.9331 | 0.9337 |
    | n_estimators : 100 | 0.9407 | 0.9254 | 0.9381 | 0.9364 | 0.9383 |
    | n_estimators : 200 | 0.9450 | 0.9314 | 0.9431 | 0.9408 | 0.9410 |
    
    | **LGBM** | max_depth : None | max_depth : 2 | max_depth : 4 | max_depth : 8 | max_depth : 16 |
    | --- | --- | --- | --- | --- | --- |
    | n_estimators : 50 | 0.9223 | 0.8721 | 0.9097 | 0.9247 | 0.9223 |
    | n_estimators : 100 | 0.9381 | 0.9038 | 0.9274 | 0.9386 | 0.9381 |
    | n_estimators : 200 | 0.9423 | 0.9208 | 0.9367 | 0.9436 | 0.9423 |
    
- employee
    
    
    | **RF** | max_depth : None | max_depth : 2 | max_depth : 4 | max_depth : 8 | max_depth : 16 |
    | --- | --- | --- | --- | --- | --- |
    | n_estimators : 50 | 0.9877 | 0.7938 | 0.9396 | 0.9684 | 0.9850 |
    | n_estimators : 100 | 0.9878 | 0.8693 | 0.9160 | 0.9711 | 0.9855 |
    | n_estimators : 200 | 0.9880 | 0.8684 | 0.9159 | 0.9720 | 0.9863 |
    
    | **GBM** | max_depth : None | max_depth : 2 | max_depth : 4 | max_depth : 8 | max_depth : 16 |
    | --- | --- | --- | --- | --- | --- |
    | n_estimators : 50 | 0.9766 | 0.9535 | 0.9777 | 0.9828 | 0.9772 |
    | n_estimators : 100 | 0.9759 | 0.9663 | 0.9796 | 0.9854 | 0.9771 |
    | n_estimators : 200 | 0.9756 | 0.9747 | 0.9818 | 0.9862 | 0.9761 |
    
    | **XGBM** | max_depth : None | max_depth : 2 | max_depth : 4 | max_depth : 8 | max_depth : 16 |
    | --- | --- | --- | --- | --- | --- |
    | n_estimators : 50 | 0.9348 | 0.9120 | 0.9313 | 0.9331 | 0.9337 |
    | n_estimators : 100 | 0.9407 | 0.9254 | 0.9381 | 0.9364 | 0.9383 |
    | n_estimators : 200 | 0.9450 | 0.9314 | 0.9431 | 0.9408 | 0.9410 |
    
    | **LGBM** | max_depth : None | max_depth : 2 | max_depth : 4 | max_depth : 8 | max_depth : 16 |
    | --- | --- | --- | --- | --- | --- |
    | n_estimators : 50 | 0.9806 | 0.9533 | 0.9768 | 0.9806 | 0.9807 |
    | n_estimators : 100 | 0.9853 | 0.9642 | 0.9784 | 0.9830 | 0.9857 |
    | n_estimators : 200 | 0.9874 | 0.9749 | 0.9793 | 0.9870 | 0.9877 |

분석 결과는 다음과 같다.

- n_estimators : n_estimators는 optimal한 값이 존재하는 것 같다는 결론을 내릴 수 있을 것 같다. 데이터셋이나 모델에 관계없이 n_estimators의 값이 커질수록 성능이 순차적으로 좋아지거나 나빠지거나 좋아졌다가 나빠지는 등 2차 함수 형태의 결과를 확인할 수 있었다. 또한 성능에도 어느정도 유의미한 영향을 미치는 것을 알 수 있었다.
- max_depth : max_depth는 데이터나 모델별로 어떤 패턴을 관찰할 수 없었다. 즉 이 하이퍼파라미터는 데이터셋과 모델에 따라 여러 시도를 통해 조절해야하는 값이라는 결론을 낼 수 있을 것 같다. 또한 성능에는 n_estimators에 비해 더 큰 영향을 주는 것을 확인할 수 있었는데, 이 hyperparameter는 overfitting과 관련이 있기 때문인 것으로 생각된다.

### 3. 데이터셋별 feature importance 비교

이번에는 모델별로 성능에 중요한 영향을 미친 feature들을 비교하기 위해 feature importance가 높은 상위 10개에 대한 plot을 비교해보았다.

- titanic

![image](https://user-images.githubusercontent.com/79893946/204270876-48844019-b17d-481a-abac-ed0a01d8cd30.png)


- nba

![image](https://user-images.githubusercontent.com/79893946/204271139-1ba2213f-c986-427a-bcff-d830ed66d232.png)

- wine

![image](https://user-images.githubusercontent.com/79893946/204271289-5725abe6-6015-44e7-be9b-49895b7cceab.png)


- electrical_grid

![image](https://user-images.githubusercontent.com/79893946/204271483-16cc6542-a752-4f54-bbfe-64fe71d1965a.png)


- employee

![image](https://user-images.githubusercontent.com/79893946/204271588-03cd7d00-5388-4604-b8b6-0f6a794a1159.png)


데이터 수에 관계없이 모델에 따라 Feature importance 값 상위 feature들은 순위가 조금은 다를지라도 어느정도 비슷한 결과를 보이는 것을 알 수 있었다. 대부분의 plot에서 feature importance가 급감하기 시작하는 feature들이 있는데 이는 tree 기반 모델이기 때문에 information gain이 뚝 떨어지는 split이 있기 때문인 것으로 생각되고 급감한 feature부터는 모델별로 일관성이 없었다. 즉 높은 information gain을 갖는 feature들은 어느 정도 일관성을 갖고 있는 것 같고, 이 feature들을 중심으로 feature engineering을 하면 모델 성능을 높이는 데에 도움이 될 것 같다는 생각이 들었다.

### 4. 데이터셋별 hyperparameter tuning 전후 결과 비교 [Accuracy]

- titanic

| Model | default | tuning |
| --- | --- | --- |
| RF | 0.7994 | 0.8043 |
| GBM | **0.8154** | **0.8396** |
| XGBM | 0.7752 | 0.8155 |
| LGBM | 0.7881 | 0.8218 |
- nba

| Model | default | tuning |
| --- | --- | --- |
| RF | 0.6927 | **0.7065** |
| GBM | **0.6970** | 0.6991 |
| XGBM | 0.6552 | 0.6883 |
| LGBM | 0.6884 | 0.7002 |
- wine

| Model | default | tuning |
| --- | --- | --- |
| RF | 0.9943 | 0.9952 |
| GBM | 0.9936 | 0.9945 |
| XGBM | 0.9954 | **0.9965** |
| LGBM | **0.9956** | 0.9963 |
- electrical_grid

| Model | default | tuning |
| --- | --- | --- |
| RF | 0.9171 | 0.9180 |
| GBM | 0.9193 | 0.9357 |
| XGBM | 0.9424 | **0.9450** |
| LGBM | **0.9427** | 0.9436 |
- employee

| Model | default | tuning |
| --- | --- | --- |
| RF | **0.9888** | **0.9880** |
| GBM | 0.9765 | 0.9862 |
| XGBM | 0.9869 | 0.9872 |
| LGBM | 0.9859 | 0.9877 |

hyperparameter tuning 전의 성능과 tuning 후의 성능은 거의 모든 경우에 hyperparameter tuning 후의 성능이 좋아졌다. 이는 당연한 결과라고도 볼 수는 있지만 동시에 n_estimators와 max_depth가 bagging과 boosting모델 모두에서 중요하고 의미있는 hyperparamter라는 것을 의미한다고 할 수 있다. 또한 성능 개선 여지가 많은 데이터셋의 경우에서는 tuning의 효과가 더욱 큰 것을 알 수 있었다. 

### 5. 결론

reference

[https://www.kaggle.com/getting-started/176257](https://www.kaggle.com/getting-started/176257)

[https://www.geeksforgeeks.org/ml-gradient-boosting/](https://www.geeksforgeeks.org/ml-gradient-boosting/)

[https://www.researchgate.net/figure/Sketch-of-a-gradient-boosting-machine_fig2_349921851](https://www.researchgate.net/figure/Sketch-of-a-gradient-boosting-machine_fig2_349921851)

[https://nurilee.com/2020/04/03/lightgbm-definition-parameter-tuning/](https://nurilee.com/2020/04/03/lightgbm-definition-parameter-tuning/)
