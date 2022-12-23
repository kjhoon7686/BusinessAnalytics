# Semi-Supervised-Learning

# 목차

## 개념

1. Pi-Model (Pi)
2. Mean Teacher (MT)
3. Virtual Adversarial Training (VAT)
4. Unsupervised Data Augmentation (UDA)

## 실험

1. 실험 설명
2. Label data 수에 따른 성능 비교
3. Unlabel data 비율에 따른 성능 비교
4. Label data 수와 Unlabel data 비율의 관계
5. 결론

## 개념

이론에 대한 자세한 내용은 고려대학교 산업공학과 강필성 교수님의 “Business Analytics”강의를 참고하시길 바랍니다.

### Semi-Supervised Learning이란?

Semi-Supervised Learning은 소량의 labeled data에는 supervised learning, 대량의 unlabeled data에는 unsupervised learning을 적용하여 전체적인 성능을 향상시키는 방법론이라고 할 수 있다. 이러한 Semi-Supervised Learning에는 다음과 같은 3가지 가정이 존재한다. 

![image](https://user-images.githubusercontent.com/79893946/209302599-41656cc6-6076-4b68-8ce4-e3f6ac0f5330.png)

![image](https://user-images.githubusercontent.com/79893946/209302568-63a2a787-c1ea-4c5f-bcec-86412c2ee748.png)

### Consistency Regularization

Consistency Regularization은 Unlabeled data에 realistic perturbation을 적용해도 예측 결과에는 일관성이 있을 것이다라는 가정을 motivation으로 한다. 즉 Unlabeled data와 Perturbated Unlabeled data간의 consistency를 유지하는 방향으로 학습시킴으로써 어려운 샘플에 대해서도 유연한 class 예측을 가능하게 만드는 방법론이라고 할 수 있다.

### 1. Pi-Model (Pi)

Pi-Model의 핵심 내용은 하나의 FFN 모델에 두 번의 Augmentation을 통해 두 종류의 input에 대한 output간의 consistency를 학습하는 것이다.

![image](https://user-images.githubusercontent.com/79893946/209302640-26c58df4-bf94-44c3-9a32-56e7034a4f46.png)

Pi-Model의 구성요소를 정리해보면 다음과 같다.

- Input : input은 gaussian noise와 random augmentation을 통해 두 종류로 생성된다.
- 모델 구조 : 같은 FFN 구조이지만 다른 dropout regularizaion을 사용해서 모델을 구성한다.
- Supervised Loss : Cross Entropy Loss를 사용한다.
- Unsupervised Loss : 두 모델의 output간 MSE Loss를 사용한다.

Pi-Model과 같은 논문에서 Pi-Model의 한계점을 보완하기 위해 제안한 방법론이 Temporal Ensemble이다. Pi-Model은 target값이 single evaluation에 의해 구성되기 때문에 noisy하다는 한계점을 가지고 있다. 이를 보완하기 위해 Temporal Ensemble에서는 이전 network들의 evaluation값들을 ensemble하여 prediction을 한다.

![image](https://user-images.githubusercontent.com/79893946/209302686-28d55982-227a-4ce7-aeff-77d02ebde07c.png)

Temporal Ensemble의 장점은 epoch마다 한 번의 evaluation만 하기 때문에 학습 속도가 빠르고 Noise를 줄일 수 있다는 점이고 단점은 ensemble value를 추가적으로 저장해야 하고 추가적인 EMA 하이퍼파라미터가 필요하다는 점이다. 

### 2. Mean Teacher (MT)

Mean Teacher는 Pi-Model은 epoch당 한 번의 update만 하기 때문에 모델의 정보 업데이트 속도가 느리고 하나의 모델이 teacher와 student의 역할을 동시에 하기 때문에 생성된 target의 quality가 낮다는 점을 한계점으로 지적한다.

Mean Teacher의 핵심 내용은 모델을 Teacher와 Student로 구분하고 배치 단위로 Student의 가중치를 EMA를 통해 Teacher의 가중치로 사용한다는 것으로 요약할 수 있다. 

![image](https://user-images.githubusercontent.com/79893946/209302732-f05dd2b1-bed4-4458-8a06-c5681e81216e.png)

위의 그림을 살펴보면 전반적이는 구조는 Pi-Model과 비슷하지만 모델이 student와 teacher 두 종류로 이루어져 있고, EMA를 모델 파라미터에 적용하는 것을 알 수 있다. 

![image](https://user-images.githubusercontent.com/79893946/209302766-430cf079-6a7c-4573-b440-6a480a4f1fcd.png)

Mean Teacher의 학습 과정 및 요소들을 정리해보면 다음과 같다.

- Student model에는 Labeled data가 input으로 들어가게 되고 이 모델의 output과 정답을 비교하여 지도학습을 통해 student model의 파라미터가 업데이트된다.
- 여기서 지도학습에서는 Cross Entropy Loss를 사용한다.
- Teacher model에는 Unlabeled data가 input으로 들어가게 되고 student model의 파라미터에 EMA를 적용하여 파라미터를 업데이트한다.

이러한 모델 구조를 통해 Teacher model과 Student model의 역할을 명확하게 구분하고, 파라미터를 매 training step마다 업데이트함으로써 생성된 target의 quality를 향상시키고 모델의 파라미터 업데이터 속도를 향상시킬 수 있다.

### 3. Virtual Adversarial Training (VAT)

VAT의 핵심 내용은 Random Noise가 아닌 모델이 취약한 방향의 Adversarial Noise를 적용해서 모델을 Robust하게 만든다는 내용이다.

![image](https://user-images.githubusercontent.com/79893946/209302824-914b4d3c-c584-48e7-9455-e269b327e6e8.png)

VAT는 Random Perturbation은 모델이 Adversarial Direction에서의 약간의 Perturbation에도 취약하다는 한계점을 Motivation으로 삼고 있다. 

VAT의 Main Idea는 Model을 가장 Anisotropic한 방향으로 선택적으로 Smoothing하여 각 Input Data Point를 중심으로 Isotropic하게 Smooth하도록 Output Distribution을 학습시킨다는 것이다. 여기서 Virtual이라는 이름이 붙은 이유는 Unlabeled data를 사용하기 때문이다.

![image](https://user-images.githubusercontent.com/79893946/209302864-de7513ad-854d-4753-9d84-50338223ff41.png)

위의 그림을 통해 학습 과정을 간략히 살펴보면 먼저 augmented data를 input으로 한 모델의 output과 adversarial example을 input으로 한 모델의 output간의 Consistency Loss를 KL Divergence를 통해 비지도학습이 이루어진다. 또한 이전 모델들과 마찬가지로 Cross Entropy Loss를 통해 지도학습이 이루어진다. 마지막으로 이 Loss들을 결합하여 최종적인 모델 훈련이 진행된다. 

### 4. Unsupervised Data Augmentation (UDA)

UDA의 핵심 내용은 Random Augmentation이 아닌 발전된 Augmentation을 사용해서 학습을 하자는 내용이라고 할 수 있다. 

UDA의 Motivation은 다음의 3가지 이유 때문에 Advanced augmentation이 random noise보다 consistency regularization 방법론들에서 효과적이라는 것에서 출발한다.

- 현실적인 augmentation을 만들 수 있음
- Data efficiency 향상
- 서로 다른 task들에 대해 missing inductive bias를 제공 할 수 있음

![image](https://user-images.githubusercontent.com/79893946/209302897-9f13eef6-bd88-4f4f-b8b1-7f143628aca4.png)

위의 그림을 통해 모델 구조를 살펴보면 먼저 Labeled data같은 경우에는 앞선 모델들과 마찬가지로 Cross Entropy Loss를 통해 지도 학습이 이루어지는 것을 확인할 수 있다. 다음은 파라미터가 고정된 모델의 원본 Unlabeled data에 대한 output과 advanced augmentation이 적용된 Unlabeled data에 대한 output간의 Consistency Loss를 통해 비지도학습이 이루어지는 것을 확인할 수 있다. 마찬가지로 전체적인 학습은 지도 학습과 비지도 학습의 Loss function이 결합되어 진행된다.

## 실험

실험은 ****[USB: A Unified Semi-supervised Learning Benchmark for Classification](https://arxiv.org/abs/2208.07204)****라는 논문에서 공개한 Framework에서 진행하였다.

### 1. 실험 설명

**데이터**

실험에 사용한 데이터는 CIFAR-10을 사용하였다. 이 데이터는 Computer Vision 분야에서 많이 사용되는 데이터로 10개의 class를 가지고 있고 각 class당 6000개의 image로 이루어진 balance data이다. 50000개의 training dataset과 10000개의 test dataset으로 구성되어 있다.

**모델**

실험에 사용한 모델은 Pi-Model, Mean Teacher, VAT, UDA를 사용하였고 Self-training 방법론과의 성능 차이를 비교하기 위해 Pseudo-Labeling 모델을 추가적으로 사용하였다.

**실험 세팅**

실험에 사용한 하이퍼파라미터 세팅은 다음과 같다.

- 공통 하이퍼파라미터 세팅
    - epoch : 5
    - num_train_iter : 10000
    - num_eval_iter : 1000
    - batch_size : 64
    - eval_batch_size : 256
    - crop ratio (augmentation) : 0.875
    - optimizer : SGD
    - lr : 0.03
    - momentum : 0.9
    - weight_decay : 0.0005
- Pi-Model
    - ulb_loss_ratio : 10
- Mean Teacher
    - ulb_loss_ratio : 50
- VAT
    - ema_m : 0.999
- UDA
    - ulb_loss_ratio : 1
    - ema_m : 0.999
- Pseudo-Labeling
    - ulb_loss_ratio : 1

**하이퍼파라미터 튜닝**

- num_labels : 훈련에 사용되는 label 데이터 수
- uratio : 각 batch의 label data 대비 unlabel data의 비율 (ex. batch_size가 64이고 uratio가 2라면 각 batch당 label data 64개, unlabel data 128개)

num_lables는 **4000, 1000, 500**으로 하이퍼파라미터 search를 하고 uratio는 **1, 5, 10**으로 하이퍼파라미터 search를 하여 총 45번 [모델 5개 * 3 * 3]의 실험을 진행하였다. num_labels를 하이퍼파라미터로 지정한 이유는 label data의 수가 많아질수록 성능이 좋을 것이라는 가설을 세웠는데 이를 실험적으로 증명하기 위해 지정하였다. uratio를 하이퍼파라미터로 지정한 이유는 모델이 unlabel data에 대해 정확하게 label을 생성한다면 uratio가 높을수록 성능이 좋아지고, 그렇지 않다면 uratio가 낮을수록 성능이 좋아질 것이라는 가설을 세우고 이를 증명하기 위해 지정하였다.  

성능은 accuracy로 측정하였다.

### 2. Label data 수[num_labels]에 따른 성능 비교

uratio를 1로 고정한 상태에서 모델별 num_labels에 따른 성능을 확인해 보았다.

| num_labels | 500 | 1000 | 4000 |
| --- | --- | --- | --- |
| Pi-Model | 0.5289 | 0.6509 | 0.8159 |
| Mean Teacher | 0.4843 | 0.6036 | 0.8055 |
| VAT | 0.4593 | 0.6733 | 0.8352 |
| UDA | 0.7175 | 0.7742 | 0.853 |
| Pseudo-Labeling | 0.4938 | 0.6203 | 0.8088 |

가설대로 모든 모델에서 label data의 수를 늘릴수록 성능이 상승한다는 것을 확인할 수 있었다. 이는 label data 확보의 중요성을 다시 한 번 강조하는 결과인 것 같다. 모델별로 결과를 살펴보면 UDA가 성능이 가장 뛰어났고 label 수의 증가에 따른 성능 향상 폭은 VAT가 가장 컸다는 것을 알 수 있다. 

### 3. Unlabel data 비율[uratio]에 따른 성능 비교

num_labels를 4000으로 고정한 상태에서 모델별 uratio에 따른 성능을 확인해 보았다.

| uratio | 1 | 5 | 10 |
| --- | --- | --- | --- |
| Pi-Model | 0.8159 | 0.8283 | 0.8278 |
| Mean Teacher | 0.8055 | 0.8014 | 0.8042 |
| VAT | 0.8352 | 0.842 | 0.8447 |
| UDA | 0.853 | 0.8778 | 0.8901 |
| Pseudo-Labeling | 0.8088 | 0.8094 | 0.8089 |

VAT, UDA, Pi-Model은 uratio가 높아질수록 성능이 높아지는 경향성이 있고 Mean Teacher, Pseudo-Labeling은 뚜렷한 경향성은 보이지 않았다. 이 중에서도 UDA는 uratio가 높아질수록 성능 향상 폭이 상당히 두드러지는 것을 확인할 수 있었다. 이는 UDA가 unlabel data를 가장 잘 활용하는 모델이기 때문인 것으로 생각할 수 있을 것 같다. 또한 Pi-Model은 uratio가 1에서 5로 올랐을 때 성능 향상 폭이 꽤 높았지만 10일 때는 별다른 변화가 없는 것으로 보아 unlabel data의 비율이 모델에 유효한 최대값이 존재하는 것으로 생각된다.

요약하면 전반적으로 uratio가 높아지면 성능이 향상하는 경향성을 보이기는 하지만 UDA와 Pi-Model을 제외하면 성능에 크게 영향을 주지는 않는 것 같다는 결론을 내릴 수 있을 것 같다.

### 4. Label data 수[num_labels]와 Unlabel data 비율[uratio]의 관계

이번에는 모든 num_labels와 uratio의 조합에 따른 성능을 비교해 보았다.

- num_labels : 500

| uratio | 1 | 5 | 10 |
| --- | --- | --- | --- |
| Pi-Model | 0.5289 | 0.5291 | 0.5209 |
| Mean Teacher | 0.4843 | 0.4707 | 0.4773 |
| VAT | 0.4593 | 0.5679 | 0.4584 |
| UDA | 0.7175 | 0.8027 | 0.8431 |
| Pseudo-Labeling | 0.4938 | 0.5 | 0.5002 |

- num_labels : 1000

| uratio | 1 | 5 | 10 |
| --- | --- | --- | --- |
| Pi-Model | 0.6509 | 0.6365 | 0.6398 |
| Mean Teacher | 0.6036 | 0.6095 | 0.605 |
| VAT | 0.6733 | 0.6694 | 0.6937 |
| UDA | 0.7742 | 0.8379 | 0.8679 |
| Pseudo-Labeling | 0.6203 | 0.6163 | 0.6147 |

- num_labels : 4000

| uratio | 1 | 5 | 10 |
| --- | --- | --- | --- |
| Pi-Model | 0.8159 | 0.8283 | 0.8278 |
| Mean Teacher | 0.8055 | 0.8014 | 0.8042 |
| VAT | 0.8352 | 0.842 | 0.8447 |
| UDA | 0.853 | 0.8778 | 0.8901 |
| Pseudo-Labeling | 0.8088 | 0.8094 | 0.8089 |

Label data의 수가 늘어날수록 모델 성능은 향상되지만 [3]번의 결과처럼 uratio와는 모델 성능과의 뚜렷한 연관관계가 보이지 않았다. 또한 label 수와 uratio간의 직접적인 관련성도 찾을 수 없었다. 모델별로 살펴보면 Pi-Model, Mean Teacher, Pseudo-Labeling의 경우에는 [3]번의 결과와 같이 뚜렷한 성능 증감이 없었다는 것을 알 수 있다. VAT같은 경우에는 label 수가 적을수록 uratio에 따라 성능이 크게 변했는데 이는 label 수가 적을수록 unlabel data의 학습결과 더 영향을 미친다는 것을 의미한다고 해석할 수 있을 것 같다. 마지막으로 UDA는 label 수와 관계없이 uratio가 증가할수록 성능 향상 폭이 컸는데, 이는 UDA가 가장 unlabel data를 잘 활용하는 모델인 것 같다고 해석할 수 있을 것 같다.

### 5. 결론

Pi-Model, Mean Teacher, VAT, UDA 그리고 Pseudo-Labeling의 5가지 모델에 대해 label의 수 unlabel data의 비율이라는 하이퍼파라미터에 따른 성능 변화를 실험해보았다. 전반적으로 Pseudo-Labeling보다 Consistency Regularization 모델들의 성능이 모든 하이퍼파라미터 세팅에서 더 좋다는 것을 알 수 있었다. 또한 Label data의 수가 증가할수록 모델의 성능이 향상된다는 것을 확인할 수 있었고, unlabel data의 비율에 영향을 받는 모델과 그렇지 않은 모델이 있다는 것을 확인할 수 있었다. 마지막으로 label data의 수와 unlabel data의 비율은 크게 연관성이 없었고 label data가 작을 때 unlabel data의 수에 민감한 모델이 있다는 것을 알 수 있었다. 결론적으로 label data를 많이 확보하는 것이 가장 중요하고 uratio는 주요한 하이퍼파라미터라는 것을 알 수 있었다.

reference

- [https://www.researchgate.net/figure/Adversarial-examples-using-PGD-with-and-with-noise-constraint-of-on_fig1_350132115](https://www.researchgate.net/figure/Adversarial-examples-using-PGD-with-and-with-noise-constraint-of-on_fig1_350132115)
- [https://sanghyu.tistory.com/177](https://sanghyu.tistory.com/177)
- https://github.com/microsoft/Semi-supervised-learning
