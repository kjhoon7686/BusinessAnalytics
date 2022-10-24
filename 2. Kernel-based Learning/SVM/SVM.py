# reference : https://github.com/nihil21/custom-svm/blob/master/src/svm.py
import itertools
import warnings

import cvxopt
import numpy as np
from matplotlib import pyplot as plt


class SVM:
    """Class implementing a Support Vector Machine: 
    Lagrangian Problem:
        L_P(w, b, lambda_mat) = 1/2 ||w||^2 - sum_i{lambda_i[(w * x + b) - 1]},
    에서
    dual Problem:
        L_D(lambda_mat) = sum_i{lambda_i} - 1/2 sum_i{sum_j{lambda_i lambda_j y_i y_j K(x_i, x_j)}}
    으로 변환
    Parameters
    ----------
    kernel: str, default="linear"
        커널 함수의 종류 ("linear", "rbf", "poly" or "sigmoid").
    gamma: float | None, default=None
        kernel functions의 parameter
    deg: int, default=3
        "poly" kernel function의 차수
    r: float, default=0.
        "poly" and "sigmoid" kernel functions의 parameter
    c: float | None, default=1.
        penelty term
    """

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

    def fit(self, x: np.ndarray, y: np.ndarray, verbosity: int = 1) -> None:
        """Fit the SVM on the given training set.
        Parameters
        ----------
        x: np.ndarray
            Training data with shape (n_samples, n_features).
        y: np.ndarray
            Ground-truth labels.
        verbosity: int, default=1
            Verbosity level in range [0, 3].
        """
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

    def project(
            self,
            x: np.ndarray,
            i: int | None = None,
            j: int | None = None
    ) -> np.ndarray:
        """hyperplane에 data projection
        Parameters
        ----------
        x: np.ndarray
            Data points with shape (n_samples, n_features).
        i: int | None, default=None
            First dimension to plot (in the case of non-linear kernels).
        j: int | None, default=None
            Second dimension to plot (in the case of non-linear kernels).
        Returns
        -------
        proj: np.ndarray
            hyperplane에 projection된 point들
        """
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

    def predict(self, x: np.ndarray) -> np.ndarray:
        """주어진 data point들의 class 예측.
        Parameters
        ----------
        x: np.ndarray
            Data points with shape (n_samples, n_features).
        Returns
        -------
        label: np.ndarray
            Predicted labels.
        """
        # point들의 label을 예측하기 위해서는 f(x)의 부호만 고려하면 됨
        return np.sign(self.project(x))

    @property
    def is_fit(self) -> bool:
        return self._is_fit

    @property
    def sv_x(self) -> np.ndarray:
        return self._sv_x

    @property
    def sv_y(self) -> np.ndarray:
        return self._sv_y

