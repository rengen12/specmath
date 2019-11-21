import numpy as np
import numpy.linalg as LA
from inverse.inverse import inverse
from cholesky.cholesky import Cholesky
from qrdecomp.qr import QR
from utils.utils import diag, create_diag
from gelim.gaussian_elim import GaussElim
from ludecomp.lu import LU
import scipy.linalg as LA2

# 10) Розкладання Холецкого блочно-рекурсивне (Choletsky-recursive-decomposition)
def test_cholesky():
    a = np.array([
            [4, 12, -16],
            [12, 37, -43],
            [-16, -43, 98]
        ])
    r1 = np.linalg.cholesky(a)
    r2 = Cholesky(a).decompose()
    print('test_cholesky')
    print('lib', r1)
    print('own', r2)


test_cholesky()


# 3) Звернення матриці алгоритмом прямого і зворотного ходу. (FB-matrix-inversion)
def test_inverse():
    T = np.array([
        [2, 1, 1, 0],
        [4, 3, 3, 1],
        [8, 7, 9, 5],
        [6, 7, 9, 8]
    ])

    actual = inverse(T)
    expected = LA.inv(T)
    print('test_inverse')
    print('lib', expected)
    print('own', actual)


test_inverse()


# 8) QR-факторизація послідовним алгоритмом (QR-sequential-factorization)
def test_gram_schmidt():
    T = np.random.randn(100, 60)

    actual = QR(T).gram_schmidt()
    Q, R = LA.qr(T)

    # enforce uniqueness for numpy version
    D = create_diag(np.sign(diag(R)))
    Q = np.dot(Q, D)
    R = np.dot(D, R)
    expected = (Q, R)
    print('test_gram_schmidt')
    print('lib', expected)
    print('own', actual)


test_gram_schmidt()


# 1) Рішення систем лінійних рівнянь алгоритмом прямого і зворотного ходу. (FB-algorithm-for-SLAE)
def test_linalg():
    A = np.array([[3, 1], [1, 2]])
    b = np.array([9, 8])

    expected = np.linalg.solve(A, b)
    actual = GaussElim(A, b).solve()[0]
    print('test_linalg')
    print('lib', expected)
    print('own', actual)


test_linalg()


# 5) LDU факторизация послідовним алгоритмом. (LDU-sequential-factorization)
def test_LU():
    T = np.random.randn(50, 50)

    actual = LU(T, pivoting='partial').decompose()
    expected = LA2.lu(T)
    print('test_LU')
    print('lib', expected)
    print('own', actual)


test_LU()