import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

debug = True
testImage = True

#Rango de valores para los elementos de la base [-d, d]
d = 1000
#Dimensión del retículo generado
n = 10

def fill_array(arr, size):
    """
    Completa el array con 0's para que la información
    quede en el centro
    """
    current_size = arr.shape[0]
    if current_size < size:
        num_zeros = size - current_size -1
        zeros = np.zeros(num_zeros, dtype=arr.dtype)
        arr = np.concatenate((zeros, arr))
        arr = np.concatenate((arr, np.array([0])))
    return arr

def Hadamard(B):
    """
    Calcula el ratio de Hadamard para la matriz B
    """
    d = np.linalg.det(B)
    if d == 0: return 0
    d = np.absolute(d)
    prod = 1
    for row in B:
        prod *= np.linalg.norm(row)
    return np.power(d / prod, 1/n)

def rand_unimod(n):
    """
    Devuelve una matriz unimodular de
    tamaño nxn
    """
    A = np.eye(n, dtype=int)

    for i in range(1, n):
        for j in range(i):
            random_integer = np.random.randint(-10, 10)
            A[i] -= random_integer * A[j]
        for j in range(i+1, n):
            A[i] -= np.random.randint(-10, 10) * A[j]

    for i in range(n):
        if A[i, i] < 0:
            A[i] *= -1

    return A

def new_lattice(n):
    """
    Genera matrizes aleatorias nxn
    hasta que encontremos alguna con un
    buen ratio de Hadamard y la devolvemos
    """
    B = np.random.randint(-d, d, size=(n, n))
    print(Hadamard(B))
    while Hadamard(B) < 0.8:
        print(Hadamard(B))
        B = np.random.randint(-d, d, size=(n, n))
    B = B.transpose()
    return B

def babai(B, invB, t):
    """
    Calcula el algoritmo de babai,
    devuelve el vector más cercano a t
    en el retículo dado por las columnas
    de B
    """
    s = invB.dot(t)
    s = np.round(s)
    return B.dot(s)

def encrypt(m, BB):
    """
    Encripta el array binario m según GGH
    (x_i = B_hat m_i + e)
    """
    x = BB.dot(m) + (np.random.randint(100, size=n) - 50)
    return x

def decrypt(x, B, U):
    """
    Decripta x según GGH
    """
    invU = np.linalg.inv(U)
    invB = np.linalg.inv(B)


    if debug:
        # x_i -> x_i' := B_hat m_i
        babizado = babai(B, invB, x)

        #Comparamos el mismo proceso con la base de la clave pública
        BB = np.matmul(B, U)
        babaizadomalabase = babai(BB, np.matmul(invU, invB), x)

        # Mensaje original
        print("m = ", m)
        # Mensaje encriptado
        print("enc m = ", x)
        # m_i = B^-1_hat x_i' = U^-1 B^-1 x_i'
        print("dec m = ", np.around(invU.dot(invB.dot(babizado))).astype(int))
        # Desencriptado con mala base
        print("dec m (bad base) = ", np.around(invU.dot(invB.dot(babaizadomalabase))).astype(int))

    return np.around(invU.dot(invB.dot(babizado))).astype(int)


B = new_lattice(n)
U = rand_unimod(n)

BB = np.matmul(B, U)


if testImage:
    img = cv2.imread("test")

    h = img.shape[0]
    w = img.shape[1]

    for y in range(h):
        for x in range(w):
            for k in range(3):
                m = [int(x) for x in list('{0:0b}'.format(img[y, x, k]))]
                c = encrypt(fill_array(np.array(m), n), BB)
                newcol = ((c%2).astype(int))
                img[y, x, k] = (int("".join(str(i) for i in newcol[:-1]), 2) // 2)%255
            

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    print(img[:, :, 0])
    plt.show()
    cv2.imwrite('output.jpeg', img) 


m = np.random.randint(2, size=n)

x = encrypt(m, BB)

decrypt(x, B, U)