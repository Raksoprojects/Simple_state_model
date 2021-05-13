import numpy as np
from matplotlib import pyplot as plt


def sym(A,B,C,D):

    u = 1
    y_list = np.array([])
    x = np.array([0,0,0])
    x = np.reshape(x,(3,1))
    
    for i in range(30):
        
        y = np.dot(C, x) + np.dot(D, u)
        x = np.dot(A, x) + np.dot(B, u)
        y_list = np.append(y_list, y)   
    return y_list

def symLQR(A,B):

    c1 = 2
    c2 = 5
    Q = c1 * np.identity(3)
    R = c2
    P = np.zeros((3,3), dtype=float)
    for i in range(30):
        P = Q + np.dot(np.dot(A.T, (P - np.dot(np.dot(np.dot(P, np.dot(B, np.linalg.inv(R+np.dot(B.T,np.dot(P,B))))), B.T), P))),A)
    return np.dot(np.dot(np.dot(np.linalg.inv(R + np.dot(np.dot(B.T, P), B)), B.T), P), A)
    

# Model stanowy
a0 = 0.6
a1 = 3.4
a2 = 3.5

A = np.array([[-a2,a1,-a0],[1,0,0],[0,1,0]])
B = np.array([[1],[0],[0]])
C = np.array([1,1,1])
D = 0
C = np.reshape(C, (1,3))
B = np.reshape(B, (3,1))
y = sym(A,B,C,D)
# Układ niestabilny :(

F = symLQR(A,B)
#print(F)````
Anew = A - np.dot(B, F)
#print(Anew)
ynew = sym(Anew,B,C,D)
x = np.array([0,0,0])
x = np.reshape(x,(3,1))
uk_list = np.array([])
uk = -np.dot(F,x)
for i in range(30):
    x = np.dot(Anew, x) + np.dot(B, uk)
    uk = -np.dot(F,x)
    uk_list = np.append(uk_list,uk)
plt.ylim(bottom=-10,top=10)
plt.plot(y)
plt.show()
plt.title("Odpowiedź skokowa i sygnał sterujący dla układu ze sterownikiem\n c1 = 2, c2 = 5")
plt.plot(uk_list)
plt.plot(ynew)
plt.show()



