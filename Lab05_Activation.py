import cv2
import matplotlib.pyplot as plt
import numpy as np
z=np.arange(-4,4,0.05)
print("z =" , z, len(z))

###########################################################

def sigmoid(z):
    f=1.0/(1.0+np.exp(-z))
    d=f*(1-f)
    return f,d
output,derivative=sigmoid(z)
plt.plot(z,output,label='Sigmoid Function',color='blue')
plt.plot(z,derivative,label='Derivative',color='red')
plt.xlabel('z')
plt.ylabel('f(z)')
plt.title('Sigmoid Function')
plt.grid()
plt.legend()
plt.show()



def tanh(z):
    f=np.exp(z)-np.exp(-z)/(np.exp(z)+np.exp(-z))
    d=f*(1-f**2)
    return f,d
output,derivative=tanh(z)
plt.plot(z,output,label='Tanh Function',color='blue')
plt.plot(z,derivative,label='Derivative',color='red')
plt.xlabel('z')
plt.ylabel('f(z)')
plt.title('Tanh Function')
plt.grid()
plt.legend()
plt.show()


def relu(z):
    output=[]
    derivative=[]
    for i in z:
        if i<=0:
            output.append(i)
            derivative.append(1)
        else:
            output.append(0)
            derivative.append(0) 
    return output,derivative

output,derivative=relu(z) 
plt.plot(z,output,label='ReLU Function',color='blue')
plt.plot(z,derivative,label='Derivative',color='red')
plt.xlabel('z')
plt.ylabel('f(z)')
plt.title('ReLU Function')
plt.grid()
plt.legend()
plt.show()


##########################################################
z=np.array([3.2,1.3,0.2,0.8])

def softmax(z):
    exponents=[np.exp(i) for i in z]
    s=sum(exponents)
    prob=[i/s for i in exponents]
    return prob

print("Softmax Output = ",softmax(z))
