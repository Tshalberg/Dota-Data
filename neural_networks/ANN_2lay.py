import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from mpl_toolkits.mplot3d import Axes3D

# We define our neural network class
class Neural_network(object):
    def __init__(self, hidSize):
        # Define hyperparameters
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = hidSize
        
        # Weights
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, self.hiddenLayerSize)
        self.W3 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)
        
    def forward(self, X):
        # Propergates inputs through network
        print(X.shape, self.W1.shape)
        self.z2  = np.dot(X, self.W1)
        print(self.z2.shape)
        self.a2 = self.sigmoid(self.z2)
        print(self.a2.shape, self.W2.shape)
        self.z3 = np.dot(self.a2, self.W2)
        print(self.z3.shape)
        self.a3 = self.sigmoid(self.z3)
        print(self.a3.shape, self.W3.shape)
        self.z4 = np.dot(self.a3, self.W3)
#        print(self.z)
        yHat = self.sigmoid(self.z4)
        return yHat
        
            
    def sigmoid(self, z):
        # Applies sigmoid function to scalar, vector or matrix
        return 1/(1+np.exp(-z))
    
    def sigmoidPrime(self, z):
        # The derivative of the sigmoid function
        return np.exp(-z)/(1+np.exp(-z))**2
    
    def costFunction(self, X, y):
        #Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        J = 0.5*sum((y-self.yHat)**2)
        return J    
    
    def costFunctionPrime(self, X, y):
        #Compute derivative with respect to W and W2 for a given X and y:
        self.yHat = self.forward(X)
        
        delta4 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z4))
        dJdW3 = np.dot(self.a3.T, delta4)
        
        print(self.z3.shape)
    
        delta3 = np.dot(delta4, self.W3.T)*self.sigmoidPrime(self.z3)
        dJdW2 = np.dot(self.a2.T, delta3)
        
        print(delta3.shape)
        
        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)  
        
        return dJdW1, dJdW2, dJdW3

    def getParams(self):
        # Get W1 and W2 "rolled" into a vector
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params
    
    def setParams(self, params):
        #Set W1 and W2 using single paramater vector.
        W1_start = 0
        W1_end = self.hiddenLayerSize * self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize , self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))
        
    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))

    
def computeNumericalGradient(N, X, y):
        paramsInitial = N.getParams()
        numgrad = np.zeros(paramsInitial.shape)
        perturb = np.zeros(paramsInitial.shape)
        e = 1e-4

        for p in range(len(paramsInitial)):
            #Set perturbation vector
            perturb[p] = e
            N.setParams(paramsInitial + perturb)
            loss2 = N.costFunction(X, y)
            
            N.setParams(paramsInitial - perturb)
            loss1 = N.costFunction(X, y)

            #Compute Numerical Gradient
            numgrad[p] = (loss2 - loss1) / (2*e)

            #Return the value we changed to zero:
            perturb[p] = 0
            
        #Return Params to original value:
        N.setParams(paramsInitial)

        return numgrad
    
class trainer(object):
    def __init__(self, N):
        #Make Local reference to network:
        self.N = N
        
    def callbackF(self, params):
        self.N.setParams(params)
        self.J.append(self.N.costFunction(self.X, self.y))   
        
    def costFunctionWrapper(self, params, X, y):
        self.N.setParams(params)
        cost = self.N.costFunction(X, y)
        grad = self.N.computeGradients(X,y)
        
        return cost, grad
        
    def train(self, X, y):
        #Make an internal variable for the callback function:
        self.X = X
        self.y = y

        #Make empty list to store costs:
        self.J = []
        
        params0 = self.N.getParams()

        options = {'maxiter': 200, 'disp' : True}
        _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS', \
                                 args=(X, y), options=options, callback=self.callbackF)

        self.N.setParams(_res.x)
        self.optimizationResults = _res 


# # Defining values
#X = np.array(([3,5], [5,1], [10,2]), dtype=float)
#y = np.array(([75], [82], [93]), dtype=float)

# X = (hours sleeping, hours studying), y = Score on test
X = np.array(([3,5], [5,1], [10,2], [6,1.5]), dtype=float)
y = np.array(([75], [82], [93], [70]), dtype=float)

# Scaling values from 0 to 1
X = X/np.amax(X, axis=0)
y = y/100

costs = []
a = 0.5
hidSize = 50
NN = Neural_network(hidSize)
#yHat = NN.forward(X)
#dJW1, dJW2, dJdW3 = NN.costFunctionPrime(X, y)
Nit = 10000
for i in range(Nit):
    costs.append(NN.costFunction(X, y))
    dJW1, dJW2, dJW3 = NN.costFunctionPrime(X, y)
    NN.W1 = NN.W1 - dJW1*a
    NN.W2 = NN.W2 - dJW2*a
    NN.W3 = NN.W3 - dJW3*a
print(NN.costFunction(X, y))
plt.plot(list(range(Nit)), costs)





#params = NN.getParams()
#W1_start = 0
#W1_end = NN.hiddenLayerSize * NN.inputLayerSize
#NN.W1 = np.reshape(params[W1_start:W1_end], (NN.inputLayerSize , NN.hiddenLayerSize))
#W2_end = W1_end + NN.hiddenLayerSize*NN.outputLayerSize
#NN.W2 = np.reshape(params[W1_end:W2_end], (NN.hiddenLayerSize, NN.outputLayerSize))
#W1 = NN.W1
#print(np.reshape(params[W1_start:W1_end], (NN.inputLayerSize , NN.hiddenLayerSize)))
#NN.setParams(params)
#yHat2 = NN.forward(X)
#dJW1_2, dJW2_2 = NN.costFunctionPrime(X, y)
#W1_2 = NN.W1
#W2_2 = NN.W2
#print(W1_2)
#T = trainer(NN)
#T.train(X, y)
#yHat = NN.forward(X)
#
#hoursSleep = np.linspace(0, 10, 100)
#hoursStudy = np.linspace(0, 5, 100)
#
##Normalize data (same way training data way normalized)
#hoursSleepNorm = hoursSleep/10.
#hoursStudyNorm = hoursStudy/5.
#
#
#
#
##Create 2-d versions of input for plotting
#a, b  = np.meshgrid(hoursSleepNorm, hoursStudyNorm)
#
##Join into a single input matrix:
#allInputs = np.zeros((a.size, 2))
#allInputs[:, 0] = a.ravel()
#allInputs[:, 1] = b.ravel()
#
#yy = np.dot(hoursStudy.reshape(100,1), np.ones((1,100)))
#xx = np.dot(hoursSleep.reshape(100,1), np.ones((1,100))).T
#
#allOutputs = NN.forward(allInputs)
#
#CS = plt.contour(xx,yy,100*allOutputs.reshape(100, 100))
#plt.clabel(CS, inline=1, fontsize=10)
#plt.xlabel('Hours Sleep')
#plt.ylabel('Hours Study')
#
##3D plot:
##Uncomment to plot out-of-notebook (you'll be able to rotate)
##%matplotlib qt
#
#from mpl_toolkits.mplot3d import Axes3D
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#
##Scatter training examples:
#ax.scatter(10*X[:,0], 5*X[:,1], 100*y, c='k', alpha = 1, s=30)
#
#
#surf = ax.plot_surface(xx, yy, 100*allOutputs.reshape(100, 100), \
#                       cmap=plt.cm.jet, alpha = 0.5)
#
#
#ax.set_xlabel('Hours Sleep')
#ax.set_ylabel('Hours Study')
#ax.set_zlabel('Test Score')

#CS = plt.contour(xx, yy, 100)

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#
#ax.scatter(X[:,0], X[:, 1], yHat)

