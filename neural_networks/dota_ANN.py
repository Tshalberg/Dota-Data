import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from mpl_toolkits.mplot3d import Axes3D

# We define our neural network class
class Neural_network(object):
    def __init__(self, inSize, hidSize):
        # Define hyperparameters
        self.inputLayerSize = inSize
        self.outputLayerSize = 1
        self.hiddenLayerSize = hidSize
        
        # Weights
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)
        
    def forward(self, X):
        # Propergates inputs through network
        self.z2  = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3)
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
    
        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)
        
        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)  
        
        return dJdW1, dJdW2

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



import pandas as pd
from sklearn import preprocessing
df = pd.read_excel("data/tommyStats.xlsx")

zeus = df[df['heroes']=='Zeus']

ind_train = np.random.choice(zeus.index, int(0.6*len(zeus)))
zeus_train = zeus.loc[ind_train]
zeus_test = zeus.drop(ind_train)

def dataPrep(df, cols):
    df.reset_index(drop=True, inplace=True)
#    df['faction'] = np.array(df['faction'] == 'radiant').astype(int)
    
    df_in = df[cols]
    df_out = df[['victory']]
    
    df_out = df_out.astype(int)
    
    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(df_in)
    df_in_norm = pd.DataFrame(np_scaled)
    
    
    x = np.array(df_in_norm)
    y = np.array(df_out)
    
    return x, y

cols = ['gold_per_min', 'xp_per_min', "kills", "assists", "deaths", 'duration']
inSize = len(cols)
x_train, y_train = dataPrep(zeus_train, cols)
x_test, y_test = dataPrep(zeus_test, cols)
#cost = NN.costFunction(X, y)
#dJW1, dJW2 = NN.costFunctionPrime(X, y)
#yHat = NN.forward(X)
# X = (hours sleeping, hours studying), y = Score on test
#X = np.array(([3,5], [5,1], [10,2], [6,1.5]), dtype=float)
#y = np.array(([75], [82], [93], [70]), dtype=float)
#
## Scaling values from 0 to 1
#X = X/np.amax(X, axis=0)
#y = y/100
#np.random.seed(1)
costs_train = []
costs_test = []
a = 0.005
Nit = 10000
hidSize = 20
NN = Neural_network(inSize, hidSize)
for i in range(Nit):
    costs_train.append(NN.costFunction(x_train, y_train))
    costs_test.append(NN.costFunction(x_test, y_test))
    dJW1, dJW2 = NN.costFunctionPrime(x_train, y_train)
    NN.W1 = NN.W1 - dJW1*a
    NN.W2 = NN.W2 - dJW2*a
#yHat = NN.forward(X)
#costs.append(NN.costFunction(X, y))
print(NN.costFunction(x_test, y_test))
plt.plot(list(range(Nit)), costs_train)
plt.plot(list(range(Nit)), costs_test)