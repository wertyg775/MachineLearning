import numpy as np

class logisticreg_regularise:

    def __init__(self, lr = 0.001, num_iterations = 1000, reg = None, reg_strength = 0.0 , fit_intercept = True, verbose = False):
        self.lr = lr
        self.num_iterations = num_iterations
        self.reg = reg
        self.reg_strength = reg_strength
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.w = None


    def _sigmoid(self, z):
        return 1/(1 + np.exp(-z))
    

    def _add_intercept(self, X):
        if not self.fit_intercept:
            return X
        
        ones = np.ones((X.shape[0], 1))
        return np.hstack((ones, X))

    
    def fit(self, X, y):
        if self.fit_intercept:
            X = self._add_intercept

        m, n = X.shape #capture dimension of input data
        self.w = np.zeros(n) #initialises the weights to matrix of zeroes of n size to represent number of features


        for i in range(self.num_iterations):
            z = np.dot(X, self.w) #compute dot product of input against weight, equivalent of matrix multiplication
            y_pred = self._sigmoid(z) #get the prediction values of every row with respect to current iterations weights

            #gradient when regularisation strength is 0
            gradient = (1/m) * np.dot(X.T, (y_pred - y)) #compute gradient of current iteration

            # add regularisation, what it effectively does is limits the weights to prevent overfitting 
            if self.reg == 'l2' and self.reg_strength > 0:
                gradient[1:] += (self.reg_strenth / m) * self.w[1:] #regularisation is only applied to weights , and not bias, hence array of gradient applied is [1:] since [0] is the bias term
            elif self.reg == 'l1' and self.reg_strength > 0:
                gradient[1:] += (self.reg_strength / m) * np.sign(self.w[1:]) #same case as l2 regularisation

            
            self.w += -self.lr * gradient # update the gradients at every loss


            if self.verbose and i % (self.num_iter // 5) == 0:
                loss = (-1/m) * np.sum(
                    y * np.log(y_pred + 1e-9) + (1 - y) * np.log(1 - y_pred + 1e-9)
                )
                if self.reg == 'l2':
                    loss += (self.reg_strength / (2*m)) * np.sum(self.w[1:] ** 2)
                elif self.reg == 'l1':
                    loss += (self.reg_strength / m) * np.sum(np.abs(self.w[1:]))
                print(f"Iter {i}: loss = {loss:.6f}")





