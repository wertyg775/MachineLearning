import numpy as np

class LogisticRegressionScratch:


    def __init__(self, lr = 0.02, num_iterations = 1000, fit_intercept = True ):

        self.lr = lr
        self.num_iterations =num_iterations
        self.fit_intercept = fit_intercept
        self.w = None
        self.b = None

    def _add_intercept(self, X):
        if not self.fit_intercept:
            return X
        
        #similar to linear regression, fit column of 1s to the input features to represent the intercept during mat mul
        ones = np.ones(X.shape[0],1)
        X_aug = np.hstack([ones, X])


    #sigmoid function is the formula for logistic regresssion similar to how Y = Xw + b is for linear regresssion
    #takes the linear regression formula and use it as the variable for the sigmoid activation function
    def _sigmoid(self, z):
        return 1/ (1+np.exp(-z))
    
    #trained using gradient descent algorithm, logistic regression does not have a closed-form solution for its formula unlike linear regression with least squares
    #Step 1. initialise weights to zero for first iteration using np.zeros to get an array of 0s of length n columns
    #Step 2. Calculate the predicted value using sigmoid(z), where z = np.dot(X, w) dot product creating a m x 1 matrix of predicted y values 
    #Step 3. Compute the cross entropy loss of the predicted y value against the actual value y
    #Step 4. Compute the partial derivate of each weight and bias against the loss function
    #Step 5. Update the values of weights, w and bias, b using gradient descent with respective learning rate and partial derivative
    #Step 6. Do step 2 again for 1000 iterations to get a converged matrix of weights where the loss function is minimised(gradient of loss function is 0 respect to weight and bias)

    def fit(self, X, y, verbose = False):
        
        #adds column of 1s with m rows
        #used so that weight and bias matrix is combined with bias as the first value
        if self.fit_intercept:
            X = self.add_intercept(X)

        m, n = X.shape
        self.w = np.zeros(n) #initialises the weights of n columns which represents all the features  at first iteration to be zero


        for i in range(self.num_iterations):
            z = np.dot(X, self.w)
            y_hat = self._sigmoid(z) #

            #dot product between transposed X against the error(difference between predicted and actual value)
            gradient = (1/m) * np.dot(X.T, (y_hat - y)) 

            #gradien descent, updates for every iteration which nudges the weights to local minima(zero gradient), converges 
            self.w -= gradient * self.lr

            #prints the loss function value for every 5 steps
            if self.verbose and i % (self.num_iter // 5) == 0:
                loss = - (1/m) * np.sum(y * np.log(y_hat + 1e-9) + (1 - y) * np.log(1 - y_hat + 1e-9))
                print(f"Iteration {i}: Loss = {loss:.6f}")


     
