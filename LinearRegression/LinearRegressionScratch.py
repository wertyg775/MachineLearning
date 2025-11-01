import numpy as np

class LinearRegressionScratch:
    #lr is the learning rate(rate the gradient descent updates the weights)
    #num_iter is the iterations of how much the gradient descent algorithm is ran,
    #fit_intercept : bool(include the bias term or not)
    def __init__(self, lr =1e-3, num_iter=1000, fit_intercept = True,):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        #w is the matrix of weights
        self.w = None
        #b is the bias term
        self.b = None


    #add a column of ones to input matrix so that when multiplied it becomes 1 . b to get the intercept
    def _add_intercept(self, X):
        if not self.fit_intercept:
            return X
        #creates a matrix of 1s with m rows and 1 column
        ones = np.ones((X.shape[0], 1))
        #adds the matrix of 1s to the first column of dataframe X
        return np.hstack([ones, X])
    
    #fit using least squares method, (not gradien descent)
    #cons is its not scalable
    def fit_normal_equation(self, X, y):
        #Updates the dataframe to add 1s to 1st column
        X_aug = self._add_intercept(X)

        # @ is for matrix multiplication 
        A = X_aug.T @ X_aug
        b = X_aug.T @ y
        #find the value theta that solves the linear system A x theta = b
        self.theta = np.linalg.solve(A, b)

        if self.fit_intercept:
            #self theta is the full matrix of bias + weight, since 1s column is first , bias value is index 0
            self.b = self.theta[0]
            #the weights are from index 1 to final index
            self.w = self.theta[1:]

        else:
            self.b = 0.0
            self.w = self.theta

    #find the best fit using gradient descent
    def _fit_gradient_descent(self, X, y, verbose = False):

        #rows and columns of X
        m, n = X.shape

        #Initialise parameters of weight and bias
        #create a zero matrix of n columns
        self.w = np.zeros(n)
        self.b = 0.0

        #update the weights and bias , w and b for 1000 iterations to get a converged value
        for i in range(self.num_iter):
            #get the prediction of the for the current iteration
            # X is matrix multiplied by weights, w and added the bias term
            y_pred = X.dot(self.w) + self.b

            #compute the error(how far off the prediction is against the actual value of y)
            #gets a matrix of error for m rows , 1 column
            error = y_pred - y

            #gradient of cost function with respect to w and bias
            dw = (1/m) * (X.T.dot(error))
            db = (1/m) * np.sum(error)

            #gradient descent step: w = w - dw * lr
            self.w -= self.lr * dw
            self.b -= self.lr * db




    


