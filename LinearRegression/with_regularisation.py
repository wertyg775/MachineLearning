import numpy as np

class with_regularisation:

#reg strentgh is lambda()
    def __init__(self, lr = 1e-3, num_iterations = 1000, fit_intercept = True, reg= None, reg_strength= 0.0):
        self.lr = lr
        self.num_iterations = 1000
        self.fit_intercept = fit_intercept
        self.reg = reg
        self.reg_strength = reg_strength
        self.w = None
        self.b = None


    def add_intercept(self, X):
        if not self.fit_intercept:
            return X

        else:
            ones = np.ones(X.shape[0] , 1)
            X_aug = np.hstack([ones, X])
            return X_aug
        

    def fit_normal_equation(self, X, y):
        X_aug = self.add_intercept(X)
        m, n = X_aug.shape #m is row , n is columns
        #Build regularisation matrix
        if self.reg == 'l2' and self.reg_strength > 0 :

            #first column of 1s(represent intercept aka bias) is not regularised/punished 
            L = np.eye(n) #creates identity matrix of n x n
            
            #Make the the first value to 0 to not regularise the intercept
            if self.fit_intercept:
                L[0,0] = 0

            L *= self.reg_strength #creates a matrix of lambda(reg strength) values 
            A = X_aug.T @ X_aug + L

        else: #if no regularisation
            A = X_aug.T @ X_aug
        
        #solve for Aθ = B
        B = X_aug @ y
        self.theta = np.linalg(A, B)

        if self.fit_intercept:
            self.b = self.theta[0]
            self.w = self.theta[1:]

        else:
            self.b = 0.0
            self.w = self.theta

    def fit_gradient_descent(self, X, y, verbose = False):
        m, n = X.shape

        #First initialise parameters for first iteration of gradient descent
        self.w = np.zeros(n)
        self.b = 0.0

        #for 1000 number of iterations , calculate the mean square error and then the partial derivatives of the current iteration
        for i in range (self.num_iterations):
            pred = X.dot(self.w) + self.b
            error = pred - y
            dw = (1/m) * (X.T.dot(error))
            db = (1/m) * np.sum(error)

            if self.reg == 'l2' and self.reg_strength > 0:
                dw += (self.reg_strength / m) * self.w
            elif self.reg == 'l1' and self.reg_strength > 0:
                dw += (self.reg_strength / m) * np.sign(self.w)

            self.w -= self.lr * dw
            self.b -= self.lr * db

            if verbose and (i % (self.num_iterations // 5) == 0):
                loss = (0.5/m) * np.sum(error**2)
                print(f"iter {i}, loss {loss:.6g}")











