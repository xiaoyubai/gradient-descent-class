import numpy as np
from regression_functions import add_intercept

class GradientAscent(object):

    def __init__(self, cost, gradient, predict_func, fit_intercept=True, scale=True):
        '''
        INPUT: GradientAscent, function, function
        OUTPUT: None
        Initialize class variables. Takes two functions:
        cost: the cost function to be minimized
        gradient: function to calculate the gradient of the cost function
        '''
        # Initialize coefficients in run method once you know how many features
        # you have.
        self.coeffs = None
        self.cost = cost
        self.gradient = gradient
        self.predict_func = predict_func
        self.fit_intercept = fit_intercept
        self.scale = scale

    def run(self, X, y, alpha=0.01, num_iterations=10000, step_size=0.000001):
        '''
        INPUT: GradientAscent, 2 dimensional numpy array, numpy array
               float, int
        OUTPUT: None
        Run the gradient ascent algorithm for num_iterations repititions. Use
        the gradient method and the learning rate alpha to update the
        coefficients at each iteration.
        '''
        if self.scale:
            X = self.scale_X(X)
        if self.fit_intercept:
            X = add_intercept(X)
        self.coeffs = np.zeros(X.shape[1])

        if step_size:
            cost1 = self.cost(X, y, self.coeffs)
            while True:
                self.coeffs = self.coeffs + alpha * self.gradient(X, y, self.coeffs, l=1)
                cost2 = self.cost(X, y, self.coeffs)
                if abs(cost2 - cost1) < step_size:
                    print cost2
                    break
                else:
                    cost1 = cost2
        else:
            for i in xrange(num_iterations):
                self.coeffs = self.coeffs + alpha * self.gradient(X, y, self.coeffs, l=1)

    def s_run(self, X, y, alpha=0.01, num_iterations=100000, step_size=0.0000000001):

        if self.scale:
            X = self.scale_X(X)
        if self.fit_intercept:
            X = add_intercept(X)
        self.coeffs = np.zeros(X.shape[1])

        if step_size:
            cost1 = self.cost(X, y, self.coeffs)
            while True:
                a = range(len(X))
                np.random.shuffle(a)
                for i in a:
                    self.coeffs = self.coeffs + alpha * self.gradient(X[i], y[i], self.coeffs, l=1)
                    # if abs(cost2 - cost1) < step_size:
                    #     break
                    # else:
                    #     cost1 = cost2
                cost2 = self.cost(X, y, self.coeffs)
                if abs(cost2 - cost1) < step_size:
                    break
                else:
                    cost1 = cost2
        else:
            print "step_size parameter must be initialized!"
            for i in xrange(num_iterations):
                np.random.shuffle(X)
                for i, x in enumerate(X):
                    self.coeffs = self.coeffs + alpha * self.gradient(x, y[i], self.coeffs, l=1)

    def predict(self, X):
        '''
        INPUT: GradientAscent, 2 dimensional numpy array
        OUTPUT: numpy array (ints)
        Use the coeffs to compute the prediction for X. Return an array of 0's
        and 1's. Call self.predict_func.
        '''
        if self.scale:
            X = self.scale_X(X)
        if self.fit_intercept:
            X = add_intercept(X)
        return self.predict_func(X, self.coeffs)

    def scale_X(self, X):

        mean = X.mean(axis=0)
        std = X.std(axis=0)
        return (X - mean) / std
