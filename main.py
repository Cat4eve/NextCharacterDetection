import numpy as np

class RNN:
    def __init__(self, iteration_count=1000, learning_rate=1e-1):
        self.learning_rate = learning_rate
        self.iteration_count = iteration_count

        self.ini = False

        self.letters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ,.!`"? '

        self.t = {}
        for i, v in enumerate(self.letters):
            self.t[v] = np.zeros(len(self.letters), dtype=int)
            self.t[v][i] = 1
        
    def __call__(self, X, y=None, training=True):
        
        n = X.shape[0]
        d = len(self.letters)        

        if not self.ini:
            # Model parameters
            self.Wxh = np.random.randn(n, d) * 0.01  # input to hidden
            self.Whh = np.random.randn(n, n) * 0.01  # hidden to hidden
            self.Why = np.random.randn(d, n) * 0.01  # hidden to output
            self.h = np.zeros((n, n)) # hidden encoded
            self.hb = np.zeros(n) # hidden bias
            self.yb = np.zeros(d) # output bias

            self.ini = True


        if not training: return self.forward(X)


        for i in range(self.iteration_count):
            self.backward(X, y)
            

        return self.forward(X)
    

    def forward(self, X):
        y = []
        n = X.shape[0]
        d = len(self.letters)

        for i in range(n):
            t = np.tanh(
                    self.Whh @ self.h[i]
                    + self.Wxh @ X[i]
                    + self.hb[i]
                )
            if i < n - 1: self.h[i+1] = t[i]
            y.append(self.Why @ t + self.yb) 

        return self.softmax(np.array(y))
    

    def backward(self, X, y):

        n = X.shape[0]
        d = len(self.letters)

        cross_entropy_derivative = self.forward(X) - y
        tanh_derivative = 1 - self.forward(X)**2

        self.dWxh = np.zeros((n, d))
        self.dWhh = np.zeros((n, n))
        self.dWhy = np.zeros((d, n))
        self.dhb = np.zeros(n)
        self.dyb = np.zeros(d)

        for i in range(n):
            self.dWxh =  tanh_derivative @ self.Why @ (X[i] + self.dWxh)
            self.dWhh = (self.h[i] + self.dWhh) @ (1-self.forward(X)**2) @ self.Why
            self.dWhy += (self.h[i][np.newaxis] @ cross_entropy_derivative).T
            self.dhb = tanh_derivative @ self.Why
            self.dyb = tanh_derivative
            

        self.Wxh -= self.learning_rate * self.dWxh @ cross_entropy_derivative
        self.Whh -= self.learning_rate * self.dWhh @ cross_entropy_derivative
        self.Why -= self.learning_rate * self.dWhy / n
        self.hb -= self.learning_rate * self.dhb @ cross_entropy_derivative
        self.yb -= self.learning_rate * self.dyb @ cross_entropy_derivative

        print(self.loss(y, self.forward(X)))

    
    def one_hot(self, text):
        a = []
        for i in range(len(text)):
            a.append(self.t[text[i]])
        return np.array(a)
    
    def loss(self, y_true, y_pred):
        return -np.sum(y_true * np.log(y_pred))
    
    def softmax(self, x):
        return np.exp(x)/np.sum(np.exp(x))
    
    def show_prediction_as_letters(self, x):
        n = x.shape[0]
        d = x.shape[1]
        z = self.softmax(x)
        y = ''

        for i in range(n):
            m = max(z[i])
            p = np.zeros(d)
            for j in range(d):
                p[j] = 0 if z[i][j] < m else 1

            v = ''
            for j in range(d):
                if (self.t[self.letters[j]] == p).all():
                    v += self.letters[j]
            y += v
        
        return y
    



text = "Lorem ipsum dolor sit amet."
rnn = RNN()
text1 = rnn.one_hot(text[:-1])
text2 = rnn.one_hot(text[1:])

rnn(text1, text2, True)