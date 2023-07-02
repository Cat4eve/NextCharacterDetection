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
        # print(self.h[59].shape, self.Whh.shape)

        for i in range(n):
            t = np.tanh(
                    self.Whh @ self.h[i]
                    + self.Wxh @ X[i]
                    + self.hb[i]
                )
            if i < n - 1: self.h[i+1] = t[i]
            y.append(self.Why @ t + self.yb) 

        return self.show_prediction_as_letters(np.array(y))
    

    def backward(self, X, y):

        cross_entropy_derivative = self.forward(X) - y

        for x in X:
            pass

        self.dWxh = (1 - self.forward(X)**2) @ self.Why @ self.Whh
        self.dWhh = (1 - self.forward(X)**2) @ self.Whh
        self.dWhy = (1 - self.forward(X)**2)
        self.dh = (1 - self.forward(X)**2) @ self.Whh
        self.dhb = (1 - self.forward(X)**2)
        self.db = 1
            

        self.Wxh -= self.learning_rate * self.dWxh
        self.Whh -= self.learning_rate * self.dWhh
        self.Why -= self.learning_rate * self.dWhy
        self.hb -= self.learning_rate * self.dhb
        self.b -= self.learning_rate * self.db
        self.h -= self.learning_rate * self.dh

    
    def one_hot(self, text):
        a = []
        for i in range(len(text)):
            a.append(self.t[text[i]])
        return np.array(a)
    
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
    



text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
rnn = RNN()
text = rnn.one_hot(text)

y1 = rnn(text, None, False)
y2 = rnn.forward(text)
print(y2)