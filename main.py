import numpy as np

class RNN:
    def __init__(self, iteration_count=1000, learning_rate=1e-1):
        self.learning_rate = learning_rate
        self.iteration_count = iteration_count

        self.t = {}
        for i, v in enumerate('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'):
            self.t[v] = np.zeros(52, dtype=int)
            self.t[v][i] = 1
        
    def __call__(self, X, y=None, training=True):
        if not training: return self.forward(X)

        n = X.shape[0]
        self.n = n

        # Model parameters
        self.Wxh = np.random.randn(n, n) * 0.01  # input to hidden
        self.Whh = np.random.randn(n, n) * 0.01  # hidden to hidden
        self.Why = np.random.randn(n, n) * 0.01  # hidden to output
        self.h = np.random.randn((n, n)) * 0.01 # hidden encoded
        self.hb = np.zeros(n) # hidden bias
        self.b = 0 # output bias


        for i in self.iteration_count:
            self.backward(X, y)
            loss = self.loss(self.forward(X), y)
            if i % 100 == 0: print(loss)


        return self.__call__(X, y, False)
    

    def forward(self, X):
        y = []
        for i in range(self.n):
            t = np.tanh(self.Whh @ self.h[i] + self.Wxh @ X[i] + self.hb[i])
            y.append(self.Why @ t + self.b) 
        return np.exp(y) / np.sum(np.exp(y))
    

    def backward(self, X, y):
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

    def loss(self, y_pred, y_true):
        loss = -np.sum(y_true * np.log(y_pred + 1e-8))
        return loss
    


rnn = RNN()

text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
text = text.split(" ")

rnn(
    [rnn.one_hot(t[0:-1]) for t in text], 
    rnn.one_hot(text)
)