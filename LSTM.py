class EmbeddingLayer:
    
    def __init__(self, W):
        self.params = [W] # (V, D)
        self.grads = [np.zeros_like(W)]
        self.idx = None
     
    def forward(self, x):
        # input: x (N, 1)
        
        W, = self.params
        self.idx = x
        
        return W[x] # (N, D)
        
    def backward(self, dx):
        # input: dx (N, D)
        
        dW, = self.grads
        dW[...] = 0
        np.add.at(dW, self.idx, dx) # dW self.idx 행에 dx더함 -> self.grads도 같이 바뀜
        
        return None

class Embedding:
    
    def __init__(self, W):
        
        self.params = [W] # (V, D)
        self.grads = [np.zeros_like(W)]
        self.W = W
        self.layers = []
        
    def forward(self, input_x):
        # input: input_x  (N, T)
        # output: x       (N, T, D)
        
        N, T = input_x.shape
        V, D = self.W.shape
        
        x = np.empty((N, T, D), dtype='f')
        for t in range(T):
            layer = EmbeddingLayer(self.W)
            x[:, t, :] = layer.forward(input_x[:, t]) # (N, D)
            self.layers.append(layer)
        
        return x
        
    def backward(self, dx):
        # input dx: (N, T, D)
        
        N, T, D = dx.shape
        
        grad = 0
        for t in range(T):
            layer = self.layers[t]
            layer.backward(dx[:, t, :])
            grad += layer.grads[0]    # (V, D)
        
        self.grads[0][...] = grad
         
        return None
    
    
class LSTMLayer:
    
    def __init__(self, Wx, Wh, b):
        
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx),np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None
        
    def forward(self, x, h_prev, c_prev):
        
        Wx, Wh, b = self.params # (D, 4H), (H, 4H) (4H, )
        N, H = h_prev.shape
        A = np.matmul(x, Wx) + np.matmul(h_prev, Wh) + b # (N, 4H)
        
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        
        f = sigmoid(A[:, :H])
        g = np.tanh(A[:, H:2*H])
        i = sigmoid(A[:, 2*H:3*H])
        o = sigmoid(A[:, 3*H:])
        
        c_next = f * c_prev + i * g
        h_next = o * np.tanh(c_next)
        
        self.cache = (x, h_prev, c_prev, f, g, i, o, c_next)
        
        return h_next, c_next
        
    def backward(self, dh_next, dc_next):
        
        Wx, Wh, b = self.params
        x, h_prev, c_prev, f, g, i, o, c_next = self.cache
        
        tanh_c_next = np.tanh(c_next)
        ds = dc_next + (dh_next * o) * (1 - tanh_c_next**2)
        
        dc_prev = ds * f
        df = ds * c_prev * f * (1 - f)
        dg = ds * i * (1 - g**2)
        di = ds * g * i * (1 - i)
        do = dh_next * tanh_c_next * o * (1-o)
        dA = np.hstack([df, dg, di, do]) # (N, 4H)
        
        dWx = np.matmul(x.T, dA)         # (D, N) (N, 4H)
        dx = np.matmul(dA, Wx.T)         # (N, 4H) (4H, D)
        
        dWh = np.matmul(h_prev.T, dA)    # (H, N) (N, 4H) 
        dh_prev = np.matmul(dA, Wh.T)    # (N, 4H) (4H, H)
            
        db = np.sum(dA, axis=0)
        
        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        return dx, dh_prev, dc_prev
        
class LSTM:
    
    def __init__(self, Wx, Wh, b, stateful=False):
        
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        
        self.stateful = stateful
        self.h, self.c = None, None
        self.dh = None
        self.layers = []
        
    def forward(self, x):
        # input: x  (N, T, D)
        # output: h (N, T, H)
        
        Wx, Wh, b = self.params
        N, T, D = x.shape
        H = Wh.shape[0]
        
        if self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f')
        if self.stateful or self.c is None:
            self.c = np.zeros((N, H), dtype='f')
        
        h = np.empty((N, T, H), dtype='f')
        for t in range(T):
            layer = LSTMLayer(Wx, Wh, b)
            self.h, self.c = layer.forward(x[:, t, :], self.h, self.c)
            h[:, t, :] = self.h
            
            self.layers.append(layer)
        
        return h
        
    def backward(self, dh):
        # input: dh  (N, T, H)
        # output: dx (N, T, D)
        
        Wx, Wh, b = self.params
        N, T, H = dh.shape
        D = Wx.shape[0]
    
        dx = np.empty((N, T, D), dtype='f')
        _dh, dc = 0, 0
        grads = [0, 0, 0]
    
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, _dh, dc = layer.backward(dh[:, t, :]+ _dh, dc)
            dx[:, t, :] = dx
            
            for i, grad in enumerate(layer.grads):
                grads[i] += grad
        
        for i, grad in enumerate(grads):
            self.grads[i][...] = grad
        
        self.dh = _dh
        
        return dx
        
    def set_state(self, h, c=None):
        self.h = h
        self.c = c

    def reset_state(self):
        self.h = None
        self.c = None
    
class Affine:
    
    def __init__(self, Wa, ba):
        
        self.params = [Wa, ba] # (H, V), (V, )
        self.grads = [np.zeros_like(Wa), np.zeros_like(ba)]
        self.cache = None
    
    def forward(self, h):
        # input: h  (N, T, H)
        # output: a (N, T, V)
        
        Wa, ba = self.params
        N, T, H = h.shape
        
        h = h.reshape(N*T, -1)     # (NT, H)
        a = np.matmul(h, Wa) + ba  # (NT, V)
        a = a.reshape(N, T, -1)    # (N, T, V)
        
        self.cache = h
        
        return a
    
    def backward(self, da):
        # input: da  (N, T, V)
        # output: dh (N, T, H)
        
        Wa, ba = self.params
        N, T, V = da.shape
        h = self.cache             # (NT, H)    
        da = da.reshape(N*T, -1)   # (NT, V)
        
        dWa = np.matmul(h.T, da)   # (H, NT) (NT, V)
        dba = np.sum(da, axis=0)   # (V, )
        dh = np.matmul(da, Wa.T)   # (NT, V) (V, H) 
        dh = dh.reshape(N, T, -1)  # (N, T, H)
    
        self.grads[0][...] = dWa
        self.grads[1][...] = dba
    
        return dh
    
    
class Softmax:
    
    def __init__(self):
        self.cache = None
        
    def forward(self, a, label):
        # input: a(N, T, V) label(N, T)
        # output: y(N, T)
        
        N, T, V = a.shape
        a = a.reshape(N*T, -1)                           # (NT, V)
        label = label.reshape(N*T)                       # (NT, )
        
        # softmax 계산
        a = a - a.max(axis=1, keepdims=True)
        a_exp = np.exp(a)
        a_stm = a_exp / a_exp.sum(axis=1, keepdims=True) # (NT, V)
        
        # 정답 label만 선택 
        y = a_stm[np.arange(N*T), label]                 # (NT, )
        self.cache = (y, label, a_stm)
        
        return y.reshape(N, T)
    
    def backward(self, dy):
        # input: dy  (N, T)
        # output: da (N, T, V)
        
        N, T = dy.shape
        dy = dy.reshape(N*T)                              # (NT, )
        y, label, a_stm = self.cache                      # (NT, )
        
        a_stm[np.arange(N*T), label] = dy * (y * (1 - y)) # (NT, V)
        a_stm = a_stm/(N*T)
        da = a_stm.reshape(N, T, -1)                      # (N, T, V)

        return da
    
    
class CEloss:
    
    def __init__(self):
        self.cache = None
        
    def forward(self, y):
        # input: y     (N, T)
        # output: loss (1, 1)
        
        N, T = y.shape
        _y = y.reshape(N*T) # (NT, )
        
        loss = -np.sum(np.log(_y))
        loss = loss/(N*T)
        self.cache = y
        
        return loss
    
    def backward(self, dloss=1):
        # input: dloss  
        # output: dy (N, T)

        y = self.cache      
        N, T = y.shape
        
        y = y.reshape(N*T)   # (NT, )
        dy = dloss * (-1/y) 
        
        return dy.reshape(N, T)
    