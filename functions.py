from random import randint
import numpy as np
from numpy import matmul
from numpy.linalg import norm
from collections import namedtuple
from copy import deepcopy
from random import sample

def gen_next(chunk, A):
    '''A is a list of lists'''   
    l = len(chunk)
    for a in A:
        L = len(a)
        for i in range(L):
            if a[i:i+l] == chunk:
                yield a[i+1:i+l+1]

def get_gen_len(g):
    i = 0
    while 1:
        try:
            next(g)
        except StopIteration:
            break
        i += 1
    return i
    

class ngram:
    def __init__(self, n, train_txt):
        self.n = n
        self.train_txt = train_txt

    def chunking(self, num, idx):
        chunk = ''
        for i in range(num):
            chunk += self.train_txt[idx+i]
        return chunk

    def freq_count(self, num, target):
        count = 0
        for i in range(len(self.train_txt)-(self.n-1)):
            if self.chunking(num, i) == target:
                count += 1
        return count

    def sentence_prob(self, data, show=False):
        p, skip = 1, 0
        for i in range(len(data)-self.n+1):
            target_numer, target_denom = data[i:i+self.n], data[i:i+self.n-1]   
            numer = self.freq_count(self.n, target_numer)
            denom = self.freq_count(self.n-1, target_denom)
            if show == True:
                print(f'{target_numer}/{target_denom}')
                print(f'{numer}/{denom}')
            try:
                p *= numer/denom
            except ZeroDivisionError:
                skip = 1
        return p if skip == 0 else 'division by zero'
        
    def get_chunk_occurance(sentences, n):
        chunks = sum([['/'.join(s[i:i+n-1]) for i in range(len(s)-n+2)]
                                        for s in sentences], [])
    
        keys = set([i for i in chunks])
        my_dict = {}
        for k in keys:
            my_dict[k] = chunks.count(k)
        return my_dict

    def generate(self):
        '''Randomly generates sentences based on the N-gram algorithm'''
        lines = deepcopy(self.train_txt)
        for l in lines:
            for i in range(self.n-1):
                l.insert(0,'b')
            l.append('\n')
        
        chunk = lines[randint(0, len(lines)-1)][:self.n-1]
        sentence = chunk[0]

        while 1:
            g1, g2 = gen_next(chunk, lines), gen_next(chunk, lines)
            idx = randint(1, get_gen_len(g1))
            for i in range(idx):
                chunk = next(g2)                 
            sentence += chunk[0]
            if '\n' in chunk:
                sentence += ''.join(chunk[1:])
                break    
        return sentence[self.n-1:-1]


#def indicator(condition):
    #if eval(condition):
        #return 1
    #else:
        #return 0

def subgradient(x, y, w, b, c):
    Grad = namedtuple('Grad', ['w', 'b'])
    N = len(y)
    s_w, s_b = np.zeros(x.shape[1]), 0
    for i in range(N):
         #condition = 'y[i]*(matmul(w.transpose(), x[i]) - b) < 1'
         ind = 1 if y[i]*(matmul(w.transpose(), x[i]) - b) < 1 else 0
         #s_w += -y[i]*indicator(condition)*x[i]
         #s_b += -y[i]*indicator(condition)
         s_w += -y[i]*ind*x[i]
         s_b += -y[i]*ind
    grad = Grad(s_w + 2*c*w, s_b)
    return grad

def subGD(alpha, batch_size, x, y, w0, b0, c, epsilon=None, n_epochs=1000):
    N, w, b = len(y), w0, b0
    n_iters = int(N/batch_size)
    #while 1:
    for i in range(n_epochs):
        for j in range(n_iters):
            w_pre, b_pre = w, b
            subgrad = subgradient(x[j*batch_size:(j+1)*batch_size], 
                                  y[j*batch_size:(j+1)*batch_size],
                                  w, b, c)
            w -= alpha*subgrad.w
            b -= alpha*subgrad.b
            #if norm(w - w_pre) < epsilon and abs(b - b_pre) < epsilon:
                #break
    return w, b

def assign(data, centroids):
    res = [None]*len(data)
    
    for i, x in enumerate(data):
        Ds = [norm(x - y) for y in centroids]
        res[i] = Ds.index(min(Ds))

    return res

def update(data, res):
    groups = [[] for i in set(res)]
    for i, x in enumerate(data):
        groups[res[i]].append(x)
    groups = [np.array(i) for i in groups]
    centroids = [np.mean(i, axis=0) for i in groups]

    return centroids

def my_kmeans(data, k, centroids=None):
    if centroids == None:
        centroids = sample(list(data), k)

    previous = None
    while 1:
        res = assign(data, centroids)
        centroids = update(data, res)
        if previous == res:
            break
        previous = res

    return np.array(res)