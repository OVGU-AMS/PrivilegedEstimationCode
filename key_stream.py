"""

"""

import numpy as np
import scipy as sp
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes

class KeyStreamPairFactory:
    @staticmethod
    def make_pair():
        key = get_random_bytes(16)
        cipher1 = AES.new(key, AES.MODE_CTR)
        cipher2 = AES.new(key, AES.MODE_CTR, nonce=cipher1.nonce)
        return KeyStream(cipher1), KeyStream(cipher2)


class KeyStream:
    def __init__(self, cipher):
        self.cipher = cipher
        self.bytes_to_read = 16
        self.max_read_int = 2**(16*8) - 1
        return
    
    def next(self):
        next_read = self.cipher.encrypt(self.bytes_to_read*b'\x00')
        next_int = int.from_bytes(next_read, byteorder='big', signed=False)
        return next_int
    
    def next_as_std_uniform(self):
        next_int = self.next()
        next_unif = (next_int/self.max_read_int)
        return next_unif
    
    def next_n_as_gaussian(self, n, mean, covariance):
        next_unif = np.array([self.next_as_std_uniform() for _ in range(n)])
        next_gauss = sp.stats.multivariate_normal.cdf(next_unif, mean=mean, cov=covariance)
        return next_gauss



# a, b = KeyStreamPairFactory.make_pair()
# for i in range(1000):
#     print(a.next())
# #print(b.next())
# #print(a.next())
# #print(b.next())
# #print(a.next())
# #print(b.next())
