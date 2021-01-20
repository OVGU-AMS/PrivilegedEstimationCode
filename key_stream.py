"""

"""

import numpy as np
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes

class KeyStreamPairFactory:
    @staticmethod
    def make_pair(dimension, min_range, max_range):
        key = get_random_bytes(16)
        cipher1 = AES.new(key, AES.MODE_CTR)
        cipher2 = AES.new(key, AES.MODE_CTR, nonce=cipher1.nonce)
        return KeyStream(cipher1, dimension, min_range, max_range), KeyStream(cipher2, dimension, min_range, max_range)


class KeyStream:
    def __init__(self, cipher, dimension, min_range, max_range):
        self.cipher = cipher
        self.dimension = dimension

        # How to interpret random bytes into random float
        self.bytes_to_read = 16
        self.read_int_divisor = 2**(16*8-1) - 1
        self.min_float = min_range
        self.max_float = max_range
        return
    
    def next(self):
        float_list = []
        for _ in range(self.dimension):
            next = self.cipher.encrypt(self.bytes_to_read*b'\x00')
            next_int = int.from_bytes(next, byteorder='big', signed=True)
            next_float = (next_int/self.read_int_divisor) * (self.max_float - self.min_float) + self.min_float
            float_list.append(next_float)
        return np.array(float_list)



# a, b = KeyStreamPairFactory.make_pair()
# for i in range(1000):
#     print(a.next())
# #print(b.next())
# #print(a.next())
# #print(b.next())
# #print(a.next())
# #print(b.next())
