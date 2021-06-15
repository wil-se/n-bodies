#!/usr/bin/python

import random
bodies_number = input()

print(bodies_number)
for i in range(int(bodies_number)):
    print("{} {} {} {} 0 0 0".format(
        random.randint(200, 3000) * pow(-1, random.randint(0, 1)),                 # x
        random.randint(200, 3000) * pow(-1, random.randint(0, 1)),                 # y
        random.randint(200, 3000) * pow(-1, random.randint(0, 1)),                 # z
        random.randint(20, 100),                 # mass
        #random.randint(200, 1000) * pow(-1, random.randint(0, 1)),                 # sx
        #random.randint(200, 1000) * pow(-1, random.randint(0, 1)),                 # sy
        #random.randint(200, 1000))
        
        ))                # sz
