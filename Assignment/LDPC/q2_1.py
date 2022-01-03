#import libraries
import numpy as np
import matplotlib.pyplot as plt
import math

def rref(H):

    ''' return reduce row echelon form'''
    m, n = H.shape
    # k = n-m

    i = 0
    j = 0
    r = 0

    H_hat = H.copy()

    while True:

        if i >= m or j >= n:
            break
        
        # print(H_hat)
        
        #check if entry = 0
        if H_hat[i,j]== 0:

            #find next row with non zero entry to swap
            r = i

            while r < m and H_hat[r,j] == 0 :
                r += 1
            
            #if no nonzero found skip to next column
            if r == m:
                j+=1
                break
        
        #swap row
        zero_row = H_hat[i,:].copy()
        nonzero = H_hat[r,:].copy()
        H_hat[i,:] = nonzero
        H_hat[r,:] = zero_row
        
        # print(H_hat)

        #perform row elimination
        for row in range(m):
            if row == i:
                continue
            #if entry does not = 0, add nonzero row, remember this is in F2 (so mod 2 addition)
            if H_hat[row,j] != 0:
                H_hat[row,:] = (H_hat[row,:] + H_hat[i,:]) % 2
        
        # print(H_hat)

        i+=1
        j+=1
    
    return H_hat

def gen_G(H):
    m, n = H.shape
    k = n-m

    H_hat = rref(H)
    # print("H_hat:\n",H_hat)
    P = H_hat[:, m:n]
    # print("P:\n",P)
    G = np.vstack((P, np.identity(k)))
    # print("G\n",G) 

    return H_hat, G


#initialize H
H = np.array([[1, 1, 1, 1, 0, 0],[0, 0, 1, 1, 0, 1],[1, 0, 0, 1, 1, 0]], dtype=int)

H_hat, G = gen_G(H)

print("H_hat:\n",H_hat, "\nG:\n", G)