#import libraries
import numpy as np
import matplotlib.pyplot as plt
import math

#note that H1.txt and y1.txt must be saved to the same directory as this script

#import y1
file_y = open("y1.txt")
y_content = file_y.read()
file_y.close()
y = y_content.split("\n")
# print(y1)
y = y[:-1]
y1 = np.array([int(i) for i in y])


#import H1
file_h = open("H1.txt")
H_content = file_h.read()
H1 = H_content.split("\n")
H1 = H1[:-1]
H = [np.array(i.split(), dtype=int) for i in H1]
h1 = np.vstack(H)


def decode(H, y, p=.1, max_iter = 20):

    success = -1
    m, n = H.shape

    #list of factor checks
    B = []
    for i in range(m):
        B.append(np.where(H[i]==1)[0])
    # print("Bit Check:",B)

    #probability y given x

    #if true val is 1
    P_y_x1 = np.zeros(len(y))
    #if true val is 0
    P_y_x0 = np.zeros(len(y))

    for i in range(len(y)):
        P_y_x1[i] = p**((y[i]+1)%2) * (1-p)**((y[i]+1+1)%2)
        P_y_x0[i] = p**(y[i]) * (1-p)**((y[i]+1)%2)

    #initialize
    M = np.zeros((m,n))

    # print(M)

    for i in range(m):
        for j in range(n):
            #for efficiency: take difference and use log liklihood:
            # then u(0) - u(1) -> log(u(0))-log(u(1)) = log(u(0)/u(1))
            M[i,j] = np.log([P_y_x0[j]/P_y_x1[j]])
    
    # print("Initial Message:\n",M)
   

    for iter in range(max_iter):
        
        #factor to variable 
        M_f = np.zeros((m,n))

        for j in range(len(B)):
                
            new_row = np.zeros(n)
            # print("\nrow:", j)

            #get values of M for all indexes where h==1 for the jth row
            prod = M[j,(B[j])]
            # print(prod)
            
            for i in range(len(prod)):

                # print("\nindex:", B[j][i])

                #take product of all incoming messages excludng the ith bit
                to_prod = np.hstack((prod[0:i],prod[i+1:len(prod)]))
                product = np.prod(np.tanh(to_prod/2))
                to_log = (1+product)/(1-product)
                # print("log:",to_log)
                
                new_row[B[j][i]] = np.log(to_log)
            
            # print(j, new_row)
                
            M_f[j,:] = new_row
            # M[j,:] = new_row
        
        # print(M_f)

        #calc log liklihood of posterior
        posterior = np.sum(M_f, axis = 0)
        
        for i in range(n):
            # calc log like
            posterior[i] += np.log([P_y_x0[i]/P_y_x1[i]])
        
        # print("\nposterior",posterior)
        
        #update prediction
        z = np.zeros(n)
        for i in range(n):
            if posterior[i]>0:
                z[i]=0
            else:
                z[i]=1
        
        # print("\nprediction:", z)
            
        #test, dont forget mod 2
        test = np.matmul(H, z.T)% 2

        #check if parity conditions met or reached max iterations
        if iter == max_iter or np.all(test == 0):
            success = 0
            break
        
        #if not continue
        else:
            
            #update variable to factor
            for i in range(n):
                for j in range(m):
                    M[j,i] = posterior[i] - M_f[j,i]
    
    return success, iter+1, z

#decode the message
success,iter,decoded = decode(h1,y1, p=.1, max_iter=20)
print(success, "\niterations:",iter, "\nDecoded vector:\n",decoded)

#the original message in ascii is below:

#translate 
x = np.reshape(decoded.astype(int), (125,8))
x = x.astype(str)
lst = x.tolist()

b = []
for i in range(len(lst)):
    b.append(''.join(map(str, lst[i])))

for i in range(31):
    print(chr(int(b[i],2)))