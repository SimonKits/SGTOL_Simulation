import numpy as np
import matplotlib.pyplot as plt
import random


def DTheta_estimator_fixed_theta(Theta,X_0,dX_0,demand,N,S,h,b,p):
    '''
    Single run of the IPA for a fixed value of Theta.
    Returns a number which is a single estimation of D(Theta).
    '''
    #setting values for X_0(theta) and X'_0(Theta)
    X_cur = X_0
    dX_cur = dX_0

    #we will also need to remember these values one step back
    X_prev = 0
    dX_prev = 0

    #array for the derivatives C'_k(Theta)
    C_derivatives = np.zeros(N)
    
    for k in range(N):
    
        #calculate the derivative of C_k(Theta) in parts
        if X_cur < 0: 
            hold_der = 0
        elif k == 0:
            hold_der = dX_0
        else:
            hold_der = 1 if (0 <= Theta - X_prev + demand[k-1] <= S) else dX_prev
    
        back_der = -1*dX_cur if (X_cur <= demand[k]) else 0
    
        prod_der = (1 - dX_cur) if (0 <= Theta - X_cur - demand[k] <= S) else 0
    
    
        #Adding the derivative of C_k(Theta) to the derivatives array
        C_derivatives[k] = h*hold_der + b*back_der + p*prod_der
    
    
        #updating X'(Theta) we must also remember the previous value
        dX_prev = dX_cur
        dX_cur = 1 if ( 0<= Theta - X_cur + demand[k] <= S) else dX_prev
    
        #updating X(Theta) again we must remember the previous value
        X_prev = X_cur
        production =  min(S,max(Theta-X_cur +demand[k],0))
        X_cur = X_prev - demand[k] + production
    
    #sum C'_k and divide by N
    DTheta_estimator  = 1/N*sum(C_derivatives)
    
    return(DTheta_estimator)


def Theta_loop(scale,begin,end,stepsize,N,S,h,b,p):
    '''
    loops over all Theta in range with certain stepsize and outputs 
    and array with one estimations of D(Theta) for each Theta. For each Theta
    the same array of exponential demands is used.
    '''
    
    demand = np.random.exponential(scale,N) #exponential demands
    runs = int((end - begin)/stepsize) + 1 #number of individual Theta's in range
    DTheta_estimators = np.zeros(runs)
    
    for i in range(runs): 
        Theta = begin + i*stepsize
        X_0  = Theta #specified in problem description
        dX_0 = 1 #this value will have to be set manually as it was derived symbolically
        DTheta_estimator = DTheta_estimator_fixed_theta(Theta,X_0,dX_0,demand,N,S,h,b,p)
        DTheta_estimators[i] = DTheta_estimator
    
    return(DTheta_estimators)


def IPA():
    '''
    Calculates IPA estimator for D(Theta). Returns two arrays. One for D(Theta)
    within given range and certain step size, and another for all the Theta values
    '''
    scale = 15 #scale of exponentially distributed demands
    begin,end,stepsize = 10,40,0.1 #range of Theta values
    S,h,b,p = 20,4,8,10 #parameters of inventory problem
    N = 20 #observation periods
    n = 1000 #number of runs Theta_loop algorithm
    
    #the output arrays
    IPA = np.zeros(int((end - begin)/stepsize) + 1)
    Theta_axis = np.arange(begin,end + stepsize,stepsize)
    
    #loop Theta_loop function
    for i in range(n):
        DTheta_estimators = Theta_loop(scale,begin,end,stepsize,N,S,h,b,p)
        IPA = IPA + DTheta_estimators
    
    #we must still divide IPA by the number of runs
    IPA = 1/n*IPA
    
    return(IPA,Theta_axis)
    

def main():
    y, x = IPA()
    plt.plot(x,y)
    plt.show()    
    
    
if __name__ == '__main__':
    main()
