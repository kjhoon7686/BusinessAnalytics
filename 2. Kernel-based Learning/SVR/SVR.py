# reference : https://github.com/KS-prashanth/Epslion-SVR/blob/main/Epsilon_svr/epsilonsvr.py
# epsilon svr 
def eps_svr(X_train,Y_train,X_test,kernel,epsilon, c,kernel_param):
    """implements the CVXOPT version of epsilon SVR"""
    m, n = X_train.shape      #m is num samples, n is num features
    #Finding the kernels i.e. k(x,x')
    k = np.zeros((m,m))
    for i in range(m):
        for j in range(m):
            k[i][j] = kernel(X_train[i,:], X_train[j,:], kernel_param)

    P= np.hstack((k,-1*k))
    P= np.vstack((P,-1*P))
    q= epsilon*np.ones((2*m,1))
    qadd=np.vstack((-1*Y_train,Y_train))
    q=q+qadd
    A=np.hstack((np.ones((1,m)),-1*(np.ones((1,m)))))

    #define matrices for optimization problem       
    P = cvxopt.matrix(P)
    q = cvxopt.matrix(q)
    A = cvxopt.matrix(A)
    b = cvxopt.matrix(np.zeros((1,1)))

    c= float(c)
    temp=np.vstack((np.eye(2*m),-1*np.eye(2*m)))
    G=cvxopt.matrix(temp)

    temp=np.vstack((c*np.ones((2*m,1)),np.zeros((2*m,1))))
    h = cvxopt.matrix(temp)
    #solve the optimization problem
    sol = cvxopt.solvers.qp(P,q,G,h,A,b,solver='glpk')
    #lagrange multipliers
    l = np.ravel(sol['x'])
    #extracting support vectors i.e. non-zero lagrange multiplier
    alpha=l[0:m]
    alpha_star=l[m:]

    bias= sol['y']
    print("bias="+str(bias))
    #find weight vector and predict y
    Y_pred = np.zeros(len(X_test))
    for i in range(len(X_test)):
        res=0
        for u_,v_,z in zip(alpha,alpha_star,X_train):
            res+=(u_ - v_)*kernel(X_test[i],z,kernel_param)
        Y_pred[i]= res
    Y_pred = Y_pred+bias[0,0]

    return Y_pred