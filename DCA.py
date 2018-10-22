import numpy as np
import cvxpy as cp
from time import time

XYZ = {}
XYZ['time'] = []
XYZ['value'] = []
XYZ['violation'] = []

def DCA(path, tau__):

    data_path = path
    tau = tau__
    tol = 1e-6
    max_iters = 1000
    opt_val_tol = 1e-6

    print_solving_process = None
    debug = False

    np.random.seed(2)

    def convertToListOfMatrices(list, K, numRow, numCol):
        newlist = []
        for i in range(K):
            list[i].shape = (numRow, numCol)
            newlist.append(np.matrix(list[i]))
        return newlist


    def convertArrayToMatrix(arr, numRow, numCol):
        arr.shape = (numRow, numCol)
        return np.matrix(arr)


    # ================== LOAD DATA =========================
    data = np.load(data_path)

    K = int(data['datasize'][0])
    R = int(data['datasize'][1])



    f = convertToListOfMatrices(data['f'], K, R, 1)
    g = convertToListOfMatrices(data['g'], K, R, 1)
    s = data['s']
    eta = convertArrayToMatrix(data['eta'], R, 1)
    sigmaR = float(data['sigma'][0])
    sigmaD = float(data['sigma'][1])
    p = data['p']
    D =  data['D'].real# convertToListOfMatrices(data['D'], K, R, R)
    D_Matrix = convertArrayToMatrix(data['D_Matrix'], R, R)
    Q = data['Q'] # convertToListOfMatrices(data['Q'], K, R, R)
    A = convertToListOfMatrices(data['A'], K, R, R)
    gamma = data['gamma']
    # =========================================================
    print("running test at:", path)
    print("K = ", K, "R = ", R, "gammak = ", gamma[0], "tau = ", tau__)


    # =================== INTRODUCE NEW NOTATION =====================
    t0 = time()
    # make a double size matrix of this form
    # [Re(input_mat)   -Im(input_mat)
    #  Im(input_mat     Re(input_mat)]
    def make_hat_matrix(Matrix):
        Mat_hat = np.vstack((np.hstack((Matrix.real, -Matrix.imag)),
                             np.hstack((Matrix.imag, Matrix.real))))
        return Mat_hat

    # make D_hat
    D_hat = make_hat_matrix(D_Matrix)

    # make list of Dk_hat
    list_Dk_hat = []
    for Dk in D:
        Dk_hat = make_hat_matrix(Dk)
        list_Dk_hat.append(Dk_hat)

    # make list of Ak_hat
    list_Ak_hat = []
    for Ak in A:
        Ak_hat = make_hat_matrix(Ak)
        list_Ak_hat.append(Ak_hat)

    # make list of Q_hat
    list_Qk_hat = []
    for Qk in Q:
        Qk_hat = make_hat_matrix(Qk)
        list_Qk_hat.append(Qk_hat)

    def qcq(xl, tl):
        x = cp.Variable(shape=(2 * R, 1))
        t = cp.Variable()
        objective = cp.Minimize(cp.quad_form(x,D_hat) + tau*t)
        constraints = [0 <= t]
        for k in range(K):
            Dk_hat = list_Dk_hat[k]
            Ak_hat = list_Ak_hat[k]
            Qk_hat = list_Qk_hat[k]
            constraints.append(gamma[k] * (cp.quad_form(x,Qk_hat) + sigmaR * cp.quad_form(x,Dk_hat) + sigmaD) -
                                        2 * (Ak_hat * xl).T * x + 2 * (Ak_hat * xl).T * xl - xl.T * Ak_hat * xl <= t)

        prob = cp.Problem(objective, constraints)

        #The optimal objective value is returned by `prob.solve()`.

        result = prob.solve(solver=cp.ECOS)

        # The optimal value for x is stored in `x.value`.
        return np.matrix(x.value), np.matrix(t.value), objective.value


    def check_constraint_dual(x,t,xl,tl):
        for k in range(K):
            lhs = gamma[k] * (x.T*Qk_hat*x + sigmaR * x.T*Dk_hat*x + sigmaD)
            lhs += - 2 * (Ak_hat * xl).T * x + 2 * (Ak_hat * xl).T * xl - xl.T * Ak_hat *xl
            rhs = t
            if(lhs > rhs):
                return False, lhs - rhs

            return True,0

    def check_constraint_primal(x,t):
        Max_Violation = 0
        for k in range(K):
            lhs = gamma[k] * (x.T*Qk_hat*x + sigmaR * x.T*Dk_hat*x + sigmaD)
            lhs += - x.T*Ak_hat*x
            rhs = 0

            if(lhs > rhs):
                return False, lhs - rhs
                # Max_Violation = np.maximum(lhs - rhs, Max_Violation)

        return True, 0
        # if (Max_Violation == 0):
        #     return True, 0

        # return False, Max_Violation


    def test_qcp():
        xl = np.matrix(np.random.rand(2 * R, 1))
        tl = np.random.random()
        xl_1, tl_1, opt_val = qcq(xl, tl)

        print("x optimal: \n", xl_1)
        print("t optimal: \n", tl_1)
        print("optimal value: ", opt_val)
        const_sat, vio = check_constraint_dual(xl_1, tl_1, xl, tl)
        print("constraint dual satisfied : ", const_sat)
        print("constraint dual violation : ", vio)


    def dca(max_iters):
        xl = np.matrix(np.random.rand(2 * R, 1))
        tl = np.random.random()

        print("Solving ...")

        for i in range(max_iters):
            xl_1, tl_1, opt_val = qcq(xl,tl)

            curr_opt_val = xl_1.T * D_hat * xl_1

            if(debug):
                print("===========================================")
                print("x optimal: \n", xl_1)
                print("t optimal: \n", tl_1)
                print("dual optimal value: ", opt_val)

                const_sat, vio = check_constraint_dual(xl_1, tl_1, xl, tl)
                print("constraint dual satisfied at iter %d: "%(i), const_sat)
                print("constraint dual violation at iter %d: " % (i), vio)

                const_sat, vio = check_constraint_primal(xl_1, tl_1)
                print("constraint primal satisfied at iter %d: " % (i), const_sat)
                print("constraint primal violation at iter %d: " % (i), vio)


                print("primal objective value: ", curr_opt_val)

            ul = np.vstack((xl, tl))
            ul_1 = np.vstack((xl_1, tl_1))
            equation = np.linalg.norm(ul_1 - ul) / (1+np.linalg.norm(ul))

            prev_opt_val = xl.T * D_hat * xl

            if (debug):

                print("equation: ", equation)

            if(equation < tol or abs(prev_opt_val - curr_opt_val)<opt_val_tol):
                return xl_1,tl_1

            xl,tl = xl_1, tl_1

        return xl,tl

    def checkOriginConstraint(i, w):
        Ak = A[i]
        Qk = Q[i]
        Dk = D[i]
        ulhs = w.H * Ak * w
        llhs = w.H * Qk * w + sigmaR * w.H * Dk * w + sigmaD

        lhs = 1.0 * ulhs.flat[0] / llhs.flat[0]
        rhs = gamma[i]

        # print("LHS: ", lhs)
        # print("RHS: ", rhs)
        # print("-----LHS - RHS =                                  ", lhs - rhs)
        if (np.real(lhs) >= np.real(rhs)):
            return 0
        return -np.real(lhs - rhs)

    # get the most serious violation
    def getMSV(w):
        MCV = -1
        for i in range(K):
            MCV = np.maximum(MCV, checkOriginConstraint(i, w))
        return MCV


    x_opt, t_opt = dca(max_iters)

    w = np.matrix(x_opt[0:20]) + 1j * np.matrix(x_opt[20:40])

    # print x_opt[0:20]
    primal_optimal_value =x_opt.T*D_hat*x_opt

    # const_sat, violation =  check_constraint_primal(x_opt, t_opt)
    # print("Check constraint on optimal x and t ?", const_sat)
    # print("Constraint violation on optimal x and t: ", violation)
    # print ("Is satified constraints: ", violation <= 0.000001)
    # print ("Is satified constraints: ", violation)
    t2 = time()
    print("Solving time: ", (t2-t0))
    print("Optimal value:", primal_optimal_value.flat[0])
    print("MSV:          ", getMSV(w))
    XYZ['time'].append(t2 - t0)
    XYZ['value'].append(primal_optimal_value.flat[0])
    XYZ['violation'].append(getMSV(w))

TAU = []
TAU[0:19] = [6000] * 19
TAU[19:22] = [1000] * 3
TAU[22:23] = [100]
TAU[23:27] = [6000] * 4
TAU[27:28] = [100]
TAU[28:29] = [6000]
TAU[29:32] = [100] * 3
TAU[32:43] = [6000] * 11
TAU[43:44] = [100]
TAU[44:51] = [6000] * 7
TAU[51:60] = [300] * 9
TAU[60:62] = [100] * 2
TAU[62:101] = [300] * 39

for i in range(2, 101, 1):
    path = 'Newfolder/autogeneratingdata_' + str(i) + '.npz'
    DCA(path, TAU[i])

np.savez("DCA_ECOS_Result", TIME=XYZ['time'], VALUE=XYZ['value'], VIOLATION=XYZ['violation'])
