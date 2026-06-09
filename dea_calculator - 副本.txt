import pandas as pd
import numpy as np
import sys
from scipy import optimize
from tqdm import tqdm, trange

# E_t(x_t,y_t), 即SBM效率值，返回值为CRS和VRS的元组
def Ecv_tt(cur, x_t, yg_t, yb_t=np.ndarray(0)):
    m, n = x_t.shape
    s1 = yg_t.shape[0]
    if (yb_t.size > 0):
        s2 = yb_t.shape[0]
        f = np.concatenate([np.zeros(n), -1/(m*x_t[:, cur]),
                            np.zeros(s1+s2), np.array([1])])
        Aeq1 = np.hstack([x_t,
                          np.identity(m),
                          np.zeros((m, s1+s2)),
                          -x_t[:, cur, None]])
        Aeq2 = np.hstack([yg_t,
                          np.zeros((s1, m)),
                          -np.identity(s1),
                          np.zeros((s1, s2)),
                          -yg_t[:, cur, None]])
        Aeq3 = np.hstack([yb_t,
                          np.zeros((s2, m)),
                          np.zeros((s2, s1)),
                          np.identity(s2),
                          -yb_t[:, cur, None]])
        Aeq4 = np.hstack([np.zeros(n),
                          np.zeros(m),
                          1/((s1+s2)*(yg_t[:, cur])),
                          1/((s1+s2)*(yb_t[:, cur])),
                          np.array([1])]).reshape(1, -1)
        # 规模报酬可变约束
        Aeq5 = np.hstack([np.ones(n),
                          np.zeros((m+s1+s2)),
                          np.array([-1])]).reshape(1, -1)
        Aeqvrs = np.vstack([Aeq1, Aeq2, Aeq3, Aeq4, Aeq5])
        beqvrs = np.concatenate(
            [np.zeros(m+s1+s2), np.array([1]), np.array([0])])
        Aeqcrs = np.vstack([Aeq1, Aeq2, Aeq3, Aeq4])
        beqcrs = np.concatenate([np.zeros(m+s1+s2), np.array([1])])
        bounds = tuple([(0, None) for t in range(n+s1+s2+m+1)])
        rescrs = optimize.linprog(c=f, A_eq=Aeqcrs, b_eq=beqcrs, bounds=bounds)
        resvrs = optimize.linprog(c=f, A_eq=Aeqvrs, b_eq=beqvrs, bounds=bounds)
    else:
        f = np.concatenate([np.zeros(n), -1/(m*x_t[:, cur]),
                            np.zeros(s1), np.array([1])])
        Aeq1 = np.hstack([x_t,
                          np.identity(m),
                          np.zeros((m, s1)),
                          -x_t[:, cur, None]])
        Aeq2 = np.hstack([yg_t,
                          np.zeros((s1, m)),
                          -np.identity(s1),
                          -yg_t[:, cur, None]])
        Aeq4 = np.hstack([np.zeros(n),
                          np.zeros(m),
                          1/((s1)*(yg_t[:, cur])),
                          np.array([1])]).reshape(1, -1)
        # 规模报酬可变约束
        Aeq5 = np.hstack([np.ones(n),
                          np.zeros((m+s1)),
                          np.array([-1])]).reshape(1, -1)
        Aeqvrs = np.vstack([Aeq1, Aeq2, Aeq4, Aeq5])
        beqvrs = np.concatenate([np.zeros(m+s1), np.array([1]), np.array([0])])
        Aeqcrs = np.vstack([Aeq1, Aeq2, Aeq4])
        beqcrs = np.concatenate([np.zeros(m+s1), np.array([1])])
        bounds = tuple([(0, None) for t in range(n+s1+m+1)])
        resvrs = optimize.linprog(c=f, A_eq=Aeqvrs, b_eq=beqvrs, bounds=bounds)
        rescrs = optimize.linprog(c=f, A_eq=Aeqcrs, b_eq=beqcrs, bounds=bounds)
    return (rescrs, resvrs)

# E_t(x_t+1,y_t+1), 返回值为CRS和VRS的元组
def Ecv_tt1(cur, x_t, x_t1, yg_t, yg_t1, yb_t=np.ndarray(0), yb_t1=np.ndarray(0)):
    m, n = x_t.shape
    s1 = yg_t.shape[0]
    if (yb_t.size > 0):
        s2 = yb_t.shape[0]
        f_tt1 = np.concatenate([np.zeros(n), -1/(m*x_t1[:, cur]),
                                np.zeros(s1+s2), np.array([1])])
        Aeq1_tt1 = np.hstack([x_t,
                              np.identity(m),
                              np.zeros((m, s1+s2)),
                              -x_t1[:, cur, None]])
        Aeq2_tt1 = np.hstack([yg_t,
                              np.zeros((s1, m)),
                              -np.identity(s1),
                              np.zeros((s1, s2)),
                              -yg_t1[:, cur, None]])
        Aeq3_tt1 = np.hstack([yb_t,
                              np.zeros((s2, m)),
                              np.zeros((s2, s1)),
                              np.identity(s2),
                              -yb_t1[:, cur, None]])
        Aeq4_tt1 = np.hstack([np.zeros(n),
                              np.zeros(m),
                              1/((s1+s2)*(yg_t1[:, cur])),
                              1/((s1+s2)*(yb_t1[:, cur])),
                              np.array([1])]).reshape(1, -1)
        # 规模报酬可变约束
        Aeq5_tt1 = np.hstack([np.ones(n),
                              np.zeros((m+s1+s2)),
                              np.array([-1])]).reshape(1, -1)
        Aeqvrs_tt1 = np.vstack(
            [Aeq1_tt1, Aeq2_tt1, Aeq3_tt1, Aeq4_tt1, Aeq5_tt1])
        beqvrs_tt1 = np.concatenate(
            [np.zeros(m+s1+s2), np.array([1]), np.array([0])])

        Aeqcrs_tt1 = np.vstack([Aeq1_tt1, Aeq2_tt1, Aeq3_tt1, Aeq4_tt1])
        beqcrs_tt1 = np.concatenate([np.zeros(m+s1+s2), np.array([1])])
        bounds = tuple([(0, None) for t in range(n+s1+s2+m+1)])
        rescrs_tt1 = optimize.linprog(
            c=f_tt1, A_eq=Aeqcrs_tt1, b_eq=beqcrs_tt1, bounds=bounds)
        resvrs_tt1 = optimize.linprog(
            c=f_tt1, A_eq=Aeqvrs_tt1, b_eq=beqvrs_tt1, bounds=bounds)
    else:
        f_tt1 = np.concatenate([np.zeros(n), -1/(m*x_t1[:, cur]),
                                np.zeros(s1), np.array([1])])
        Aeq1_tt1 = np.hstack([x_t,
                              np.identity(m),
                              np.zeros((m, s1)),
                              -x_t1[:, cur, None]])
        Aeq2_tt1 = np.hstack([yg_t,
                              np.zeros((s1, m)),
                              -np.identity(s1),
                              -yg_t1[:, cur, None]])
        Aeq4_tt1 = np.hstack([np.zeros(n),
                              np.zeros(m),
                              1/((s1)*(yg_t1[:, cur])),
                              np.array([1])]).reshape(1, -1)
        # 规模报酬可变约束
        Aeq5_tt1 = np.hstack([np.ones(n),
                              np.zeros((m+s1)),
                              np.array([-1])]).reshape(1, -1)
        Aeqvrs_tt1 = np.vstack([Aeq1_tt1, Aeq2_tt1, Aeq4_tt1, Aeq5_tt1])
        beqvrs_tt1 = np.concatenate(
            [np.zeros(m+s1), np.array([1]), np.array([0])])

        Aeqcrs_tt1 = np.vstack([Aeq1_tt1, Aeq2_tt1, Aeq4_tt1])
        beqcrs_tt1 = np.concatenate([np.zeros(m+s1), np.array([1])])
        bounds = tuple([(0, None) for t in range(n+s1+m+1)])
        resvrs_tt1 = optimize.linprog(
            c=f_tt1, A_eq=Aeqvrs_tt1, b_eq=beqvrs_tt1, bounds=bounds)
        rescrs_tt1 = optimize.linprog(
            c=f_tt1, A_eq=Aeqcrs_tt1, b_eq=beqcrs_tt1, bounds=bounds)

    return (rescrs_tt1, resvrs_tt1)

# E_t+1(x_t,y_t), 返回值为CRS和VRS的元组
def Ecv_t1t(cur, x_t, x_t1, yg_t, yg_t1, yb_t=np.ndarray(0), yb_t1=np.ndarray(0)):
    m, n = x_t.shape
    s1 = yg_t.shape[0]
    if (yb_t.size > 0):
        s2 = yb_t.shape[0]
        f_t1t = np.concatenate([np.zeros(n), -1/(m*x_t[:, cur]),
                                np.zeros(s1+s2), np.array([1])])
        Aeq1_t1t = np.hstack([x_t1,
                              np.identity(m),
                              np.zeros((m, s1+s2)),
                              -x_t[:, cur, None]])
        Aeq2_t1t = np.hstack([yg_t1,
                              np.zeros((s1, m)),
                              -np.identity(s1),
                              np.zeros((s1, s2)),
                              -yg_t[:, cur, None]])
        Aeq3_t1t = np.hstack([yb_t1,
                              np.zeros((s2, m)),
                              np.zeros((s2, s1)),
                              np.identity(s2),
                              -yb_t[:, cur, None]])
        Aeq4_t1t = np.hstack([np.zeros(n),
                              np.zeros(m),
                              1/((s1+s2)*(yg_t[:, cur])),
                              1/((s1+s2)*(yb_t[:, cur])),
                              np.array([1])]).reshape(1, -1)
        Aeq5_t1t = np.hstack([np.ones(n),
                              np.zeros((m+s1+s2)),
                              np.array([-1])]).reshape(1, -1)
        Aeqvrs_t1t = np.vstack(
            [Aeq1_t1t, Aeq2_t1t, Aeq3_t1t, Aeq4_t1t, Aeq5_t1t])
        beqvrs_t1t = np.concatenate(
            [np.zeros(m+s1+s2), np.array([1]), np.array([0])])

        Aeqcrs_t1t = np.vstack([Aeq1_t1t, Aeq2_t1t, Aeq3_t1t, Aeq4_t1t])
        beqcrs_t1t = np.concatenate([np.zeros(m+s1+s2), np.array([1])])
        bounds = tuple([(0, None) for t in range(n+s1+s2+m+1)])
        rescrs_t1t = optimize.linprog(
            c=f_t1t, A_eq=Aeqcrs_t1t, b_eq=beqcrs_t1t, bounds=bounds)
        resvrs_t1t = optimize.linprog(
            c=f_t1t, A_eq=Aeqvrs_t1t, b_eq=beqvrs_t1t, bounds=bounds)
    else:
        f_t1t = np.concatenate([np.zeros(n), -1/(m*x_t[:, cur]),
                                np.zeros(s1), np.array([1])])
        Aeq1_t1t = np.hstack([x_t1,
                              np.identity(m),
                              np.zeros((m, s1)),
                              -x_t[:, cur, None]])
        Aeq2_t1t = np.hstack([yg_t1,
                              np.zeros((s1, m)),
                              -np.identity(s1),
                              -yg_t[:, cur, None]])
        Aeq4_t1t = np.hstack([np.zeros(n),
                              np.zeros(m),
                              1/((s1)*(yg_t[:, cur])),
                              np.array([1])]).reshape(1, -1)
        # 规模报酬可变约束
        Aeq5_t1t = np.hstack([np.ones(n),
                              np.zeros((m+s1)),
                              np.array([-1])]).reshape(1, -1)
        Aeqvrs_t1t = np.vstack([Aeq1_t1t, Aeq2_t1t, Aeq4_t1t, Aeq5_t1t])
        beqvrs_t1t = np.concatenate(
            [np.zeros(m+s1), np.array([1]), np.array([0])])

        Aeqcrs_t1t = np.vstack([Aeq1_t1t, Aeq2_t1t, Aeq4_t1t])
        beqcrs_t1t = np.concatenate([np.zeros(m+s1), np.array([1])])
        bounds = tuple([(0, None) for t in range(n+s1+m+1)])
        resvrs_t1t = optimize.linprog(
            c=f_t1t, A_eq=Aeqvrs_t1t, b_eq=beqvrs_t1t, bounds=bounds)
        rescrs_t1t = optimize.linprog(
            c=f_t1t, A_eq=Aeqcrs_t1t, b_eq=beqcrs_t1t, bounds=bounds)
    return (rescrs_t1t, resvrs_t1t)

# 超效率的E_t(x_t,y_t), 即超效率SBM效率值，返回值为CRS和VRS的元组
def SupEcv_tt(cur, x_t, yg_t, yb_t=np.ndarray(0)):
    m, n = x_t.shape
    s1 = yg_t.shape[0]
    if (yb_t.size > 0):
        s2 = yb_t.shape[0]
        f = np.concatenate([np.zeros(n), 1/(m*x_t[:, cur]),
                            np.zeros(s1+s2), np.array([1])])
        Aeq1 = np.hstack([np.zeros(n),
                          np.zeros(m),
                          -1/((s1+s2)*(yg_t[:, cur])),
                          -1/((s1+s2)*(yb_t[:, cur])),
                          np.array([1])]).reshape(1, -1)
        # 规模报酬可变约束
        Aeq2 = np.hstack([np.ones(n),
                          np.zeros((m+s1+s2)),
                          np.array([-1])]).reshape(1, -1)  # 规模报酬可变增加的约束
        Aeqvrs = np.vstack([Aeq1, Aeq2])  # 合并等式约束矩阵
        Aeqvrs[:, cur] = 0  # 超效率是把计算点排除在基准点之外，即将正在计算列的系数置为0
        beqvrs = np.concatenate(
            [np.array([1]), np.array([0])])  # 增加VRS约束方程右边的0

        Aeqcrs = Aeq1
        beqcrs = np.array([1])
        Aub1 = np.hstack([x_t,
                          -np.identity(m),
                          np.zeros((m, s1+s2)),
                          -x_t[:, cur, None]])

        Aub2 = np.hstack([-yg_t,
                          np.zeros((s1, m)),
                          -np.identity(s1),
                          np.zeros((s1, s2)),
                          yg_t[:, cur, None]])

        Aub3 = np.hstack([yb_t,
                          np.zeros((s2, m)),
                          np.zeros((s2, s1)),
                          -np.identity(s2),
                          -yb_t[:, cur, None]])
        Aub = np.vstack([Aub1, Aub2, Aub3])
        bub = np.zeros(m+s1+s2)
        Aub[:, cur] = 0  # 超效率是把计算点排除在基准点之外，即将正在计算列的系数置为0
        bounds = tuple([(0, None) for i in range(n+s1+s2+m+1)])
        resvrs = optimize.linprog(
            c=f, A_ub=Aub, b_ub=bub, A_eq=Aeqvrs, b_eq=beqvrs, bounds=bounds)
        rescrs = optimize.linprog(
            c=f, A_ub=Aub, b_ub=bub, A_eq=Aeqcrs, b_eq=beqcrs, bounds=bounds)
    else:
        f = np.concatenate([np.zeros(n), 1/(m*x_t[:, cur]),
                            np.zeros(s1), np.array([1])])
        Aeq1 = np.hstack([np.zeros(n),
                          np.zeros(m),
                          -1/((s1)*(yg_t[:, cur])),
                          np.array([1])]).reshape(1, -1)
        # 规模报酬可变约束
        Aeq2 = np.hstack([np.ones(n),
                          np.zeros((m+s1)),
                          np.array([-1])]).reshape(1, -1)  # 规模报酬可变增加的约束
        Aeqvrs = np.vstack([Aeq1, Aeq2])  # 合并等式约束矩阵
        Aeqvrs[:, cur] = 0  # 超效率是把计算点排除在基准点之外，即将正在计算列的系数置为0
        beqvrs = np.concatenate(
            [np.array([1]), np.array([0])])  # 增加VRS约束方程右边的0

        Aeqcrs = Aeq1
        beqcrs = np.array([1])
        Aub1 = np.hstack([x_t,
                          -np.identity(m),
                          np.zeros((m, s1)),
                          -x_t[:, cur, None]])
        Aub2 = np.hstack([-yg_t,
                          np.zeros((s1, m)),
                          -np.identity(s1),
                          yg_t[:, cur, None]])
        Aub = np.vstack([Aub1, Aub2])
        bub = np.zeros(m+s1)
        Aub[:, cur] = 0  # 超效率是把计算点排除在基准点之外，即将正在计算列的系数置为0
        bounds = tuple([(0, None) for i in range(n+s1+m+1)])
        resvrs = optimize.linprog(
            c=f, A_ub=Aub, b_ub=bub, A_eq=Aeqvrs, b_eq=beqvrs, bounds=bounds)
        rescrs = optimize.linprog(
            c=f, A_ub=Aub, b_ub=bub, A_eq=Aeqcrs, b_eq=beqcrs, bounds=bounds)
    return (rescrs, resvrs)

# 超效率的E_t(x_t+1,y_t+1), 返回值为CRS和VRS的元组
def SupEcv_tt1(cur, x_t, x_t1, yg_t,  yg_t1, yb_t=np.ndarray(0), yb_t1=np.ndarray(0)):
    m, n = x_t.shape
    s1 = yg_t.shape[0]
    if (yb_t.size > 0):
        s2 = yb_t.shape[0]
        f_tt1 = np.concatenate([np.zeros(n), 1/(m*x_t1[:, cur]),
                                np.zeros(s1+s2), np.array([1])])
        Aeq1_tt1 = np.hstack([np.zeros(n),
                              np.zeros(m),
                              -1/((s1+s2)*(yg_t1[:, cur])),
                              -1/((s1+s2)*(yb_t1[:, cur])),
                              np.array([1])]).reshape(1, -1)

        Aeq2_tt1 = np.hstack([np.ones(n),
                              np.zeros((m+s1+s2)),
                              np.array([-1])]).reshape(1, -1)  # 规模报酬可变增加的约束
        Aeqvrs_tt1 = np.vstack([Aeq1_tt1, Aeq2_tt1])  # 合并等式约束矩阵
        # 计算tt1或t1t时，基准点本来就不包含计算点，所以不能考虑下面这条条件
        # Aeqvrs_tt1[:, cur] = 0  # 超效率是把计算点排除在基准点之外，即将正在计算列的系数置为0
        beqvrs_tt1 = np.concatenate(
            [np.array([1]), np.array([0])])  # 增加VRS约束方程右边的0

        Aeqcrs_tt1 = Aeq1_tt1
        beqcrs_tt1 = np.array([1])
        Aub1_tt1 = np.hstack([x_t,
                              -np.identity(m),
                              np.zeros((m, s1+s2)),
                              -x_t1[:, cur, None]])

        Aub2_tt1 = np.hstack([-yg_t,
                              np.zeros((s1, m)),
                              -np.identity(s1),
                              np.zeros((s1, s2)),
                              yg_t1[:, cur, None]])

        Aub3_tt1 = np.hstack([yb_t,
                              np.zeros((s2, m)),
                              np.zeros((s2, s1)),
                              -np.identity(s2),
                              -yb_t1[:, cur, None]])
        Aub_tt1 = np.vstack([Aub1_tt1, Aub2_tt1, Aub3_tt1])
        bub_tt1 = np.zeros(m+s1+s2)
        # 计算tt1或t1t时，基准点本来就不包含计算点，所以不能考虑下面这条条件
        # Aub_tt1[:, cur] = 0
        bounds = tuple([(0, None) for i in range(n+s1+s2+m+1)])
        resvrs_tt1 = optimize.linprog(
            c=f_tt1, A_ub=Aub_tt1, b_ub=bub_tt1, A_eq=Aeqvrs_tt1, b_eq=beqvrs_tt1, bounds=bounds)
        rescrs_tt1 = optimize.linprog(
            c=f_tt1, A_ub=Aub_tt1, b_ub=bub_tt1, A_eq=Aeqcrs_tt1, b_eq=beqcrs_tt1, bounds=bounds)
    else:
        f_tt1 = np.concatenate([np.zeros(n), 1/(m*x_t1[:, cur]),
                                np.zeros(s1), np.array([1])])
        Aeq1_tt1 = np.hstack([np.zeros(n),
                              np.zeros(m),
                              -1/((s1)*(yg_t1[:, cur])),
                              np.array([1])]).reshape(1, -1)

        Aeq2_tt1 = np.hstack([np.ones(n),
                              np.zeros((m+s1)),
                              np.array([-1])]).reshape(1, -1)  # 规模报酬可变增加的约束
        Aeqvrs_tt1 = np.vstack([Aeq1_tt1, Aeq2_tt1])  # 合并等式约束矩阵
        # 计算tt1或t1t时，基准点本来就不包含计算点，所以不能考虑下面这条条件
        # Aeqvrs_tt1[:, cur] = 0  # 将正在计算列的系数置为0，对应约束条件中的求和不包括i
        beqvrs_tt1 = np.concatenate(
            [np.array([1]), np.array([0])])  # 增加VRS约束方程右边的0

        Aeqcrs_tt1 = Aeq1_tt1
        beqcrs_tt1 = np.array([1])
        Aub1_tt1 = np.hstack([x_t,
                              -np.identity(m),
                              np.zeros((m, s1)),
                              -x_t1[:, cur, None]])
        Aub2_tt1 = np.hstack([-yg_t,
                              np.zeros((s1, m)),
                              -np.identity(s1),
                              yg_t1[:, cur, None]])
        Aub_tt1 = np.vstack([Aub1_tt1, Aub2_tt1])
        bub_tt1 = np.zeros(m+s1)
        # 计算tt1或t1t时，基准点本来就不包含计算点，所以不能考虑下面这条条件
        # Aub_tt1[:, cur] = 0
        bounds = tuple([(0, None) for i in range(n+s1+m+1)])
        resvrs_tt1 = optimize.linprog(
            c=f_tt1, A_ub=Aub_tt1, b_ub=bub_tt1, A_eq=Aeqvrs_tt1, b_eq=beqvrs_tt1, bounds=bounds)
        rescrs_tt1 = optimize.linprog(
            c=f_tt1, A_ub=Aub_tt1, b_ub=bub_tt1, A_eq=Aeqcrs_tt1, b_eq=beqcrs_tt1, bounds=bounds)

    return (rescrs_tt1, resvrs_tt1)

# 超效率的E_t+1(x_t,y_t), 返回值为CRS和VRS的元组
def SupEcv_t1t(cur, x_t, x_t1, yg_t, yg_t1, yb_t=np.ndarray(0), yb_t1=np.ndarray(0)):
    m, n = x_t.shape
    s1 = yg_t.shape[0]
    if (yb_t.size > 0):
        s2 = yb_t.shape[0]
        f_t1t = np.concatenate([np.zeros(n), 1/(m*x_t[:, cur]),
                                np.zeros(s1+s2), np.array([1])])
        Aeq1_t1t = np.hstack([np.zeros(n),
                              np.zeros(m),
                              -1/((s1+s2)*(yg_t[:, cur])),
                              -1/((s1+s2)*(yb_t[:, cur])),
                              np.array([1])]).reshape(1, -1)

        Aeq2_t1t = np.hstack([np.ones(n),
                              np.zeros((m+s1+s2)),
                              np.array([-1])]).reshape(1, -1)  # 规模报酬可变增加的约束
        Aeqvrs_t1t = np.vstack([Aeq1_t1t, Aeq2_t1t])  # 合并等式约束矩阵
        # 计算tt1或t1t时，基准点本来就不包含计算点，所以不能考虑下面这条条件
        # Aeqvrs_t1t[:, cur] = 0  # 将正在计算列的系数置为0，对应约束条件中的求和不包括i
        beqvrs_t1t = np.concatenate(
            [np.array([1]), np.array([0])])  # 增加VRS约束方程右边的0

        Aeqcrs_t1t = Aeq1_t1t
        beqcrs_t1t = np.array([1])
        Aub1_t1t = np.hstack([x_t1,
                              -np.identity(m),
                              np.zeros((m, s1+s2)),
                              -x_t[:, cur, None]])

        Aub2_t1t = np.hstack([-yg_t1,
                              np.zeros((s1, m)),
                              -np.identity(s1),
                              np.zeros((s1, s2)),
                              yg_t[:, cur, None]])

        Aub3_t1t = np.hstack([yb_t1,
                              np.zeros((s2, m)),
                              np.zeros((s2, s1)),
                              -np.identity(s2),
                              -yb_t[:, cur, None]])
        Aub_t1t = np.vstack([Aub1_t1t, Aub2_t1t, Aub3_t1t])
        bub_t1t = np.zeros(m+s1+s2)
        # 计算tt1或t1t时，基准点本来就不包含计算点，所以不能考虑下面这条条件
        # Aub_t1t[:, cur] = 0
        bounds = tuple([(0, None) for i in range(n+s1+s2+m+1)])
        resvrs_t1t = optimize.linprog(
            c=f_t1t, A_ub=Aub_t1t, b_ub=bub_t1t, A_eq=Aeqvrs_t1t, b_eq=beqvrs_t1t, bounds=bounds)
        rescrs_t1t = optimize.linprog(
            c=f_t1t, A_ub=Aub_t1t, b_ub=bub_t1t, A_eq=Aeqcrs_t1t, b_eq=beqcrs_t1t, bounds=bounds)
    else:
        f_t1t = np.concatenate([np.zeros(n), 1/(m*x_t[:, cur]),
                                np.zeros(s1), np.array([1])])
        Aeq1_t1t = np.hstack([np.zeros(n),
                              np.zeros(m),
                              -1/((s1)*(yg_t[:, cur])),
                              np.array([1])]).reshape(1, -1)
        # 规模报酬可变增加的约束
        Aeq2_t1t = np.hstack([np.ones(n),
                              np.zeros((m+s1)),
                              np.array([-1])]).reshape(1, -1)
        Aeqvrs_t1t = np.vstack([Aeq1_t1t, Aeq2_t1t])  # 合并等式约束矩阵
        # 计算tt1或t1t时，基准点本来就不包含计算点，所以不能考虑下面这条条件
        # Aeqvrs_t1t[:, cur] = 0  # 将正在计算列的系数置为0，对应约束条件中的求和不包括i
        beqvrs_t1t = np.concatenate(
            [np.array([1]), np.array([0])])  # 增加VRS约束方程右边的0

        Aeqcrs_t1t = Aeq1_t1t
        beqcrs_t1t = np.array([1])
        Aub1_t1t = np.hstack([x_t1,
                              -np.identity(m),
                              np.zeros((m, s1)),
                              -x_t[:, cur, None]])
        Aub2_t1t = np.hstack([-yg_t1,
                              np.zeros((s1, m)),
                              -np.identity(s1),
                              yg_t[:, cur, None]])
        Aub_t1t = np.vstack([Aub1_t1t, Aub2_t1t])
        bub_t1t = np.zeros(m+s1)
        # 计算tt1或t1t时，基准点本来就不包含计算点，所以不能考虑下面这条条件
        # Aub_t1t[:, cur] = 0
        bounds = tuple([(0, None) for i in range(n+s1+m+1)])
        resvrs_t1t = optimize.linprog(
            c=f_t1t, A_ub=Aub_t1t, b_ub=bub_t1t, A_eq=Aeqvrs_t1t, b_eq=beqvrs_t1t, bounds=bounds)
        rescrs_t1t = optimize.linprog(
            c=f_t1t, A_ub=Aub_t1t, b_ub=bub_t1t, A_eq=Aeqcrs_t1t, b_eq=beqcrs_t1t, bounds=bounds)
    return (rescrs_t1t, resvrs_t1t)

# 全局的E_g(x_t,y_t), 返回值为CRS和VRS的元组
def Ec_g(cur_g, x_t, yg_t, x_g, yg_g, yb_t=np.ndarray(0), yb_g=np.ndarray(0)):
    m, n = x_t.shape
    mg, ng = x_g.shape
    s1 = yg_t.shape[0]
    if (yb_t.size > 0):  # 包含非期望产出
        s2 = yb_t.shape[0]
        f = np.concatenate([np.zeros(ng), -1/(m*x_g[:, cur_g]),
                            np.zeros(s1+s2), np.array([1])])
        Aeq1 = np.hstack([x_g,
                          np.identity(m),
                          np.zeros((m, s1+s2)),
                          -x_g[:, cur_g, None]])
        Aeq2 = np.hstack([yg_g,
                          np.zeros((s1, m)),
                          -np.identity(s1),
                          np.zeros((s1, s2)),
                          -yg_g[:, cur_g, None]])
        Aeq3 = np.hstack([yb_g,
                          np.zeros((s2, m)),
                          np.zeros((s2, s1)),
                          np.identity(s2),
                          -yb_g[:, cur_g, None]])
        Aeq4 = np.hstack([np.zeros(ng),
                          np.zeros(m),
                          1/((s1+s2)*(yg_g[:, cur_g])),
                          1/((s1+s2)*(yb_g[:, cur_g])),
                          np.array([1])]).reshape(1, -1)
        # 规模报酬可变约束
        Aeq5 = np.hstack([np.ones(ng),
                          np.zeros((m+s1+s2)),
                          np.array([-1])]).reshape(1, -1)
        Aeqvrs = np.vstack([Aeq1, Aeq2, Aeq3, Aeq4, Aeq5])
        beqvrs = np.concatenate(
            [np.zeros(m+s1+s2), np.array([1]), np.array([0])])
        Aeqcrs = np.vstack([Aeq1, Aeq2, Aeq3, Aeq4])
        beqcrs = np.concatenate([np.zeros(m+s1+s2), np.array([1])])
        bounds = tuple([(0, None) for t in range(ng+s1+s2+m+1)])
        rescrs = optimize.linprog(c=f, A_eq=Aeqcrs, b_eq=beqcrs, bounds=bounds)
        resvrs = optimize.linprog(c=f, A_eq=Aeqvrs, b_eq=beqvrs, bounds=bounds)
    else:
        f = np.concatenate([np.zeros(ng), -1/(m*x_g[:, cur_g]),
                            np.zeros(s1), np.array([1])])
        Aeq1 = np.hstack([x_g,
                          np.identity(m),
                          np.zeros((m, s1)),
                          -x_g[:, cur_g, None]])
        Aeq2 = np.hstack([yg_g,
                          np.zeros((s1, m)),
                          -np.identity(s1),
                          -yg_g[:, cur_g, None]])
        Aeq4 = np.hstack([np.zeros(ng),
                          np.zeros(m),
                          1/((s1)*(yg_g[:, cur_g])),
                          np.array([1])]).reshape(1, -1)
        # 规模报酬可变约束
        Aeq5 = np.hstack([np.ones(ng),
                          np.zeros((m+s1)),
                          np.array([-1])]).reshape(1, -1)
        Aeqvrs = np.vstack([Aeq1, Aeq2, Aeq4, Aeq5])
        beqvrs = np.concatenate([np.zeros(m+s1), np.array([1]), np.array([0])])
        Aeqcrs = np.vstack([Aeq1, Aeq2, Aeq4])
        beqcrs = np.concatenate([np.zeros(m+s1), np.array([1])])
        bounds = tuple([(0, None) for t in range(ng+s1+m+1)])
        resvrs = optimize.linprog(c=f, A_eq=Aeqvrs, b_eq=beqvrs, bounds=bounds)
        rescrs = optimize.linprog(c=f, A_eq=Aeqcrs, b_eq=beqcrs, bounds=bounds)
    return (rescrs, resvrs)

# 超效率的全局的E_g(x_t,y_t), 返回值为CRS和VRS的元组
def SupEc_g(cur_g, x_t, yg_t, x_g, yg_g, yb_t=np.ndarray(0), yb_g=np.ndarray(0)):
    m, n = x_t.shape
    mg, ng = x_g.shape
    s1 = yg_t.shape[0]
    if (yb_t.size > 0):
        s2 = yb_t.shape[0]
        f = np.concatenate([np.zeros(ng), 1/(m*x_g[:, cur_g]),
                            np.zeros(s1+s2), np.array([1])])
        Aeq1 = np.hstack([np.zeros(ng),
                          np.zeros(m),
                          -1/((s1+s2)*(yg_g[:, cur_g])),
                          -1/((s1+s2)*(yb_g[:, cur_g])),
                          np.array([1])]).reshape(1, -1)
        # 规模报酬可变约束
        Aeq2 = np.hstack([np.ones(ng),
                          np.zeros((m+s1+s2)),
                          np.array([-1])]).reshape(1, -1)  # 规模报酬可变增加的约束
        Aeqvrs = np.vstack([Aeq1, Aeq2])  # 合并等式约束矩阵
        Aeqvrs[:, cur_g] = 0  # 超效率是把计算点排除在基准点之外，即将正在计算列的系数置为0
        beqvrs = np.concatenate(
            [np.array([1]), np.array([0])])  # 增加VRS约束方程右边的0

        Aeqcrs = Aeq1
        beqcrs = np.array([1])
        Aub1 = np.hstack([x_g,
                          -np.identity(m),
                          np.zeros((m, s1+s2)),
                          -x_g[:, cur_g, None]])

        Aub2 = np.hstack([-yg_g,
                          np.zeros((s1, m)),
                          -np.identity(s1),
                          np.zeros((s1, s2)),
                          yg_g[:, cur_g, None]])

        Aub3 = np.hstack([yb_g,
                          np.zeros((s2, m)),
                          np.zeros((s2, s1)),
                          -np.identity(s2),
                          -yb_g[:, cur_g, None]])
        Aub = np.vstack([Aub1, Aub2, Aub3])
        bub = np.zeros(m+s1+s2)
        Aub[:, cur_g] = 0  # 超效率是把计算点排除在基准点之外，即将正在计算列的系数置为0
        bounds = tuple([(0, None) for i in range(ng+s1+s2+m+1)])
        resvrs = optimize.linprog(
            c=f, A_ub=Aub, b_ub=bub, A_eq=Aeqvrs, b_eq=beqvrs, bounds=bounds)
        rescrs = optimize.linprog(
            c=f, A_ub=Aub, b_ub=bub, A_eq=Aeqcrs, b_eq=beqcrs, bounds=bounds)
    else:
        f = np.concatenate([np.zeros(ng), 1/(m*x_g[:, cur_g]),
                            np.zeros(s1), np.array([1])])
        Aeq1 = np.hstack([np.zeros(ng),
                          np.zeros(m),
                          -1/((s1)*(yg_g[:, cur_g])),
                          np.array([1])]).reshape(1, -1)
        # 规模报酬可变约束
        Aeq2 = np.hstack([np.ones(ng),
                          np.zeros((m+s1)),
                          np.array([-1])]).reshape(1, -1)  # 规模报酬可变增加的约束
        Aeqvrs = np.vstack([Aeq1, Aeq2])  # 合并等式约束矩阵
        Aeqvrs[:, cur_g] = 0  # 超效率是把计算点排除在基准点之外，即将正在计算列的系数置为0
        beqvrs = np.concatenate(
            [np.array([1]), np.array([0])])  # 增加VRS约束方程右边的0

        Aeqcrs = Aeq1
        beqcrs = np.array([1])
        Aub1 = np.hstack([x_g,
                          -np.identity(m),
                          np.zeros((m, s1)),
                          -x_g[:, cur_g, None]])
        Aub2 = np.hstack([-yg_g,
                          np.zeros((s1, m)),
                          -np.identity(s1),
                          yg_g[:, cur_g, None]])
        Aub = np.vstack([Aub1, Aub2])
        bub = np.zeros(m+s1)
        Aub[:, cur_g] = 0  # 超效率是把计算点排除在基准点之外，即将正在计算列的系数置为0
        bounds = tuple([(0, None) for i in range(ng+s1+m+1)])
        resvrs = optimize.linprog(
            c=f, A_ub=Aub, b_ub=bub, A_eq=Aeqvrs, b_eq=beqvrs, bounds=bounds)
        rescrs = optimize.linprog(
            c=f, A_ub=Aub, b_ub=bub, A_eq=Aeqcrs, b_eq=beqcrs, bounds=bounds)
    return (rescrs, resvrs)


# 定义一些参数
dmus = 30  # 有多少个决策单元
periods = 5  # 有多少个观察时期
nx = 3  # 投入变量个数
ny = 4  # 产出变量个数
nb = 1  # 非期望产出变量个数
undesirable = 1  # 1为存在非期望产出；0为不存在非期望产出
sup = 1  # 1为超效率;0为非超效率

df = pd.read_excel("t1.xlsx", index_col=0)
data = df.values
ec_tt,  ev_tt = [], []
ec_tt1, ev_tt1 = [], []
ec_t1t, ev_t1t = [], []
ec_gt, ev_gt = [], []

for t in tqdm(range(periods),desc='标准效率参数'):
    data_t = data[dmus*t:dmus*(t+1), :]  # 获取对应期的全部数据
    if (t != periods-1):  # 获取t+1期的所有数据
        data_t1 = data[dmus*(t+1):dmus*(t+2), :]
    dmuname = data_t[:, 0]  # 获取决策单元的名称
    year = data_t[0, 1]  # 获取当前期
    x_t = data_t[:, 2:nx+2]  # t期投入变量
    x_t = x_t.T
    x_t1 = data_t1[:, 2:nx+2]  # t+1期投入变量
    x_t1 = x_t1.T
    x_g = data[:, 2:nx+2]  # 全局投入变量
    x_g = x_g.T
    yg_t = data_t[:, nx+2:nx+ny+2]  # t期产出变量
    yg_t = yg_t.T
    yg_t1 = data_t1[:, nx+2:nx+ny+2]  # t+1期产出变量
    yg_t1 = yg_t1.T
    yg_g = data[:, nx+2:nx+ny+2]  # 全局产出变量
    yg_g = yg_g.T
    if undesirable == 1:
        yb_t = data_t[:, nx+ny+2:nx+ny+nb+2]  # t期非期望产出变量
        yb_t = yb_t.T
        yb_t1 = data_t1[:, nx+ny+2:nx+ny+nb+2]  # t+1期非期望产出变量
        yb_t1 = yb_t1.T
        yb_g = data[:, nx+ny+2:nx+ny+nb+2]  # 全局非期产出变量
        yb_g = yb_g.T
    for i in range(dmus):
        if (undesirable == 1):  # 含非期望产出
            # tt 标准SBM模型效率E_t(x_t,y_t,z_t)
            res = Ecv_tt(cur=i, x_t=x_t, yg_t=yg_t, yb_t=yb_t)
            ec_tt.append((dmuname[i], year, res[0].fun))
            ev_tt.append((dmuname[i], year, res[1].fun))

            # gt 全局SBM模型效率E_g(x_t,y_t,z_t)
            res = Ec_g(cur_g=i+t*dmus, x_t=x_t, yg_t=yg_t, x_g=x_g,
                       yg_g=yg_g, yb_t=yb_t, yb_g=yb_g)
            ec_gt.append((dmuname[i], year, res[0].fun))
            ev_gt.append((dmuname[i], year, res[1].fun))

            # 最后一期不用计算tt1和t1t
            if (t == periods-1):
                continue

            # tt1 相邻SBM模型效率E_t(x_t+1,y_t+1,z_t+1)
            res = Ecv_tt1(cur=i, x_t=x_t, x_t1=x_t1, yg_t=yg_t,
                          yg_t1=yg_t1, yb_t=yb_t, yb_t1=yb_t1)
            ec_tt1.append((dmuname[i], year, res[0].fun))
            ev_tt1.append((dmuname[i], year, res[1].fun))

            # t1t 相邻SBM模型效率E_t+1(x_t,y_t,z_t)
            res = Ecv_t1t(cur=i, x_t=x_t, x_t1=x_t1, yg_t=yg_t,
                          yg_t1=yg_t1, yb_t=yb_t, yb_t1=yb_t1)
            ec_t1t.append((dmuname[i], year, res[0].fun))
            ev_t1t.append((dmuname[i], year, res[1].fun))

        if (undesirable == 0):  # 不含非期望产出
            # tt 标准SBM模型效率E_t(x_t,y_t)
            res = Ecv_tt(cur=i, x_t=x_t, yg_t=yg_t)
            ec_tt.append((dmuname[i], year, res[0].fun))
            ev_tt.append((dmuname[i], year, res[1].fun))

            # gt 全局SBM模型效率E_g(x_t,y_t)
            res = Ec_g(cur_g=i+t*dmus, x_t=x_t, yg_t=yg_t, x_g=x_g, yg_g=yg_g)
            ec_gt.append((dmuname[i], year, res[0].fun))
            ev_gt.append((dmuname[i], year, res[1].fun))

            # 最后一期不用计算tt1和t1t
            if (t == periods-1):
                continue

            # tt1 相邻SBM模型效率E_t(x_t+1,y_t+1)
            res = Ecv_tt1(cur=i, x_t=x_t, x_t1=x_t1,  yg_t=yg_t, yg_t1=yg_t1)
            ec_tt1.append((dmuname[i], year, res[0].fun))
            ev_tt1.append((dmuname[i], year, res[1].fun))

            # t1t 相邻SBM模型效率E_t+1(x_t,y_t)
            res = Ecv_t1t(cur=i, x_t=x_t, x_t1=x_t1,  yg_t=yg_t, yg_t1=yg_t1)
            ec_t1t.append((dmuname[i], year, res[0].fun))
            ev_t1t.append((dmuname[i], year, res[1].fun))


# 超效率部分
for t in tqdm(range(periods),desc='超效率参数'):
    data_t = data[dmus*t:dmus*(t+1), :]  # 获取对应期的全部数据
    if (t != periods-1):  # 获取t+1期的所有数据
        data_t1 = data[dmus*(t+1):dmus*(t+2), :]
    dmuname = data_t[:, 0]  # 获取决策单元的名称
    year = data_t[0, 1]  # 获取当前期
    x_t = data_t[:, 2:nx+2]  # t期投入变量
    x_t = x_t.T
    x_g = data[:, 2:nx+2]  # 全局投入变量
    x_g = x_g.T
    x_t1 = data_t1[:, 2:nx+2]  # t+1期投入变量
    x_t1 = x_t1.T
    yg_t = data_t[:, nx+2:nx+ny+2]  # t期产出变量
    yg_t = yg_t.T
    yg_t1 = data_t1[:, nx+2:nx+ny+2]  # t+1期产出变量
    yg_t1 = yg_t1.T
    yg_g = data[:, nx+2:nx+ny+2]  # 全局产出变量
    yg_g = yg_g.T

    if undesirable == 1:
        yb_t = data_t[:, nx+ny+2:nx+ny+nb+2]  # t期非期望产出变量
        yb_t = yb_t.T
        yb_t1 = data_t1[:, nx+ny+2:nx+ny+nb+2]  # t+1期非期望产出变量
        yb_t1 = yb_t1.T
        yb_g = data[:, nx+ny+2:nx+ny+nb+2]  # 全局非期望产出变量
        yb_g = yb_g.T

    for i in range(dmus):
        if (undesirable == 1):
            # tt 标准super SBM模型效率E_t(x_t,y_t,z_t)
            res = SupEcv_tt(cur=i, x_t=x_t, yg_t=yg_t, yb_t=yb_t)
            # 非super无可行解时，用超效率模型解
            if (ec_tt[i+t*dmus][2] == None):
                ec_tt[i+t*dmus] = (dmuname[i], year, res[0].fun)
            if (ev_tt[i+t*dmus][2] == None):
                ev_tt[i+t*dmus] = (dmuname[i], year, res[1].fun)
            # 超效率参数下，保存超效率结果
            if (sup == 1):
                if (res[0].fun != None):  # super结果仍有不可行解的可能
                    if (res[0].fun > 1):
                        ec_tt[i+t*dmus] = (dmuname[i], year, res[0].fun)
                if (res[1].fun != None):  # super结果仍有不可行解的可能
                    if (res[1].fun > 1):
                        ev_tt[i+t*dmus] = (dmuname[i], year, res[1].fun)

            # gt 全局super SBM模型效率E_g(x_t,y_t,z_t)
            res = SupEc_g(cur_g=i+t*dmus, x_t=x_t, yg_t=yg_t,
                          x_g=x_g, yg_g=yg_g, yb_t=yb_t, yb_g=yb_g)
            # 超效率参数下，保存超效率结果
            if (sup == 1):
                if (res[0].fun != None):  # super结果仍有不可行解的可能
                    if (res[0].fun > 1):
                        ec_gt[i+t*dmus] = (dmuname[i], year, res[0].fun)
                if (res[1].fun != None):  # super结果仍有不可行解的可能
                    if (res[1].fun > 1):
                        ev_gt[i+t*dmus] = (dmuname[i], year, res[1].fun)

            # 最后一期不用计算tt1和t1t
            if (t == periods-1):
                continue

            # tt1 相邻super SBM模型效率E_t(x_t+1,y_t+1,z_t+1)
            res = SupEcv_tt1(cur=i, x_t=x_t, x_t1=x_t1, yg_t=yg_t,
                             yg_t1=yg_t1, yb_t=yb_t, yb_t1=yb_t1)
            # 非super无可行解时，用超效率模型解
            if (ec_tt1[i+t*dmus][2] == None):
                ec_tt1[i+t*dmus] = (dmuname[i], year, res[0].fun)
            if (ev_tt1[i+t*dmus][2] == None):
                ev_tt1[i+t*dmus] = (dmuname[i], year, res[1].fun)
            # 超效率模型，保存超效率结果
            if (sup == 1):
                if (res[0].fun != None):  # super结果仍有不可行解的可能
                    if (res[0].fun > 1):
                        ec_tt1[i+t*dmus] = (dmuname[i], year, res[0].fun)
                if (res[1].fun != None):  # super结果仍有不可行解的可能
                    if (res[1].fun > 1):
                        ev_tt1[i+t*dmus] = (dmuname[i], year, res[1].fun)

            # t1t 相邻super SBM模型效率E_t+1(x_t,y_t,z_t)
            res = SupEcv_t1t(cur=i, x_t=x_t, x_t1=x_t1, yg_t=yg_t,
                             yg_t1=yg_t1, yb_t=yb_t, yb_t1=yb_t1)
            # 非super无可行解时，用超效率模型解
            if (ec_t1t[i+t*dmus][2] == None):
                ec_t1t[i+t*dmus] = (dmuname[i], year, res[0].fun)
            if (ev_t1t[i+t*dmus][2] == None):
                ev_t1t[i+t*dmus] = (dmuname[i], year, res[1].fun)
            # 超效率模型，保存超效率结果
            if (sup == 1):
                if (res[0].fun != None):  # super结果仍有不可行解的可能
                    if (res[0].fun > 1):
                        ec_t1t[i+t*dmus] = (dmuname[i], year, res[0].fun)
                if (res[1].fun != None):  # 超效率结果仍有不可行解的可能
                    if (res[1].fun > 1):
                        ev_t1t[i+t*dmus] = (dmuname[i], year, res[1].fun)

        if (undesirable == 0):  # 不含非期望产出
            # tt 标准super SBM模型效率E_t(x_t,y_t)
            res = SupEcv_tt(cur=i, x_t=x_t, yg_t=yg_t)
            # 非super无可行解时，用超效率模型解
            if (ec_tt[i+t*dmus][2] == None):
                ec_tt[i+t*dmus] = (dmuname[i], year, res[0].fun)
            if (ev_tt[i+t*dmus][2] == None):
                ev_tt[i+t*dmus] = (dmuname[i], year, res[1].fun)
            # 超效率模型，保存超效率结果
            if (sup == 1):
                if (res[0].fun != None):  # super结果仍有不可行解的可能
                    if (res[0].fun > 1):
                        ec_tt[i+t*dmus] = (dmuname[i], year, res[0].fun)
                if (res[1].fun != None):  # super结果仍有不可行解的可能
                    if (res[1].fun > 1):
                        ev_tt[i+t*dmus] = (dmuname[i], year, res[1].fun)

            # gt 全局super SBM模型效率E_g(x_t,y_t)
            res = SupEc_g(cur_g=i+t*dmus, x_t=x_t,
                          yg_t=yg_t, x_g=x_g, yg_g=yg_g)
            # 非super无可行解时，用超效率模型解
            if (ec_gt[i+t*dmus][2] == None):
                ec_gt[i+t*dmus] = (dmuname[i], year, res[0].fun)
            if (ev_gt[i+t*dmus][2] == None):
                ev_gt[i+t*dmus] = (dmuname[i], year, res[1].fun)
            # 超效率模型，保存超效率结果
            if (sup == 1):
                if (res[0].fun != None):  # super结果仍有不可行解的可能
                    if (res[0].fun > 1):
                        ec_gt[i+t*dmus] = (dmuname[i], year, res[0].fun)
                if (res[1].fun != None):  # super结果仍有不可行解的可能
                    if (res[1].fun > 1):
                        ev_gt[i+t*dmus] = (dmuname[i], year, res[1].fun)

            # 最后一期不用计算tt1和t1t
            if (t == periods-1):
                continue

            # tt1 相邻super SBM模型效率E_t(x_t+1,y_t+1)
            res = SupEcv_tt1(cur=i, x_t=x_t, x_t1=x_t1, yg_t=yg_t, yg_t1=yg_t1)
            # 非super无可行解时，用超效率模型解
            if (ec_tt1[i+t*dmus][2] == None):
                ec_tt1[i+t*dmus] = (dmuname[i], year, res[0].fun)
            if (ev_tt1[i+t*dmus][2] == None):
                ev_tt1[i+t*dmus] = (dmuname[i], year, res[1].fun)
            # 超效率模型，保存超效率结果
            if (sup == 1):
                if (res[0].fun != None):  # super结果仍有不可行解的可能
                    if (res[0].fun > 1):
                        ec_tt1[i+t*dmus] = (dmuname[i], year, res[0].fun)
                if (res[1].fun != None):  # super结果仍有不可行解的可能
                    if (res[1].fun > 1):
                        ev_tt1[i+t*dmus] = (dmuname[i], year, res[1].fun)

            # t1t 相邻超效率SBM模型效率E_t+1(x_t,y_t)
            res = SupEcv_t1t(cur=i, x_t=x_t, x_t1=x_t1, yg_t=yg_t, yg_t1=yg_t1)
            # 非super无可行解时，用超效率模型解
            if (ec_t1t[i+t*dmus][2] == None):
                ec_t1t[i+t*dmus] = (dmuname[i], year, res[0].fun)
            if (ev_t1t[i+t*dmus][2] == None):
                ev_t1t[i+t*dmus] = (dmuname[i], year, res[1].fun)
            # 超效率模型，保存超效率结果
            if (sup == 1):
                if (res[0].fun != None):  # super结果仍有不可行解的可能
                    if (res[0].fun > 1):
                        ec_t1t[i+t*dmus] = (dmuname[i], year, res[0].fun)
                if (res[1].fun != None):  # super结果仍有不可行解的可能
                    if (res[1].fun > 1):
                        ev_t1t[i+t*dmus] = (dmuname[i], year, res[1].fun)

with tqdm(total=dmus*(2*periods-1)+1) as pbar:
    pbar.set_description('计算指数及分解:')
    all_index = []
    # tt与t1t,tt1在存储上差一个时期，先把tt和gt放入all_index，把t1t和tt1的位置先置为“-”
    for i in range(dmus):
        all_index.append((ec_tt[i][0], ec_tt[i][1], ec_tt[i]
                        [2], ev_tt[i][2], '-', '-', '-', '-', ec_gt[i][2], ev_gt[i][2]))
        # 更新进度显示
        pbar.update(1)
    # 第二期开始，t1t和tt1在存储时就对应第1期，下标用i。而tt和gt需要排后一期，下标要加上一个dmus，即i+dmus。
    for i in range(dmus*(periods-1)):
        all_index.append((ec_tt[i+dmus][0], ec_tt[i+dmus][1], ec_tt[i+dmus][2],
                        ev_tt[i+dmus][2], ec_tt1[i][2], ev_tt1[i][2], ec_t1t[i][2], ev_t1t[i][2], ec_gt[i+dmus][2], ev_gt[i+dmus][2]))
        # 更新进度显示
        pbar.update(1)

    # 计算指数分解
    ecc, ecv, tcc, pec, ptc, sec, sch, stc, bpcc, bpcv = [
    ], [], [], [], [], [], [], [], [], []
    mlc, mlv, mgc, mgv = [], [], [], []
    Zofio2007, rd1997, fglr1994, fglr1992_c, fglr1992_v, pl2005_c, pl2005_v = [
    ], [], [], [], [], [], []
    # 所有结果合并到all_index，
    # 0:名称，  1:时间
    # 2:ectt,   3:evtt
    # 4:ectt1,  5:evtt1
    # 6:ect1t,  7:evt1t
    # 8:ecgt,   8:evgt
    # tt和t1t1只是差一个时间，所以i为tt时，i+dmus为对应的t1t1
    # tt1和t1t1都有一个因素在下一个时间，所以都为i+dmus
    # 计算指数

    for i in range(dmus*(periods-1)):
        # MLC规模报酬不变的FGLR(1992)分解
        if (None in (all_index[i+dmus][2], all_index[i][2])):
            ecc.append(None)
        else:
            ecc.append(all_index[i+dmus][2]/all_index[i][2])
        if (None in (all_index[i+dmus][4], all_index[i+dmus][2], all_index[i][2], all_index[i+dmus][6])):
            tcc.append(None)
        else:
            tcc.append(np.sqrt(
                (all_index[i+dmus][4]/all_index[i+dmus][2])*(all_index[i][2]/all_index[i+dmus][6])))
        if (None in (ecc[i], tcc[i])):
            mlc.append(None)
        else:
            mlc.append(ecc[i]*tcc[i])
        fglr1992_c.append((all_index[i][0], str(
            all_index[i][1])+'-'+str(all_index[i+dmus][1]), mlc[i], ecc[i], tcc[i]))

        # MLV规模报酬可变的FGLR(1992)分解
        if (None in (all_index[i+dmus][3], all_index[i][3])):
            pec.append(None)
        else:
            pec.append(all_index[i+dmus][3]/all_index[i][3])
        if (None in (all_index[i+dmus][5], all_index[i+dmus][3], all_index[i][3], all_index[i+dmus][7])):
            ptc.append(None)
        else:
            ptc.append(np.sqrt(
                (all_index[i+dmus][5]/all_index[i+dmus][3])*(all_index[i][3]/all_index[i+dmus][7])))
        if (None in (pec[i], ptc[i])):
            mlv.append(None)
        else:
            mlv.append(pec[i]*ptc[i])
        fglr1992_v.append((all_index[i][0], str(
            all_index[i][1])+'-'+str(all_index[i+dmus][1]), mlv[i], pec[i], ptc[i]))

        # MLC指数的FGLR(1994)分解
        if (None in (all_index[i+dmus][2], all_index[i+dmus][3], all_index[i][2], all_index[i][3])):
            sec.append(None)
        else:
            sec.append((all_index[i+dmus][2]/all_index[i+dmus]
                        [3])/(all_index[i][2]/all_index[i][3]))
        fglr1994.append((all_index[i][0], str(
            all_index[i][1])+'-'+str(all_index[i+dmus][1]), mlc[i], pec[i], sec[i], tcc[i]))

        # MLC指数的RD(1997)分解
        if (None in (all_index[i+dmus][4], all_index[i+dmus][5], all_index[i][2], all_index[i][3], all_index[i+dmus][2], all_index[i+dmus][3], all_index[i+dmus][6], all_index[i+dmus][7])):
            sch.append(None)
        else:
            temp1 = (all_index[i+dmus][4]/all_index[i+dmus][5]) / \
                (all_index[i][2]/all_index[i][3])
            temp2 = (all_index[i+dmus][2]/all_index[i+dmus][3]) / \
                (all_index[i+dmus][6]/all_index[i+dmus][7])
            sch.append(np.sqrt(temp1*temp2))
        rd1997.append((all_index[i][0], str(all_index[i][1])+'-' +
                    str(all_index[i+dmus][1]), mlc[i], pec[i], sch[i], ptc[i]))

        # MLC指数的Zofio(2007)分解
        if (None in (all_index[i][2], all_index[i][3], all_index[i+dmus][6], all_index[i+dmus][7], all_index[i+dmus][4], all_index[i+dmus][5], all_index[i+dmus][2], all_index[i+dmus][3])):
            stc.append(None)
        else:
            temp1 = (all_index[i][2]/all_index[i][3]) / \
                (all_index[i+dmus][6]/all_index[i+dmus][7])
            temp2 = (all_index[i+dmus][4]/all_index[i+dmus][5]) / \
                (all_index[i+dmus][2]/all_index[i+dmus][3])
            stc.append(np.sqrt(temp1*temp2))
        Zofio2007.append((all_index[i][0], str(
            all_index[i][1])+'-'+str(all_index[i+dmus][1]), mlc[i], pec[i], sec[i], ptc[i], stc[i]))

        # GMC指数的PL(2005)分解
        if (None in (all_index[i+dmus][8], all_index[i][8], all_index[i+dmus][2], all_index[i][2])):
            bpcc.append(None)
        else:
            bpcc.append((all_index[i+dmus][8]/all_index[i+dmus]
                        [2])/(all_index[i][8]/all_index[i][2]))
        if (None in (ecc[i], bpcc[i])):
            mgc.append(None)
        else:
            mgc.append(ecc[i]*bpcc[i])
        pl2005_c.append((all_index[i][0], str(
            all_index[i][1])+'-'+str(all_index[i+dmus][1]), mgc[i], ecc[i], bpcc[i]))

        # GMV指数的PL(2005)分解
        if (None in (all_index[i+dmus][9], all_index[i][9], all_index[i+dmus][3], all_index[i][3])):
            bpcv.append(None)
        else:
            bpcv.append((all_index[i+dmus][9]/all_index[i+dmus]
                        [3])/(all_index[i][9]/all_index[i][3]))
        if (None in (all_index[i][3], all_index[i+dmus][3])):
            ecv.append(None)
        else:
            ecv.append(all_index[i+dmus][3]/all_index[i][3])
        if (None in (ecv[i], bpcv[i])):
            mgv.append(None)
        else:
            mgv.append(ecv[i]*bpcv[i])
        pl2005_v.append((all_index[i][0], str(
            all_index[i][1])+'-'+str(all_index[i+dmus][1]), mgv[i], ecv[i], bpcv[i]))
        # 更新进度显示
        pbar.update(1)

    # 写入结果
    ew = pd.ExcelWriter("allindex.xlsx")
    effs = pd.DataFrame(all_index, columns=('dmu', '时间', 'ectt',
                        'evtt', 'ectt1', 'evtt1', 'ect1t', 'evt1t', 'ecgt', 'evgt'))
    dffglr1992_c = pd.DataFrame(fglr1992_c, columns=(
        'dmu', '时间', 'mlc(M指数)', 'ecc(效率变化)', 'tcc(技术进步)'))
    dffglr1992_v = pd.DataFrame(fglr1992_v, columns=(
        'dmu', '时间', 'mlv(M指数)', 'pec(纯技术效率变动)', 'ptc(纯技术进步)'))
    dffglr1994 = pd.DataFrame(fglr1994, columns=(
        'dmu', '时间', 'mlc(M指数)', 'pec(纯技术效率变动)', 'sec(规模报变动)', 'tcc(技术进步)'))
    dfrd1997 = pd.DataFrame(rd1997, columns=(
        'dmu', '时间', 'mlc(M指数)', 'pec(纯技术效率变动)', 'sch(规模效率变动)', 'ptc(纯技术进步)'))
    dfZofio2007 = pd.DataFrame(Zofio2007, columns=(
        'dmu', '时间', 'mlc(M指数)', 'pec(纯技术效率变动)', 'sec(规模报变动)', 'ptc(纯技术进步)', 'stc(规模技术变化)'))
    dfpl2005_c = pd.DataFrame(pl2005_c, columns=(
        'dmu', '时间', 'mgc(GM指数)', 'ecc(效率变化)', 'bpcc(技术差距变动)'))
    dfpl2005_v = pd.DataFrame(pl2005_v, columns=(
        'dmu', '时间', 'mgv(GM指数)', 'ecv(效率变化)', 'bpcv(技术差距变动)'))

    effs.to_excel(ew, sheet_name='sbmeffs', index=False)
    dffglr1992_c.to_excel(ew, sheet_name='FGLR1992_C')
    dffglr1992_v.to_excel(ew, sheet_name='FGLR1992_V')
    dffglr1994.to_excel(ew, sheet_name='FGLR1994')
    dfrd1997.to_excel(ew, sheet_name='RD1997')
    dfZofio2007.to_excel(ew, sheet_name='Zofio2007')
    dfpl2005_c.to_excel(ew, sheet_name='PL2005_C')
    dfpl2005_v.to_excel(ew, sheet_name='PL2005_v')

    ew.close()
    # 更新进度显示
    pbar.update(1)
