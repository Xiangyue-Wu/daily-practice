import time
import numpy as np
import matplotlib.pyplot as plt


def g2d(G):
    l = G.shape[0]
    D = np.zeros((l * l, l * l), dtype=float)
    for i in range(1, l + 1):
        for j in range(1, l + 1):
            if G[i - 1, j - 1] == 0:
                for m in range(1, l + 1):
                    for n in range(1, l + 1):
                        if G[m - 1, n - 1] == 0:
                            im = abs(i - m)
                            jn = abs(j - n)
                            if im + jn == 1 or (im == 1 and jn == 1):
                                D[(i - 1) * l + j - 1, (m - 1) * l + n - 1] = (im + jn) ** 0.5
    return D


def main():
    G = np.array([
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    ], dtype=int)
    MM = G.shape[0]
    Tau = np.ones((MM * MM, MM * MM), dtype=float)
    Tau = 8.0 * Tau
    K = 100
    M = 50
    S = 1
    E = MM * MM
    Alpha = 1
    Beta = 7
    Rho = 0.3
    Q = 1
    minkl = float("inf")
    mink = 0
    minl = 0
    D = g2d(G)
    N = D.shape[0]
    a = 1
    Ex = a * ((E % MM) - 0.5)
    if Ex == -0.5:
        Ex = MM - 0.5
    Ey = a * (MM + 0.5 - np.ceil(E / MM))
    Eta = np.zeros(N, dtype=float)
    for i in range(1, N + 1):
        ix = a * ((i % MM) - 0.5)
        if ix == -0.5:
            ix = MM - 0.5
        iy = a * (MM + 0.5 - np.ceil(i / MM))
        if i != E:
            Eta[i - 1] = 1 / (((ix - Ex) ** 2 + (iy - Ey) ** 2) ** 0.5)
        else:
            Eta[i - 1] = 100
    ROUTES = [[[] for _ in range(M)] for _ in range(K)]
    PL = np.zeros((K, M), dtype=float)
    tic = time.time()
    for k in range(K):
        for m in range(M):
            W = S
            Path = [S]
            PLkm = 0.0
            TABUkm = np.ones(N, dtype=int)
            TABUkm[S - 1] = 0
            DD = D.copy()
            DW = DD[W - 1, :].copy()
            DW1 = np.where(DW != 0)[0] + 1
            for j in range(len(DW1)):
                if TABUkm[DW1[j] - 1] == 0:
                    DW[DW1[j] - 1] = 0
            LJD = np.where(DW != 0)[0] + 1
            Len_LJD = len(LJD)
            while W != E and Len_LJD >= 1:
                PP = np.zeros(Len_LJD, dtype=float)
                for i in range(Len_LJD):
                    PP[i] = (Tau[W - 1, LJD[i] - 1] ** Alpha) * (Eta[LJD[i] - 1] ** Beta)
                sumpp = np.sum(PP)
                PP = PP / sumpp
                Pcum = np.cumsum(PP)
                Select = np.where(Pcum >= np.random.rand())[0]
                to_visit = LJD[Select[0]]
                Path.append(int(to_visit))
                PLkm = PLkm + DD[W - 1, to_visit - 1]
                W = int(to_visit)
                for kk in range(1, N + 1):
                    if TABUkm[kk - 1] == 0:
                        DD[W - 1, kk - 1] = 0
                        DD[kk - 1, W - 1] = 0
                TABUkm[W - 1] = 0
                DW = DD[W - 1, :].copy()
                DW1 = np.where(DW != 0)[0] + 1
                for j in range(len(DW1)):
                    if TABUkm[DW1[j] - 1] == 0:
                        DW[j] = 0
                LJD = np.where(DW != 0)[0] + 1
                Len_LJD = len(LJD)
            ROUTES[k][m] = Path
            if Path[-1] == E:
                PL[k, m] = PLkm
                if PLkm < minkl:
                    mink = k + 1
                    minl = m + 1
                    minkl = PLkm
            else:
                PL[k, m] = 0
        Delta_Tau = np.zeros((N, N), dtype=float)
        for m in range(M):
            if PL[k, m] != 0:
                ROUT = ROUTES[k][m]
                TS = len(ROUT) - 1
                PL_km = PL[k, m]
                for s in range(TS):
                    x = ROUT[s]
                    y = ROUT[s + 1]
                    Delta_Tau[x - 1, y - 1] = Delta_Tau[x - 1, y - 1] + Q / PL_km
                    Delta_Tau[y - 1, x - 1] = Delta_Tau[y - 1, x - 1] + Q / PL_km
        Tau = (1 - Rho) * Tau + Delta_Tau
    _ = time.time() - tic
    plotif = 1
    if plotif == 1:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        minPL = np.zeros(K, dtype=float)
        for i in range(K):
            PLK = PL[i, :]
            Nonzero = np.where(PLK != 0)[0]
            if len(Nonzero) > 0:
                PLKPLK = PLK[Nonzero]
                minPL[i] = np.min(PLKPLK)
            else:
                minPL[i] = np.nan
        plt.figure(1)
        plt.plot(minPL)
        plt.grid(True)
        plt.title('AS收敛曲线变化趋势')
        plt.xlabel('迭代次数')
        plt.ylabel('最小路径长度')
        plt.figure(2)
        plt.xlim(0, MM)
        plt.ylim(0, MM)
        for i in range(1, MM + 1):
            for j in range(1, MM + 1):
                x1 = j - 1
                y1 = MM - i
                x2 = j
                y2 = MM - i
                x3 = j
                y3 = MM - i + 1
                x4 = j - 1
                y4 = MM - i + 1
                if G[i - 1, j - 1] == 1:
                    plt.fill([x1, x2, x3, x4], [y1, y2, y3, y4], color='black', edgecolor='gray', linewidth=0.5)
                else:
                    plt.fill([x1, x2, x3, x4], [y1, y2, y3, y4], color='white', edgecolor='gray', linewidth=0.5)
        ax = plt.gca()
        ax.set_yticks(np.linspace(0, MM, 11))
        ax.set_yticklabels([20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0])
        plt.title('AS机器人运动轨迹')
        plt.xlabel('坐标x')
        plt.ylabel('坐标y')
        if mink > 0 and minl > 0:
            ROUT = ROUTES[mink - 1][minl - 1]
            LENROUT = len(ROUT)
            Rx = np.array(ROUT, dtype=float)
            Ry = np.array(ROUT, dtype=float)
            for ii in range(LENROUT):
                Rx[ii] = a * ((ROUT[ii] % MM) - 0.5)
                if Rx[ii] == -0.5:
                    Rx[ii] = MM - 0.5
                Ry[ii] = a * (MM + 0.5 - np.ceil(ROUT[ii] / MM))
            plt.plot(Rx, Ry, color='blue')
        plt.show()
    return None


if __name__ == "__main__":
    main()
