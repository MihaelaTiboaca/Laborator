from math import *
import numpy as np


def aproxvalues(XYZ, xx, yy, f):
    xp = xx
    yp = yy

    x = XYZ[:, 0]
    y = XYZ[:, 2]

    ng = len(XYZ[:, 1])
    gg = ng * 2

    ho = 0
    count = 0
    for i in range(0, ng - 1):
        D = sqrt((x[i] - x[i + 1]) ** 2 + (y[i] - y[i + 1]) ** 2)
        d = sqrt((xp[i] - xp[i + 1]) ** 2 + (yp[i] - yp[i + 1]) ** 2)
        ho = ho + (D * f) / d
        count = count + 1
    ho = ho / count

    j = 0
    b = np.empty((gg, 4), dtype=np.double)
    ff = np.empty((gg, 1), dtype=np.double)

    for i in range(0, gg, 2):
        ff[i, 0] = x[j]
        ff[i + 1, 0] = y[j]

        b[i, 0] = xp[j]
        b[i, 1] = -yp[j]
        b[i, 2] = 1.0
        b[i, 3] = 0.0

        b[i + 1, 0] = xp[j]
        b[i + 1, 1] = yp[j]
        b[i + 1, 2] = 0.0
        b[i + 1, 3] = 1.0
        j = j + 1

    bt = b.transpose()
    btb = np.linalg.inv(np.matmul(bt, b))
    btf = np.matmul(bt, ff)
    delta = np.matmul(btb, btf)

    return [delta[2], delta[3], ho, atan2(delta[0], delta[1])]


def imageresection(XYZ, xx, yy, wpk, f):
    omega = wpk[0, 0]
    phi = wpk[0, 1]
    kappa = wpk[0, 2]
    xo = wpk[0, 3]
    yo = wpk[0, 4]
    zo = wpk[0, 5]
    xp = xx
    yp = yy
    delta = [1, 1, 1, 1, 1, 1, 1]

    ng = len(XYZ[:, 1])
    gg = ng * 2

    x = XYZ[:, 0]
    y = XYZ[:, 1]
    z = XYZ[:, 2]
    ii = 0
    v = np.empty((gg, 1), dtype=np.double)
    D = np.empty((6, 1), dtype=np.double)
    S = np.empty((6, 1), dtype=np.double)
    while np.max(np.abs(delta)) > .00001:
        mw = np.array([[1, 0, 0], [0, cos(omega), sin(omega)], [0, -sin(omega), cos(omega)]])
        mp = np.array([[cos(phi), 0, -sin(phi)], [0, 1, 0], [sin(phi), 0, cos(phi)]])
        mk = np.array([[cos(kappa), sin(kappa), 0], [-sin(kappa), cos(kappa), 0], [0, 0, 1]])
        m = np.matmul(np.matmul(mk, mp), mw)

        # partial derivatives

        dx = np.empty((ng, 1), dtype=np.double)
        dy = np.empty((ng, 1), dtype=np.double)
        dz = np.empty((ng, 1), dtype=np.double)
        s = np.empty((ng, 1), dtype=np.double)
        r = np.empty((ng, 1), dtype=np.double)
        q = np.empty((ng, 1), dtype=np.double)
        for k in range(0, ng):
            dx[k, 0] = x[k] - xo
            dy[k, 0] = yo - y[k]
            dz[k, 0] = z[k] - zo
            q[k, 0] = m[2, 0] * (x[k] - xo) + m[2, 1] * (z[k] - zo) + m[2, 2] * (yo - y[k])
            r[k, 0] = m[0, 0] * (x[k] - xo) + m[0, 1] * (z[k] - zo) + m[0, 2] * (yo - y[k])
            s[k, 0] = m[1, 0] * (x[k] - xo) + m[1, 1] * (z[k] - zo) + m[1, 2] * (yo - y[k])
        j = 0
        ff = np.empty((gg, 1), dtype=np.double)
        for k in range(0, gg, 2):
            ff[k, 0] = -(q[j, 0] * xp[j] + r[j, 0] * f) / q[j, 0]
            ff[k + 1, 0] = -(q[j, 0] * yp[j] + s[j, 0] * f) / q[j, 0]
            j = j + 1
        j = 0
        b = np.empty((gg, 6), dtype=np.double)
        for k in range(0, gg, 2):
            b[k, 0] = (xp[j] / q[j, 0]) * (-m[2, 2] * dz[j, 0] + m[2, 1] * dy[j]) + (f / q[j, 0]) * (
                    -m[0, 2] * dz[j] + m[0, 1] * dy[j])
            b[k, 1] = (xp[j] / q[j, 0]) * (
                    dx[j] * cos(phi) + dz[j] * (sin(omega) * sin(phi)) + dy[j] * (-sin(phi) * cos(omega))) + (
                              f / q[j, 0]) * (
                              dx[j] * (-sin(phi) * cos(kappa)) + dz[j] * (sin(omega) * cos(phi) * cos(kappa)) +
                              dy[j] * (-cos(omega) * cos(phi) * cos(kappa)))
            b[k, 2] = (f / q[j, 0]) * (m[1, 0] * dx[j] + m[1, 1] * dz[j] + m[1, 2] * dy[j])
            b[k, 3] = -((xp[j] / q[j, 0]) * m[2, 0] + (f / q[j, 0]) * m[0, 0])
            b[k, 4] = -((xp[j] / q[j, 0]) * m[2, 1] + (f / q[j, 0]) * m[0, 1])
            b[k, 5] = ((xp[j] / q[j, 0]) * m[2, 2] + (f / q[j, 0]) * m[0, 2])

            b[k + 1, 0] = (yp[j] / q[j, 0]) * (-m[2, 2] * dz[j] + m[2, 1] * dy[j]) + (f / q[j, 0]) * (
                    -m[1, 2] * dz[j] + m[1, 1] * dy[j])
            b[k + 1, 1] = (yp[j] / q[j, 0]) * (
                    dx[j] * cos(phi) + dz[j] * (sin(omega) * sin(phi)) + dy[j] * (-sin(phi) * cos(omega))) + \
                          (f / q[j, 0]) * (
                                  dx[j] * (sin(phi) * sin(kappa)) + dz[j] * (-sin(omega) * cos(phi) * sin(kappa)) + dy[
                              j] * (cos(omega) * cos(phi) * sin(kappa)))
            b[k + 1, 2] = (f / q[j, 0]) * (-m[0, 0] * dx[j] - m[0, 1] * dz[j] - m[0, 2] * dy[j])
            b[k + 1, 3] = -((yp[j] / q[j, 0]) * m[2, 0] + (f / q[j, 0]) * m[1, 0])
            b[k + 1, 4] = -((yp[j] / q[j, 0]) * m[2, 1] + (f / q[j, 0]) * m[1, 1])
            b[k + 1, 5] = ((yp[j] / q[j, 0]) * m[2, 2] + (f / q[j, 0]) * m[1, 2])
            j = j + 1

        # Least Square
        bt = b.transpose()
        btb = np.linalg.inv(np.matmul(bt, b))
        btf = np.matmul(bt, ff)
        delta = np.matmul(btb, btf)
        v = np.subtract(np.matmul(b, delta), ff)
        if ii == 0:
            D[:, ii] = delta[:, 0]
        else:
            D = np.append(D, delta, axis=1)

        omega = omega + delta[0, 0]
        phi = phi + delta[1, 0]
        kappa = kappa + delta[2, 0]
        xo = xo + delta[3, 0]
        yo = yo + delta[5, 0]
        zo = zo + delta[4, 0]
        print(
            'Corectie omega {0}, Corectie phi {1}, Corectie k {2}, Corectie X {3}, Corectie Y {4}, Corectie Z {5}\n'.format(
                delta[0, 0], delta[1, 0], delta[2, 0], delta[3, 0], delta[4, 0], delta[5, 0]))
        ii = ii + 1

    vt = v.transpose()
    r = b.shape[0] - b.shape[1]
    sigma = sqrt(np.matmul(vt, v) / r)
    for k in range (0,6):
            sigmai = sigma * sqrt(btb[k, k])
            S[k]=sigmai
    print(
            'Precizie omega {0}, Precizie phi {1}, Precizie k {2}, Precizie X {3}, Precizie Y {4}, Precizie Z {5}\n'.format(
                S[0], S[1], S[2], S[3], S[4], S[5]))
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    fig2,bx=plt.subplots()

    ax.plot(D[0, :], 'r', marker='D', label='omega')
    ax.plot(D[1, :], 'b', marker='D', label='phi')
    ax.plot(D[2, :], 'k', marker='D', label='kappa')
    bx.plot(D[3,: ],'r', marker='D', label='X0')
    bx.plot(D[4, :], 'b', marker='D', label='Y0')
    bx.plot(D[5, :], 'k', marker='D', label='Z0')
    ax.legend()
    bx.legend()
    plt.show()
    return [xo, zo, yo, omega * 180 / pi, phi * 180 / pi, kappa * 180 / pi, sigma]


