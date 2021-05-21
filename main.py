from imageresection import *

photo = np.array(
    [[86.421, -83.977],
     [-100.916, 92.582],
     [-98.322, -89.161],
     [78.812, 98.123]]
)

xp = photo[:, 0]
yp = photo[:, 1]

# focal length
f = 152.916

# control points coordinates
XYZ = np.array(
    [[1268.102, 22.606, 1455.027],
     [732.181, 22.299, 545.344],
     [1454.553, 22.649, 731.666],
     [545.245, 22.336, 1268.232]]
)
x = XYZ[:, 0]
y = XYZ[:, 2]
z = XYZ[:, 1]


omega = 0
phi = 0
[xo, yo, zo, kappa] = aproxvalues(XYZ, xp, yp, f)
wpk = np.array([[omega, phi, kappa, xo, yo, zo]])

[Tx, Ty, Tz, w2, p2, k2, sigma] = imageresection(XYZ, xp, yp, wpk, -f)


print('X0: {0}, Y0: {1}, Z0: {2};\n omega: {3}, phi: {4}, kappa: {5}'.format(Tx, Ty, Tz, w2, p2, k2))
print('Std: {0}'.format(sigma))
