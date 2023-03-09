import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time

def calculateInertiaTensor(x, y, z):
    cx = np.mean(x)
    cy = np.mean(y)
    cz = np.mean(z)

    Ixx = np.sum(np.power(x-cx, 2))
    Iyy = np.sum(np.power(y-cy, 2))
    Izz = np.sum(np.power(z-cz, 2))

    Ixy = np.sum((x-cx)*(y-cy))
    Iyz = np.sum((y-cy)*(z-cz))
    Izx = np.sum((z-cz)*(x-cx))

    H = np.matrix([[Ixx, Ixy, Izx], [Ixy, Iyy, Iyz], [Izx, Iyz, Izz]])

    return H, cx, cy, cz

    
def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


t0 = time.time()

# These constants are to create random data for the sake of this example
N_POINTS = 100
TARGET_X_SLOPE = 2
TARGET_y_SLOPE = 3
TARGET_OFFSET  = 5
EXTENTS = 5
NOISE = 0.5

# Create random data.
# In your solution, you would provide your own xs, ys, and zs data.
xs = [np.random.uniform(2*EXTENTS)-EXTENTS for i in range(N_POINTS)]
ys = [np.random.uniform(2*EXTENTS)-EXTENTS for i in range(N_POINTS)]
zs = []
for i in range(N_POINTS):
    zs.append(xs[i]*TARGET_X_SLOPE + \
              ys[i]*TARGET_y_SLOPE + \
              TARGET_OFFSET + np.random.normal(scale=NOISE)/4)

# plot raw data
plt.figure()
ax = plt.subplot(111, projection='3d')
ax.scatter(xs, ys, zs, color='b')

# do fit
tmp_A = []
tmp_b = []
for i in range(len(xs)):
    tmp_A.append([xs[i], ys[i], 1])
    tmp_b.append(zs[i])
b = np.matrix(tmp_b).T
A = np.matrix(tmp_A)

# Manual solution
fit = (A.T * A).I * A.T * b
errors = b - A * fit
residual = np.linalg.norm(errors)

# Or use Scipy
# from scipy.linalg import lstsq
# fit, residual, rnk, s = lstsq(A, b)

print("solution: %f x + %f y + %f = z" % (fit[0], fit[1], fit[2]))
#print("errors: \n", errors)
print("residual:", residual)


# using eigenvalues and vectors


H, cx, cy, cz = calculateInertiaTensor(xs, ys, zs)

w, v = np.linalg.eig(H)

print(H)
print(w)
print(v)
print(v[2,:])

ax.scatter(cx, cy, cz, c='r')
ax.plot(xs=[cx, cx+v[0,0]*10], ys=[cy, cy+v[1,0]*10], zs=[cz, cz+v[2,0]*10], c='r')
ax.plot(xs=[cx, cx+v[0,1]*10], ys=[cy, cy+v[1,1]*10], zs=[cz, cz+v[2,1]*10], c='g')
ax.plot(xs=[cx, cx+v[0,2]*10], ys=[cy, cy+v[1,2]*10], zs=[cz, cz+v[2,2]*10], c='b')


# plot plane
xlim = ax.get_xlim()
ylim = ax.get_ylim()
X,Y = np.meshgrid(np.arange(xlim[0], xlim[1]),
                  np.arange(ylim[0], ylim[1]))
Z = np.zeros(X.shape)
for r in range(X.shape[0]):
    for c in range(X.shape[1]):
        Z[r,c] = fit[0] * X[r,c] + fit[1] * Y[r,c] + fit[2]
ax.plot_wireframe(X,Y,Z, color='k')

print("Max: {}, Min: {}, Std: {}, Mean: {}".format(np.max(errors), np.min(errors), np.std(errors), np.mean(errors)))
print("Time: {}".format(time.time()-t0))

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

set_axes_equal(ax)

plt.show()

##norm_fit = np.array([fit.item(0), fit.item(1), fit.item(2)])/np.linalg.norm(np.array([fit.item(0), fit.item(1), fit.item(2)]))
norm_fit = np.array([-fit.item(0), -fit.item(1), 1])/np.linalg.norm(np.array([-fit.item(0), -fit.item(1), 1]))

eigz = v[2,:]/np.linalg.norm(v[2,:])

eigz = np.array([eigz.item(0), eigz.item(1), eigz.item(2)])/np.linalg.norm(np.array([eigz.item(0), eigz.item(1), eigz.item(2)]))

print(norm_fit)
print(eigz[:])

np.dot(np.array(norm_fit), np.array([0, 0, 1]))

print(np.arccos(np.dot(norm_fit, np.array([0, 0, 1])))*180/np.pi)
print(np.arccos(np.dot(eigz, np.array([0, 0, 1])))*180/np.pi)

print(np.arccos(np.dot(norm_fit, eigz))*180/np.pi)