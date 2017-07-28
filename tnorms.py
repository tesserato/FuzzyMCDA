from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
# %matplotlib notebook

a = 2.5
b = 2


def f(x):
    return (a/2 + np.tanh(b * (x * 2 - 1))) / a

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('Pesquisa Tipo I: W = 0.052')
ax.set_ylabel('Pesquisa Tipo II: W = 0.241')
ax.set_zlabel('U')

# Axes3D.text('eixo x', 'y', 'z')

# Make data.
X = np.arange(0, 1, 0.01)
Y = np.arange(0, 1, 0.01)
X, Y = np.meshgrid(X, Y)
AND = f(X+.052) * f(Y+.241) #f(-1 + X)*f( .5 + Y)
# OR = (X+Y)- X*Y

# Plot the surface.
surf = ax.plot_surface(X, Y, AND,rstride=1, cstride=1, linewidth=0, antialiased=True, shade=True, cmap='coolwarm')
# surf = ax.plot_surface(X, Y, OR, antialiased=False)

#
plt.show()