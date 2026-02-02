import numpy as np
from matplotlib import pyplot as plt

def integral_cubic_kernel(r):
    if r < -2.0: return 0
    elif r < -1.0: return 1/24 * r**4 + 1/3 * r**3 + r**2 + 4/3 * r + 2/3
    elif r < 0.0: return -1/8 * r**4 - 1/3 * r**3 + 2/3 * r + 1/2
    elif r < 1.0: return 1/8 * r**4 - 1/3 * r**3 + 2/3 * r + 1/2
    elif r < 2.0: return -1/24 * r**4 + 1/3 * r**3 - r**2 + 4/3 * r + 1/3
    else: return 1

def integral_quadratic_kernel(r):
    if r < -1.5: return 0
    elif r < -0.5: return 1/6 * (3/2 + r)**3
    elif r < 0.5: return -1/3 * r**3 + 3/4 * r + 1/2
    elif r < 1.5: return -1/6 * (3/2 - r)**3 + 1
    else: return 1

x_cubic = np.linspace(-2, 2, 100)
x_quadratic = np.linspace(-1.5, 1.5, 100)

y_cubic = [integral_cubic_kernel(None, r) for r in x_cubic]
y_quadratic = [integral_quadratic_kernel(None, r) for r in x_quadratic]

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(x_cubic, y_cubic, label='Integral Cubic Kernel', color='blue')
plt.title('Integral of Cubic Kernel')
plt.xlabel('r')
plt.ylabel('Integral Value')
plt.grid()
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(x_quadratic, y_quadratic, label='Integral Quadratic Kernel', color='orange')
plt.title('Integral of Quadratic Kernel')
plt.xlabel('r')
plt.ylabel('Integral Value')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()