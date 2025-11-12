#%%

import os
import numpy as np
from tensorflow.python.ops.init_ops_v2 import Initializer
from tensorflow.python.keras.backend import dtype
from tensorflow.python.keras.utils.tf_utils import dataset_is_infinite
from tensorflow.python.ops.variables import trainable_variables
import tensorflow as tf
import math
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
    'text.latex.preamble': r'\usepackage{amsfonts}'
})
import numpy as np
import scipy.special as sc
import matplotlib.pyplot as plt
from scipy.io import savemat

def circB(x,y,z):
    #Defining some parameters to be used in the formulas
    r_sq = x**2 + y**2 + z**2
    rho_sq = x**2 + y**2
    alpha_sq = 1. + r_sq - 2. * np.sqrt(rho_sq)
    beta_sq = 1. + r_sq + 2. * np.sqrt(rho_sq)
    k_sq = 1. - alpha_sq/beta_sq

    #Evaluate elliptic integrals
    e_k_sq = sc.ellipe(k_sq)
    k_k_sq = sc.ellipk(k_sq)
    
    #Magnetic field components in Cartesian coordinates
    Bx = 2. * x * z / (alpha_sq * rho_sq * np.sqrt(beta_sq)) * ((1. + r_sq) * e_k_sq - alpha_sq * k_k_sq)
    By = y * Bx / x
    Bz = 2. / (alpha_sq * np.sqrt(beta_sq)) * ((1. - r_sq) * e_k_sq + alpha_sq * k_k_sq)
    
    return np.array([Bx,By,Bz]) * 50

# Now calculate the mag field of collection of circles
def B(x,y,z):
    return 0.6*circB(x + 1.01,y + 1.0,z - 4.0) + 0.2*circB(x - 1.01,y - 1.0, z - 4.0) - 0.8*circB(x + 1.01,y - 1.0,z - 4.0) - 0.5*circB(x - 1.01,y + 1.0,z - 4.0) - 0.98*circB(x + 1.01,y + 1.0,z + 4.0) - 0.46*circB(x - 1.01,y - 1.0,z + 4.0) + 0.35*circB(x + 1.01,y - 1.0,z + 4.0) + 0.87*circB(x - 1.01,y + 1.0,z + 4.0)

model_path = "saved_models/trained_pinn_model"
# model_path = "saved_models/trained_plain_nn_model"
pinn_model = tf.keras.models.load_model(model_path)

L = 1
N_val = 100

x_test_np_grid = np.linspace(-L, L, N_val)
y_test_np_grid = np.linspace(-L, L, N_val)
z_test_np_grid = np.linspace(-L, L, N_val)
xx, yy, zz = np.meshgrid(x_test_np_grid, x_test_np_grid, z_test_np_grid, sparse=False)
xxx, yyy = np.meshgrid(x_test_np_grid, x_test_np_grid, sparse=False)
x_test_np = xx.reshape((N_val**3, 1))
y_test_np = yy.reshape((N_val**3, 1))
z_test_np = zz.reshape((N_val**3, 1))

x_test = tf.cast(x_test_np, dtype=tf.float32)
y_test = tf.cast(y_test_np, dtype=tf.float32)
z_test = tf.cast(z_test_np, dtype=tf.float32)
print(x_test.shape)
 

inputs = tf.concat([x_test, y_test, z_test], axis = 1)
temp_final = np.array([B(x_test_np[i], y_test_np[i], z_test_np[i]) for i in range(N_val**3)])
temp_final = temp_final.reshape((N_val, N_val, N_val, 3))
model_output = tf.reshape(pinn_model.call(inputs), [N_val, N_val, N_val, 3])


B_mag_Predicted = model_output[:,:,:,0]**2 + model_output[:,:, :, 1]**2 + model_output[:, :, :, 2]**2
B_mag_True = temp_final[:,:,:,0]**2 + temp_final[:,:, :, 1]**2 + temp_final[:, :, :, 2]**2

diff = model_output-temp_final
MSE_B = diff[:,:,:,0]**2 + diff[:,:,:,1]**2 + diff[:,:,:,2]**2


fig, ax = plt.subplots(figsize=(5, 5))
c = ax.contourf(xxx, yyy, B_mag_Predicted[:,:,1], levels=50, cmap='viridis')
ax.set_xlabel(r'$y$')
ax.set_ylabel(r'$x$')
ax.set_title(r'$\|\mathbf{B}_{\mathrm{Predicted}}\|$', pad=10)
ax.set_aspect('equal')
ax.spines['top'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)

# Add colorbar
cbar = fig.colorbar(c, ax=ax)
cbar.set_label(r'$x$')

plt.tight_layout()


fig, ax = plt.subplots(figsize=(5, 5))
c = ax.contourf(xxx, yyy, B_mag_True[:,:,1], levels=50, cmap='viridis')
ax.set_xlabel(r'$y$')
ax.set_ylabel(r'$x$')
ax.set_title(r'$\|\mathbf{B}_{\mathrm{True}}\|$', pad=10)
ax.set_aspect('equal')
ax.spines['top'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)

# Add colorbar
cbar = fig.colorbar(c, ax=ax)
cbar.set_label(r'$x$')

plt.tight_layout()


fig, ax = plt.subplots(figsize=(5, 5))
c = ax.contourf(xxx, yyy, MSE_B[:,:,1], levels=50, cmap='viridis')
ax.set_xlabel(r'$y$')
ax.set_ylabel(r'$x$')
ax.set_title(r'$\|\mathbf{B}_{\mathrm{True}}-\mathbf{B}_{\mathrm{Predicted}}\|$', pad=10)
ax.set_aspect('equal')
ax.spines['top'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)

# Add colorbar
cbar = fig.colorbar(c, ax=ax)
cbar.set_label(r'$x$')


L=1
N_val = 100
x_test_np_grid = np.linspace(-L, L, N_val)
y_test_np_grid = np.linspace(-L, L, N_val)
z_test_np_grid = np.linspace(-L, L, N_val)
x_test_np = x_test_np_grid.reshape(N_val, 1)
y_test_np = -0.68
z_test_np = 0.72

x_test = tf.cast(x_test_np, dtype=tf.float32)
y_test = tf.fill([N_val,1], y_test_np)
z_test = tf.fill([N_val,1], z_test_np)


inputs = tf.concat([x_test, y_test, z_test], axis = 1)
temp_final = np.array([B(x_test_np_grid[i], y_test_np, z_test_np) for i in range(N_val)])
print(temp_final.shape)
model_output = tf.reshape(pinn_model.call(inputs), [N_val, 3])   

 
plt.subplots(1, 3, figsize=(10, 3))
plt.subplot(131)
plt.plot(x_test_np, model_output[:,0], "-b", label="Prediction")
plt.plot( x_test_np, temp_final[:,0], "-g", label="Exact")
plt.ylim([-3, 3])
plt.legend(loc="upper left")
plt.xlabel(r"$x$")
plt.ylabel(r"$B_x$")

plt.subplot(132)
plt.plot(x_test_np, model_output[:,1], "-b", label="Prediction")
plt.plot(x_test_np, temp_final[:,1], "-g", label="Exact")
plt.ylim([-3, 3])
plt.title(r"Magnetic field prediction with plain NN at $z=0.72, y=-0.68$")
plt.xlabel(r"$x$")
plt.ylabel(r"$B_y$")

plt.subplot(133)
plt.plot(x_test_np, model_output[:,2], "-b", label="Prediction")
plt.plot(x_test_np, temp_final[:,2], "-g", label="Exact")
plt.ylim([-3, 3])
plt.xlabel(r"$x$")
plt.ylabel(r"$B_z$")



plt.tight_layout()
plt.savefig('fig7.pdf', bbox_inches='tight')
plt.show()


