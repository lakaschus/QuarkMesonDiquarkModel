import numpy as np
from numpy import nan_to_num as ntn
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import ode
from numpy import gradient as grad
from findiff import FinDiff, coefficients, Coefficient
from scipy import interpolate
from joblib import Parallel, delayed
import multiprocessing

def func(k, ux, uxx, uy, uyy, uxy, x, y, T, mu):
  sqEpi = k**2 + 2*ux
  sqEs = sqEpi + 4*x*uxx
  sqENG = k**2 + 2*uy
  sqEG = sqENG + 4*y*uyy
  alpha0 = 16*mu**4 - 4*mu**2*sqEG - 4*mu**2*sqENG + sqEG*sqENG - 8*mu**2*sqEs + sqEG*sqEs + sqENG*sqEs - 16*uxy**2*x*y + 0.j
  alpha1 = 2*sqEG + 2*sqENG + 2*sqEs + 0.j
  alpha2 = 3 + 0.j
  beta0 = (-4*mu**2 + sqENG)*(-4*mu**2*sqEs + sqEG*sqEs - 16*uxy**2*x*y) + 0.j
  beta1 = 16*mu**4 - 4*mu**2*(sqEG + sqENG - 2*sqEs) + sqENG*sqEs + sqEG*(sqENG + sqEs) - 16*uxy**2*x*y + 0.j
  beta2 = 8*mu**2 + sqEG + sqENG + sqEs + 0.j
  beta3 = 1 + 0.j

  var1 = ( ( 27 * ( beta0 )**( 2 ) + ( 4 * ( beta1 )**( 3 ) + ( -18 * beta0 * beta1 * beta2 + ( -1 * ( beta1 )**( 2 ) * ( beta2 )**( 2 ) + 4 * beta0 * ( beta2 )**( 3 ) ) ) ) ) )**( 1/2 )
  var2 = ( -27 * beta0 + ( 9 * beta1 * beta2 + -2 * ( beta2 )**( 3 ) ) )

  z1 = ( ( 6 )**( -1/2 ) * ( ( ( 3 * ( 3 )**( 1/2 ) * var1 + var2 ) )**( -1/3 ) * ( ( ( 6 * ( 3 )**( 1/2 ) * var1 + 2 * var2 ) )**( 2/3 ) + ( -6 * ( 2 )**( 1/3 ) * beta1 + ( -2 * ( ( 3 * ( 3 )**( 1/2 ) * var1 + var2 ) )**( 1/3 ) * beta2 + 2 * ( 2 )**( 1/3 ) * ( beta2 )**( 2 ) ) ) ) )**( 1/2 ) )
  z2 = (1/2 * ( 3 )**( -1/2 ) * ( ( ( 3 * ( 3 )**( 1/2 ) * var1 + var2 ) )**( -1/3 ) * ( complex( 0,1 ) * ( complex( 0,1 ) + ( 3 )**( 1/2 ) ) * ( ( 6 * ( 3 )**( 1/2 ) * var1 + 2 * var2 ) )**( 2/3 ) + ( 6 * ( 2 )**( 1/3 ) * ( 1 + complex( 0,1 ) * ( 3 )**( 1/2 ) ) * beta1 + ( -4 * ( ( 3 * ( 3 )**( 1/2 ) * var1 + var2 ) )**( 1/3 ) * beta2 + complex( 0,-2 ) * ( 2 )**( 1/3 ) * ( complex( 0,-1 ) + ( 3 )**( 1/2 ) ) * ( beta2 )**( 2 ) ) ) ) )**( 1/2 ))
  z3 = (1/2 * ( 3 )**( -1/2 ) * ( ( ( 3 * ( 3 )**( 1/2 ) * var1 + var2 ) )**( -1/3 ) * ( ( -1 + complex( 0,-1 ) * ( 3 )**( 1/2 ) ) * ( ( 6 * ( 3 )**( 1/2 ) * var1 + 2 * var2 ) )**( 2/3 ) + ( 6 * ( 2 )**( 1/3 ) * ( 1 + complex( 0,-1 ) * ( 3 )**( 1/2 ) ) * beta1 + ( -4 * ( ( 3 * ( 3 )**( 1/2 ) * var1 + var2 ) )**( 1/3 ) * beta2 + complex( 0,2 ) * ( 2 )**( 1/3 ) * ( complex( 0,1 ) + ( 3 )**( 1/2 ) ) * ( beta2 )**( 2 ) ) ) ) )**( 1/2 ))

  res = ( -1/2 * ( z1 )**( -1 ) * ( ( -1 * ( z1 )**( 2 ) + ( z2 )**( 2 ) ) )**( -1 ) * ( ( -1 * ( z1 )**( 2 ) + ( z3 )**( 2 ) ) )**( -1 ) * ( alpha0 + ( ( z1 )**( 2 ) * alpha1 + ( z1 )**( 4 ) * alpha2 ) ) *
         1./np.tan( 1/2 * ( T )**( -1 ) * z1 ) + ( -1/2 * ( z2 )**( -1 ) * ( ( ( z1 )**( 2 ) + -1 * ( z2 )**( 2 ) ) )**( -1 ) * ( ( -1 * ( z2 )**( 2 ) + ( z3 )**( 2 ) ) )**( -1 ) * ( alpha0 + ( ( z2 )**( 2 ) * alpha1 + ( z2 )**( 4 ) * alpha2 ) ) * 1./np.tan( 1/2 * ( T )**( -1 ) * z2 ) + -1/2 * ( z3 )**( -1 ) * ( ( ( z1 )**( 2 ) + -1 * ( z3 )**( 2 ) ) )**( -1 ) * ( ( ( z2 )**( 2 ) + -1 * ( z3 )**( 2 ) ) )**( -1 ) * ( alpha0 + ( ( z3 )**( 2 ) * alpha1 + ( z3 )**( 4 ) * alpha2 ) ) * 1./np.tan( 1/2 * ( T )**( -1 ) * z3 ) ) )

  return np.real(res)

def Epion(ux, k):
  return np.sqrt(k**2 + 2*ux)

def Esig(x, ux, uxx, k):
  return np.sqrt(k**2 + 2*ux + 4*x*uxx)

def Eq(x, k):
  return np.sqrt(k**2 + hx**2*x)

def Epsi(x, y, k, mu, n):
  return np.sqrt(hy**2*y + ( Eq(x, k) + (-1)**n*mu )**2)

def ENG(uy, k):
  return np.sqrt(k**2 + 2*uy)

def nB(x, T):
  return 1/(np.exp(x/T) - 1)

def source(x, y, k, T, mu):
  Eps = Eq(x, k)
  Ek1 = Epsi(x, y, k, mu, 1)
  Ek2 = Epsi(x, y, k, mu, 0)
  return k**4/(12*np.pi**2)*(-12/Ek1*(1 - mu/Eps)*np.tanh(Ek1/(2*T))-12/Ek2*(1 + mu/Eps)*np.tanh(Ek2/(2*T)))

def dudk(u, k, T, mu):
  global k_stop
  u = u.reshape(-1, Nx)
  # Interpolation:
  #u_interp = interpolate.interp2d(x, y, u, kind='cubic')
  #ux = u_interp.__call__(x, y, dx=1)
  #uy = u_interp.__call__(x, y, dy=1)
  #uxx = u_interp.__call__(x, y, dx=2)
  #uyy = u_interp.__call__(x, y, dy=2)
  #uxy = u_interp.__call__(x, y, dx=1, dy=1)
  #print("interpolated derivatives: ", ux, uy, uxx, uyy, uyy, uxy)
  ux, uy = d_dx(u), d_dy(u)
  uxx, uyy = d2_dx2(u), d2_dy2(u)
  uxy = d2_dxdy(u)

  Ep = Epion(ux, k)
  Es = Esig(x, ux, uxx, k)
  Eng = ENG(uy, k)
  Eps = Eq(xgrd, k)
  Ek1 = Epsi(xgrd, ygrd, k, mu, 1)
  Ek2 = Epsi(xgrd, ygrd, k, mu, 0)
  dudk = k**4/(12*np.pi**2)*(3.0/Ep*(1.0/np.tanh(Ep/(2*T))) + 2/Eng*(2*np.sinh(Eng/T))/(np.cosh(Eng/T) - np.cosh((2*mu)/T))
                          - 8/Ek1*(1 - mu/Eps)*np.tanh(Ek1/(2*T)) - 8/Ek2*(1 + mu/Eps)*np.tanh(Ek2/(2*T))
                          - 4*(1/Eps*(np.tanh((Eps - mu)/(2*T)) + np.tanh((Eps + mu)/(2*T))))
           + 2*func(k, ux, uxx, uy, uyy, uxy, xgrd, ygrd, T, mu) )
  return np.squeeze(dudk).flatten()

def f(k, u, T, mu):
    return dudk(u, k, T, mu)

def solution2(T, mu):
    ode_order = 5
    nsteps = 100000
    ode_integrator = 'vode'#'lsoda'
    ode_method = 'bdf'#'adams'
    r_tol = 1e-15
    a_tol = 1e-15
    ode15s = ode(f)
    ode15s.set_integrator(ode_integrator, method=ode_method, order=ode_order, nsteps = nsteps, rtol=r_tol, atol=a_tol)
    #ode15s.set_integrator(ode_integrator, max_step=max_steps, rtol=r_tol, atol=a_tol, ixpr=True)
    u0_vec = np.squeeze(np.asarray(v0(xgrd, ygrd))).flatten()
    r = ode15s.set_initial_value(u0_vec, k[0]).set_f_params(T, mu)
    c = 1
    u = [ode15s.y]
    print("T, mu: ", (T, mu))
    while c < N_k:
        #print(ode15s.t)
        ode15s.integrate(ode15s.t-dk)
        c += 1
        u.append(ode15s.y)
        if ode15s.successful() == False:
            print("!Solver NOT successful!")
            print("T, mu: ", (T, mu))
            print("flow time: ", ode15s.t)
            return ode15s
    return u

cutoff, lam, v, m_lam, hx, hy, c = 1000, 0.001, 0, 969, 4.2, 3, 137**2*93
gam = 1.3
L = 120**2
L2 = L
Nx = 40
x = np.linspace(0, L, Nx)
Ny = 40
y = np.linspace(0, L2, Ny)
dx = np.abs(x[1] - x[0])
dy = np.abs(y[1] - y[0])
acc = 2
d_dx = FinDiff(1, dx, 1, acc = acc)
d2_dx2 = FinDiff(1, dx, 2, acc = acc)
d_dy = FinDiff(0, dy, 1, acc = acc)
d2_dy2 = FinDiff(0, dy, 2, acc = acc)
d2_dxdy = FinDiff((0, dy), (1, dx), acc = acc)
xgrd, ygrd = np.meshgrid(x, y)
N_k = 200
k_IR = 100
k_stop = cutoff
k = np.linspace(cutoff, k_IR, N_k) # DEFINE TIME HERE ################################################################################
dk = k[0] - k[1]

def v0(X, Y):
  return 1/2*m_lam**2*(X + gam*Y) + lam/4*(X + gam*Y - v**2)**2

start = time.time()
T_min, T_max = 10, 60
mu_min, mu_max = 250, 350
N_T, N_mu = 10, 30
T_array = np.linspace(T_min, T_max, N_T)
mu_array = np.linspace(mu_min, mu_max, N_mu)
mu_ax, T_ax = np.meshgrid(mu_array, T_array)
sol = [[None for _ in range(N_mu)] for _ in range(N_T)]
full_sol = [[None for _ in range(N_mu)] for _ in range(N_T)]
export_sol = [[None for _ in range(N_mu)] for _ in range(N_T)]
min_values, min_values_diq, m_sig, m_pi = np.zeros([N_T, N_mu]), np.zeros([N_T, N_mu]), np.zeros([N_T, N_mu]), np.zeros([N_T, N_mu])
expl = c*np.sqrt(xgrd)
num_cores = multiprocessing.cpu_count()
print("number of cores: ", num_cores)
start = time.time()
for i in range(N_T):
  print("Temp: ", T_array[i])
  full_sol[i] = Parallel(n_jobs=num_cores)(delayed(solution2)(T_array[i], mu_array[j]) for j in range(N_mu))
print("Duration: "+str(time.time()-start)[:6]+" s")

xnew = np.linspace(0, L, 8*Nx)
ynew = np.linspace(0, L2, 8*Ny)
dx_new, dy_new = xnew[1] - xnew[0], ynew[1] - ynew[0]
xgrd_new, ygrd_new = np.meshgrid(xnew, ynew)

expl = c*np.sqrt(xgrd_new)

for t in range(len(T_array)):
    for m in range(len(mu_array)):
        s = full_sol[t][m][-1].reshape(-1, Nx)
        export_sol[t][m] = s

        u_interp = interpolate.interp2d(x, y, s, kind='cubic')
        s = u_interp(xnew, ynew)
        argm = np.unravel_index(np.argmin(s - expl - 2*mu_array[m]**2*ygrd_new, axis=None), s.shape)
        print("argmin: ", argm)
        min_values[t, m] = np.sqrt(argm[1]*dx_new)
        print("min phi: ", min_values[t, m])
        min_values_diq[t, m] = np.sqrt(argm[0]*dy_new)
        print("min diq: ", min_values_diq[t, m])
        print(argm)

print(np.array(full_sol).shape)

param_list = np.array([cutoff, lam, v, m_lam, hx, hy, c, gam, L, L2, Nx, Ny, N_k, k_IR, \
                       T_min, T_max, mu_min, mu_max, N_T, N_mu, min_values, \
                       min_values_diq, export_sol, full_sol])
dat_name = 'threeColorQMDphaseDiagramFD_g_'+str(hx)+'gd_'+str(hy)+'_T_max'+str(T_max)+'N_T'+str(N_T)+'N_mu'+str(N_mu)+'Ngrd'+str(Nx)
np.save(dat_name, param_list)


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator

fig, ax1 = plt.subplots(nrows=1)
levels = MaxNLocator(nbins=32).tick_values(min_values.min(),
                                           min_values.max())
CS = ax1.contourf(mu_ax, T_ax, min_values, levels=levels)
fig.colorbar(CS, ax=ax1)
plt.title('Phase Diagram')
####
fig, ax1 = plt.subplots(nrows=1)
levels = MaxNLocator(nbins=32).tick_values(min_values_diq.min(),
                                           min_values_diq.max())
CS = ax1.contourf(mu_ax, T_ax, min_values_diq, levels=levels)
fig.colorbar(CS, ax=ax1)
plt.title('Diquark Phase Diagram')
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_wireframe(mu_ax, T_ax, min_values, color='red')
ax.plot_wireframe(mu_ax, T_ax, min_values_diq)
plt.show()
####
fig, ax1 = plt.subplots(nrows=1)
levels = MaxNLocator(nbins=32).tick_values(min_values_diq.min(),
                                           min_values_diq.max())
CS = ax1.contourf(mu_ax, T_ax, min_values_diq, levels=levels)
fig.colorbar(CS, ax=ax1)
plt.title('Diquark Phase Diagram')
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_wireframe(mu_ax, T_ax, min_values, color='red')
ax.plot_wireframe(mu_ax, T_ax, min_values_diq)
ax.view_init(0, -90)
plt.show()
