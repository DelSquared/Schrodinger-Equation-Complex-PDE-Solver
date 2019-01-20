import numpy as np 
import matplotlib.pyplot as plt 

L = 200 #number of discretisation nodes
T = 2000 #number of steps
dt = 0.001 #step size
x = range(L) #solution space (1D)

def D2(F): #Discrete Laplacian with periodic boundary conditions and next-neighbour kernel
  ker = np.array([1,-2,1])
  Fnew = np.zeros_like(F)
  for x in range(L):
    xleft = (x+L-1)%L
    xright = (x+L+1)%L
    Fnew[x]= ker[0]*F[xleft] + ker[1]*F[x] + ker[2]*F[xright]
  return Fnew

def schrodinger(Y): #schrodinger equation for a free particle (V=0) iY_t = Y_xx, letting Y=u+iv and solving Re and Im separately
  u,v=Y[0,:],Y[1,:]
  ut = -D2(v)
  vt = D2(u)
  return np.array([u+ut*dt,v+vt*dt])

uo = np.zeros(L)
vo = np.zeros(L)
xo = np.random.randint(0,L)
uo[(xo+1)%L]=1/np.sqrt(3)
uo[xo]=1/np.sqrt(3)
uo[(xo-1+L)%L]=1/np.sqrt(3)
Yo = np.array([uo,vo]) #setting initial conditions

fig=plt.figure()
plt.plot(x,np.zeros(L))
plt.plot(x,np.power(Yo[0,:],2)+np.power(Yo[1,:],2))
plt.ylim(0,1)
plt.savefig("out/Y 000.png") #plotting initial stage
plt.close(fig)
Y=schrodinger(Yo)
for t in range(T): #plotting time evolution
  fig=plt.figure()
  plt.plot(x,np.zeros(L))
  plt.plot(x,np.power(Y[0,:],2)+np.power(Y[1,:],2))
  plt.ylim(0,1)
  plt.savefig("out/Y {0:03d}.png".format(t+1))
  plt.close(fig)
  Y=schrodinger(Y)
