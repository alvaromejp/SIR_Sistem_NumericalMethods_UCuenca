# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 00:18:40 2021

@author: AlvMej
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import odeint

def met_eulerProg(f,a,b,h,Y0):    
    
    xi = a
    yi = Y0
    n = int((b-a)/h)
    ys = [Y0]
    for i in range(1,n+1): 
        yi_1 = yi + h * f(xi,yi) 
        ys.append(yi_1)
        yi = yi_1
        xi += h
    return ys
def met_eulerReg(f,a,b,h,Y0):    
    xi = a
    yi = Y0
    n = int((b-a)/h)
    ys = [Y0]  
    for i in range(1,n+1): 
        fun = lambda yi_1: yi + h * f(xi,yi_1) - yi_1
        yi_1 = fsolve(fun,1)[0]
        ys.append(yi_1)
        yi = yi_1
        xi += h
    return ys
def met_trap(f,a,b,h,Y0):    
    xi = a
    yi = Y0
    n = int((b-a)/h)
    ys = [Y0]  
    for i in range(1,n+1): 
        fun = lambda yi_1: yi + (h/2)*(f(xi,yi)+f(xi+h,yi_1))- yi_1
        yi_1 = fsolve(fun,1)[0]
        ys.append(yi_1)
        yi = yi_1
        xi += h
    return ys
def met_rk1(f,a,b,h,Y0):    
    xi = a
    yi = Y0
    n = int((b-a)/h)
    ys = [Y0] 
    for i in range(1,n+1): 
        k1 = f(xi,yi)
        k2 = f(xi+h/2, yi+h*(k1/2))
        yi_1 = yi + h*k2
        ys.append(yi_1)
        yi = yi_1
        xi += h
    return ys

def met_rk12(f,a,b,h,Y0):    
    xi = a
    yi = Y0
    n = int((b-a)/h)
    ys = [Y0] 
    for i in range(1,n+1): 
        k1 = f(xi,yi)
        k2 = f(xi+h, yi+h*k1)
        yi_1 = yi + (h/2)*(k1+k2)
        ys.append(yi_1)
        yi = yi_1
        xi += h
    return ys
def met_rk34(f,a,b,h,Y0):    
    xi = a
    yi = Y0
    n = int((b-a)/h)
    ys = [Y0] 
    for i in range(1,n+1): 
        k1 = f(xi,yi)
        k2 = f(xi+2*h/3, yi+2*h*(k1/3))
        yi_1 = yi + (h/4)*(k1+3*k2)
        ys.append(yi_1)
        yi = yi_1
        xi += h
    return ys
def met_rk4std(f,a,b,h,Y0):    
    xi = a
    yi = Y0
    n = int((b-a)/h)
    ys = [Y0] 
    for i in range(1,n+1): 
        k1 = f(xi,yi)
        k2 = f(xi+h/2, yi+h*(k1/2))
        k3 = f(xi+h/2, yi+h*(k2/2))
        k4 = f(xi+h  , yi+h*k3)
        yi_1 = yi + (h/6)*(k1+ 2*k2+ 2*k3 +k4)
        ys.append(yi_1)
        yi = yi_1
        xi += h
    return ys

def sol_SIR(met, s0, i0, r0, t0, tf, N):
    h = (tf-t0)/N
    val_s = [s0]
    val_i = [i0]
    val_r = [r0]
    
    for val in range(N):
        def y_s(t = 0, s = 0): #Ec. Diferencial
            b = 1/2
            return   -b*s*i0
        y_si = met(y_s,t0,t0+h,h,s0)[-1]
        val_s.append(y_si)
        
        def y_i(t = 0,i = 0):
            b = 1/2
            k = 1/3
            return b*s0*i-k*i
        y_ii = met(y_i,t0,t0+h,h,i0)[-1]
        val_i.append(y_ii)
        
        def y_r(t = 0, r = 0):
            k = 1/3
            return k*i0
        y_ri = met(y_r,t0,t0+h,h,r0)[-1]
        val_r.append(y_ri)
        
        s0 = y_si
        i0 = y_ii
        r0 = y_ri
        
        t0+=h
    return val_s, val_r, val_i

def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

def grafic_SIR(xs,s,r,i,tit):
    fig, ax = plt.subplots()
    plt.plot(xs,s,label="s(t)", color = "b")
    plt.plot(xs,r,label ="r(t)", color = "g")
    plt.plot(xs,i,label="i(t)", color = "r")
    plt.legend()
    plt.title('{0}'.format(tit))
    plt.xlabel("Tiempo en días")
    plt.ylabel('valores de s,t,i')
    plt.grid()
    plt.show()

# población inicial, N.
N_pob = 1 # poblaciçon de un país como España
 
# Número inicial de infectados y recuperados, I0 and R0.
i0 = 1.27e-6
r0 = 0
 
# El resto, casi todo N, es susceptible de infectarse
s0 = 1
 
# Tasas de contagio y recuperación.
beta = 1/2 # contagio
gamma = 1/3 # recuperación
 
t0 = 0
tf = 140
"""
N es el numero de particiones del intervalo 
a mayor N menor h
"""
N = 600
h = (tf-t0)/N

xs = np.linspace(t0,tf,N+1)

# Pasos temporales (en días)
t = np.linspace(t0, tf, N)
# condiciones iniciales
y0 = s0, i0, r0
# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(N_pob, beta, gamma))


######################################################################
### SOLUCION ANALITICA

"""
S, I , R son arrays con la solución analítica
"""
S, I, R = ret.T


"""
Cada lista de s_metodoiterativo contiene las aproximaciones según su método. 
"""

###################################################################

### GRAFICAS DE LAS APROXIMACIONES

title = "Metodo de Euler Implicito"
s_eulerReg, r_eulerReg, i_eulerReg = sol_SIR(met_eulerReg, s0, i0, r0, t0, tf, N)
sol_euler = grafic_SIR(xs,s_eulerReg, r_eulerReg, i_eulerReg, title)
sol_euler

title = "Metodo de Euler Explicito"
s_eulerProg, r_eulerProg, i_eulerProg = sol_SIR(met_eulerProg, s0, i0, r0, t0, tf, N)
sol_eulerProg = grafic_SIR(xs,s_eulerProg, r_eulerProg, i_eulerProg, title)
sol_eulerProg

title = "Metodo del Trapecio"
s_trap, r_trap, i_trap = sol_SIR(met_trap, s0, i0, r0, t0, tf, N)
sol_trap = grafic_SIR(xs,s_trap, r_trap, i_trap, title)
sol_trap

title = "Metodo RK b2 = 1 (Punto Medio)"
s_rk1, r_rk1, i_rk1 = sol_SIR(met_rk1, s0, i0, r0, t0, tf, N)
sol_rk1 = grafic_SIR(xs,s_rk1, r_rk1, i_rk1, title)
sol_rk1

title = "Metodo RK b2 = 1/2 (Euler Modificado)"
s_rk12, r_rk12, i_rk12 = sol_SIR(met_rk12, s0, i0, r0, t0, tf, N)
sol_rk12 = grafic_SIR(xs,s_rk12, r_rk12, i_rk12, title)
sol_rk12

title = "Metodo RK b2 = 3/4 (Heun)"
s_rk34, r_rk34, i_rk34 = sol_SIR(met_rk34, s0, i0, r0, t0, tf, N)
sol_rk34 = grafic_SIR(xs,s_rk34, r_rk34, i_rk34, title)
sol_rk34

title = "Metodo RK Orden 4 Estándar "
s_rk4std, r_rk4std, i_rk4std = sol_SIR(met_rk4std, s0, i0, r0, t0, tf, N)
sol_rk4std = grafic_SIR(xs,s_rk4std, r_rk4std, i_rk4std, title)
sol_rk4std



errorS = [abs(S - s_eulerReg[1:]), abs(S-s_eulerProg[1:]), abs(S-s_trap[1:]),abs(S-s_rk1[1:]),abs( S-s_rk12[1:]),abs( S-s_rk34[1:]), abs(S- s_rk4std[1:])]
errorR = [abs(R - r_eulerReg[1:]), abs(R-r_eulerProg[1:]), abs(R-r_trap[1:]),abs(R-r_rk1[1:]), abs(R-r_rk12[1:]), abs(R-r_rk34[1:]), abs(R- r_rk4std[1:])]
errorI = [abs(I - i_eulerReg[1:]), abs(I-i_eulerProg[1:]), abs(I-i_trap[1:]),abs(I-i_rk1[1:]),abs(I-i_rk12[1:]), abs(I-i_rk34[1:]), abs(I- i_rk4std[1:])]
# print("Analitica",S)
# print("S euler reg",s_eulerReg)
cero = []
for i in range(N+1):
    cero.append(0)
xs = np.linspace(0,140,600)


######################################################################################################################################
# GRAFICAS DE LOS ERRORES
plt.figure(figsize=(10,5))
plt.plot(xs,errorS[0],linestyle = "--", label= "Euler RG") 
plt.plot(xs,errorS[1],linestyle = "--", label= "Euler PG") 
plt.plot(xs,errorS[2], label= "Trapecio") 
plt.plot(xs,errorS[3], label= "RK1") 
plt.plot(xs,errorS[4], label= "RK12") 
plt.plot(xs,errorS[5], label= "RK34") 
plt.plot(xs,errorS[6], label= "RK Ord4") 

plt.title("Error de s(t), h = {0}".format(round(h,4)))
plt.legend()
plt.ylabel("Errores maximos absolutos")
plt.xlabel("Tiempo")
plt.show()

plt.figure(figsize=(10,5))
plt.plot(xs,errorR[0],linestyle = "--", label= "Euler RG") 
plt.plot(xs,errorR[1],linestyle = "--", label= "Euler PG") 
plt.plot(xs,errorR[2], label= "Trapecio") 
plt.plot(xs,errorR[3], label= "RK1") 
plt.plot(xs,errorR[4], label= "RK12") 
plt.plot(xs,errorR[5], label= "RK34") 
plt.plot(xs,errorR[6], label= "RK Ord4") 

plt.title("Error de r(t), h = {0}".format(round(h,4)))
plt.legend()
plt.ylabel("Errores maximos absolutos")
plt.xlabel("Tiempo")
plt.show()

plt.figure(figsize=(10,5))
plt.plot(xs,errorI[0],linestyle = "--", label= "Euler RG") 
plt.plot(xs,errorI[1],linestyle = "--", label= "Euler PG") 
plt.plot(xs,errorI[2], label= "Trapecio") 
plt.plot(xs,errorI[3], label= "RK1") 
plt.plot(xs,errorI[4], label= "RK12") 
plt.plot(xs,errorI[5], label= "RK34") 
plt.plot(xs,errorI[6], label= "RK Ord4") 

plt.title("Error de i(t), h = {0}".format(round(h,4)))
plt.ylabel("Errores maximos absolutos")
plt.xlabel("Tiempo")
plt.legend()
plt.show()

######################################################################################################################################






