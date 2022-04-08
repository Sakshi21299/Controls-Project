# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 18:45:29 2022

@author: ssnaik
"""

import pyomo.environ as pyo
import pyomo.dae as dae
import matplotlib.pyplot as plt


m = pyo.ConcreteModel()
#Fixed concentration of A1 in the injection streams
CA1_in = 1




#Parameters
m.tf = pyo.Param(initialize = 10)
m.K1 = pyo.Param(initialize = 10)
m.K2 = pyo.Param(initialize = 1)

#Set
m.t = dae.ContinuousSet(bounds = (0,m.tf))


#Variables
m.S = pyo.Var(m.t, domain = pyo.NonNegativeReals)
m.A1 = pyo.Var(m.t, domain = pyo.NonNegativeReals)
m.SA1 = pyo.Var(m.t, domain = pyo.NonNegativeReals)
m.SA1A1 = pyo.Var(m.t, domain = pyo.NonNegativeReals)
m.FA1 = pyo.Var(m.t, bounds = (0,10))


#Derivative Variables
m.dS = dae.DerivativeVar(m.S, wrt = m.t)
m.dA1 = dae.DerivativeVar(m.A1, wrt = m.t)
m.dSA1 = dae.DerivativeVar(m.SA1, wrt = m.t)
m.dSA1A1 = dae.DerivativeVar(m.SA1A1, wrt = m.t)

#Constraints
def _cS(m, t):
    return m.dS[t] ==  -m.K1*m.S[t]*(m.A1[t])
m.cS = pyo.Constraint(m.t,rule = _cS)

def _cA1(m, t):
    return m.dA1[t] ==  -m.K1*m.S[t]*m.A1[t] - m.K2*m.A1[t]*m.SA1[t] + m.FA1[t]*(CA1_in- m.A1[t])
m.cA1 = pyo.Constraint(m.t,rule = _cA1)

def _cSA1(m, t):
    return m.dSA1[t] ==  m.K1*m.S[t]*(m.A1[t]) - m.K2*m.SA1[t]*(m.A1[t])
m.cSA1 = pyo.Constraint(m.t,rule = _cSA1)

def _cSA1A1(m, t):
    return m.dSA1A1[t] ==  m.K2*m.SA1[t]*(m.A1[t]) 
m.cSA1A1 = pyo.Constraint(m.t,rule = _cSA1A1)





def _init_conditions(model):
    yield m.A1[0] == 1
    yield m.SA1[0] == 0
    yield m.SA1A1[0] == 0
    yield m.S[0] == 1
    yield m.FA1[0] == 0
    
    
m.init_conditions = pyo.ConstraintList(rule=_init_conditions)

#objective 
def _obj(model):
	return -m.SA1[m.tf] 
m.obj = pyo.Objective(rule=_obj)

# Discretize model using Orthogonal Collocation
discretizer = pyo.TransformationFactory('dae.finite_difference')

discretizer.apply_to(m,nfe=100, scheme = 'BACKWARD')

solver=pyo.SolverFactory('ipopt')

results = solver.solve(m,tee=True)
print(-1*pyo.value(m.obj))
#print(m.FA1.display())

def plotter(subplot, x, *series, **kwds): 
    plt.subplot(subplot) 
    for i, y in enumerate(series): 
        #plt.plot(x, [pyo.value(y[t]) for t in x], 'brgcmk'[i%6]+kwds.get('points',''))
        plt.plot(x, [pyo.value(y[t]) for t in x])
    plt.title(kwds.get('title',''))
    plt.legend(tuple(y.getname() for y in series)) 
    plt.xlabel(x.getname())

plotter(121, m.t, m.SA1 ,m.A1, m.SA1A1, title='Differential Variables') 
plotter(122, m.t, m.FA1, title='Control Variable', points='o') 
plt.show()







