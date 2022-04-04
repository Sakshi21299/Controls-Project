# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 18:49:24 2022

@author: mew2
"""

#pyomo model for 3 step synthesis - Control Course Project 
#optimizing for conversion in a set time t


import pyomo.environ as pyo
import pyomo.dae as dae
#import idaes

model = pyo.ConcreteModel()

model.t1 = pyo.Param(initialize = 30)

model.time = dae.ContinuousSet(bounds = (0, model.t1))
#model.t2 = model.Var(initialize = 10)
#model.t3 = model.Var(initialize = 10)

model.k = pyo.Param(initialize=0.3)

#step 1 variables 
model.A1 = pyo.Var(model.time)
model.ReactA1 = pyo.Var(model.time)
model.Unreact = pyo.Var(model.time)

def _init_conditions(model):
    yield model.A1[0] == 1.5
    yield model.ReactA1[0] == 0
    yield model.Unreact[0] == 1
    
    
	#yield model.A1[0] == 1.5
	#yield model.ReactA1[0] == 0
    #yield model.Unreact[0] == 1
model.init_conditions = pyo.ConstraintList(rule=_init_conditions)


model.dA1 = dae.DerivativeVar(model.A1, wrt = model.time)
model.dReactA1 = dae.DerivativeVar(model.ReactA1, wrt = model.time)
model.dUnreact = dae.DerivativeVar(model.Unreact, wrt = model.time)


#differential equations dA1 = -k*[A1]*Unreact
def A1dot(model,i):
	return model.dA1[i] == -model.k*model.Unreact[i]*model.A1[i]
model.A1dotcon = pyo.Constraint(model.time, rule=A1dot)

def ReactA1dot(model,i):
	return model.dReactA1[i] == model.k*model.Unreact[i]*model.A1[i]
model.ReactA1dotcon = pyo.Constraint(model.time, rule=ReactA1dot)

def Unreactdot(model,i):
	return model.dUnreact[i] == -model.k*model.Unreact[i]*model.A1[i]
model.Unreactdotcon = pyo.Constraint(model.time, rule=Unreactdot)

#constraint to reach 95% conversion 
#def conv(model):
#    return (model.Unreact[-1] - model.ReactA1[-1])/model.Unreact[-1] >= 0.95
#model.conversioncon = pyo.Constraint(rule=conv)

#objective 
def _obj(model):
	return -(model.ReactA1[model.t1])
model.obj = pyo.Objective(rule=_obj)

# Discretize model using Orthogonal Collocation
discretizer = pyo.TransformationFactory('dae.collocation')
discretizer.apply_to(model,nfe=8,ncp=5)

solver=pyo.SolverFactory('ipopt')

results = solver.solve(model,tee=True)