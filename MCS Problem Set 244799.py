#!/usr/bin/env python
# coding: utf-8

# # Mathematics of Complex Systems - Problem Set

# Candidate: 244799

# ### ODES

# $[\dot{B}]=\beta\frac{[B]}{N}[A]-\gamma[B]$ <br/>
# $[\dot{A}]=-\beta\frac{[B]}{N}[A]+\gamma[B]$

# ### Analytical Work

#  ̇1. Using the fact that $[A] + [B] = N $at all times, write $[B]$ as a function of $[B]$, i.e.,
# the expression should no longer involve $[A]$. This is your (so-called) mean-field equation. (2 marks)

# Since:<br/>
# $N = [A] + [B]$ <br/>
# $[A] = N - [B]$ <br/>
# So: <br/>
# $ [\dot{B}]=\beta\frac{[B]}{N}(N-[B])-\gamma[B]$ <br/>
# This is the <i> mean field equation. </i>

# 2. Find the equilibria of the system and determine their stability. From now on, we will refer to the non-zero equilibrium as $B^∗$. You may find it useful to write your results in terms of the following quantity $R_{0} = \frac{β}{γ}$ . Plot the phase portrait, i.e., $[B]$ vs $[A]$, identifying the equilibria and their stability (following convention described in the 2nd synchronous lecture of Unit 5). (8 marks)

# Since: <br/>
# $ [\dot{B}]=\beta\frac{[B]}{N}(N-[B])-\gamma[B]$ <br/><br/>
# We set the equation to 0 to get the equilibria and factorise:<br/>
# $ \beta\frac{[B]}{N}(N-[B])-\gamma[B] = 0$ <br/></br>
# $ \beta [B] - \frac{\beta[B]^2}{N} - \gamma[B] = 0 $ <br/> 
# $ [B](\beta - \frac{\beta[B]}{N} - \gamma) = 0 $ <br/>
# We take $(\beta - \frac{\beta[B]}{N} - \gamma) = 0$, as $[B] = 0$ is zero answer <br/>
# $ N\beta - \beta[B] - \gamma N = 0 $ <br/>
# $[B] =  N\frac{-N\gamma}{\beta}$ <br/>
# $[B] = N(1 - \frac{\gamma}{\beta})$ <br/><br/>
# $𝐵^∗ = N(1 - \frac{1}{R_0})$
# <br/> <br/>
# Fixed points are ($[A]$,$[B]$) are ($N$,$0$) and ($\frac{N}{R_0}$, $N(1-\frac{1}{R_0}$) )
# <br/><br/>
# Create the Jacobrian Matrix: <br/>
# $J$ = $\begin{bmatrix}
#        \frac{\beta[A]}{N} - \gamma & \frac{\beta[B]}{N} \\
#        \frac{-\beta[A]}{N} + \gamma & \frac{-\beta[B]}{N}
#        \end{bmatrix}$
# <br/><br/>
# From substituting in the fixed points you get the eigenvalues: <br/>
# For ($N$,$0$) the eigenvalues are:    $0$ and $\beta - \gamma$ <br/>
# For ($\frac{N}{R_0}$, $N(1-\frac{1}{R_0}$) ) the eigenvalues are: $0$ and $\gamma - \beta$
# <br/><br/>
# 
# <b> When $\gamma > \beta$ : <br/> </b>
# for ($N$,$0$), the non-zero fixed point will be negative and therefore stable.
# <br/>
# for  $N(1-\frac{1}{R_0}$) ), the non-zero fixed point will be positive and therefore be unstable. <br>
# <b> When $\beta > \gamma$ : <br/> </b>
# for ($N$,$0$), the non-zero fixed point will be positive and therefore unstable.
# <br/>
# for  $N(1-\frac{1}{R_0}$) ), the non-zero fixed point will be negative and therefore also be stable. <br/>
# <b> When $\beta = \gamma$ : <br/> </b> 
# Both eigenvalues would be 0.
# <br/> This presents an unique state where the system is not stable, as the eigenvalues are not positive, but it is also not dramatically unstable, as they are also non-negative.

# Please see attachment <b>'phase_diagram.jpg'</b> for the phase plot.
# 
# 

# In[1]:


#Packages for analysis - please run
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import fsolve
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from random import expovariate


# 3. Produce the bifurcation plot for this system, that is, plot the value of the equilibria as a function of $R_0$, with $R_0$ taking values from 0.1 to 5.0. For this question, the value of $N$ is irrelevant (provided it’s strictly positive) so use 1000 for example. This should be done using Python. (4 marks)

# In[2]:


#function for the stability point
def stability_point(N,Ro):
    
    stability_point = N - (N/Ro)
    return stability_point

#range of values for Ro
Ro_range = np.linspace(0.1, 5.0, 1000)

#store the output
stability_values = []

for Ro_value in Ro_range:
    output = stability_point(1000, Ro_value)
    stability_values.append(output)
        
#plot output
plt.plot(Ro_range,stability_values,c='slateblue')
plt.grid()
plt.ylabel("equilibria value")
plt.xlabel("Ro")
plt.title("Bifurcation Plot")
plt.show()


# 4. In this question, you are going to integrate $[B]$ analytically to obtain an expression for $[B](t)$, i.e., an expression that gives the number of individuals in state $B$ in time. It is rarely the case that this can be done but with this system, it is possible. You will do this in four steps:

# - Starting from the mean-field equation, factorise the right hand side by $[B]^2$, then write an expression for $\frac{1}{B^2}[\dot{B}]$. (4 marks)

# $ [\dot{B}]=\beta\frac{[B]}{N}(N-[B])-\gamma[B]$ <br/><br/>
# $ [\dot{B}]=[B]^2(\frac{\beta}{N[B]}(N-[B])-\frac{\gamma}{[B]})$ <br/><br/>
# $ [\dot{B}]=[B]^2(\frac{\beta-\gamma}{[B]}-\frac{\beta}{N})$ <br/><br/>
# $ \frac{1}{[B]^2}\dot{[B]}=\frac{\beta-\gamma}{[B]}-\frac{\beta}{N}$ <br/><br/>

# - Consider the following variable substitution: $y = \frac{1}{[B]}$. Using the chain rule, express $\dot{y}$ in terms of $[B]$ , then derive a simple expression for $\dot{y}$, i.e., this expression should only involve $y$ terms and parameters of the system. There shouldn't be any $[B]$ or $[A]$ . However, it will be helpful to use $B^*$ (calculated in the 2nd question) to simplify the expression. (4 marks)

# $ \dot{y} = \frac{dy}{d[B]} \frac{d[B]}{dt} $ <br/><br/>
# $ \dot{y} = \frac{-1}{[B]^2} [\dot{B}] $ <br/><br/>
# $ -\dot{y} = (\beta - \gamma)y - \frac{\beta}{N} $ <br/><br/>
# $ \dot{y} = (\beta - \gamma)y + \frac{\beta}{N} $ <br/><br/>
# 
# $ B^* = \beta - \gamma $ <br/>
# if we use $\lambda = \beta - \gamma $ : <br/>
# $ \dot{y} = \lambda y + \frac{\beta}{N} $
# <br/><br/>

# - Integrate this equation. You should be able to do this without any help, but if help is needed, you should note that this expression looks very much like the equation we solved during a synchronous lecture in Unit 4, replacing $λ$ and $I$ by appropriate quantities. You can then use the result to derive an equation for $y(t)$. Please see short document summarising the derivation from the lecture. (4 marks)

# 
# 
# integration using the product rule: <br/>
# $\frac{\beta}{N} = I, \dot{y} = \lambda y + I $ <br/><br/>
# if $ y(t) = h(t)g(t) $ <br/> <br/>
# using the product rule: <br/>
# $ y(t) = \left( \frac{I}{k \lambda} e^{\lambda t} + c \right) ke^{-\lambda t}$ <br/>
# $ y(t) = \frac{I}{\lambda} + cke^{-\lambda t} $ <br/>
# if we sub $ck$ for one constant $k_1$, and replace sub in $I$ and $\lambda$ : <br/> <br/>
# 
# $ y(t) = \frac{\frac{\beta}{N}}{\beta - \gamma} + k_1 e^{t(\gamma - \beta)} $ <br/><br/>
# 
# where $B* = N \left( 1 - \frac{1}{R_0} \right) $ <br/>
# 
# $ y(t) = \frac{1}{B*} + k_1 e^{t(\gamma - \beta)} $ <br/> <br/>
# at t = 0: <br/>
# $ y = \frac{1}{B*} + k_1 $ <br/>
# so the constant is equal to: </br>
# 
# $ k_1 = \frac{1}{B_0} - \frac{1}{B*} $ <br/>
# 
# $ y(t) = \frac{1}{B*} + \left(\frac{1}{B_0} - \frac{1}{B*} \right)e^{t(\gamma - \beta)} $
# 
# 

# - You can now produce a fully worked out expression for $[B](t)$ by remembering that $[B]=B_0$ at time $t=0$. (4 marks) <br/>
# <b>NB:</b> In going through this question, do make sure to consider all scenarios possible regarding the value of $R_0$.

# $[B] = \frac{1}{y} $ <br/>
# 
# $ B(t) = \frac{1}{\frac{1}{[B*]} + \left(\frac{1}{B_0} - \frac{1}{[B*]} \right)e^{t(\gamma - \beta)}} $ <br/> <br/>
# Multiply by $\frac{B*B_0}{B*B_0} $ <br/> <br/>
# $ B(t) = \frac{B*B_0}{B_0 + (B* - Bo)e^{t(\gamma - \beta)}} $
# 
# One possible scenario causing errors in this equation is when $R_0 = 1$ which means $\beta = \gamma$ <br/>
# This means from the expression : <br/>
# $ y(t) = \frac{\frac{\beta}{N}}{\beta - \gamma} + k_1 e^{t(\gamma - \beta)} $ <br/>
# There would be a division by 0 and there would be an error. This means for $R_0$ an equation is needed which excludes $\beta$ and $\gamma$. For this, take the initial $\dot{y}$: <br/>
# $ \dot{y} = (\beta - \gamma)y + \frac{\beta}{N} $ <br/>
# $ \dot{y} = \frac{\beta}{N} $<br/>
# intergrate to: <br/>
# $ y(t) = \frac{\beta}{N}t + c $<br/>
# at $t = 0$ : <br/>
# $y = c$ <br/>
# so the constant is found : <br/>
# $c = \frac{1}{[B]} $  <br/>
# therefore: <br/>
# $ y(t) = \frac{\beta}{N}t + \frac{1}{[B]} $<br/>
# the following expression can be used when $R_0$ = 1: <br/>
# $ B(t) = \left(\frac{\beta}{N}t + \frac{1}{[B]}\right)^{-1} $
# 

# 5. Using different values of $B_0$ (between 1 and N – briefly discuss the case $B_0 = 0$), plot solutions of $[B](t)$ for various values of R0 between 0.1 and 5.0 (with γ = 0.5 for example). Confirm your expression for $[B](t)$ is correct by (a) verifying that it converges to B∗ for large times t and (b) visually confirming agreement when integrating the mean-field equation using Euler (use Python). What happens when $R_0 = 1$? Speculate as to what this means. We will get back to this. For a given value of $R_0$, what happens when the value of $\gamma$ changes? Provide a brief explanation. (11 marks)

# In[3]:


#function for B*
def B_star(N,Ro):
    B_star = N - (N/Ro)
    return B_star
    
#Function for [dBdt]
def func_B(B0,b,g,N,t): # takes entries B0, b beta, g gamma, N and t (timepoint)
    Ro_n = b / g
    B_str = B_star(N,Ro_n)
    exponent = t * (g-b)
    
    Bt = (B_str * B0) / (B0 + (B_str - B0) * np.exp(exponent))
    
    return Bt


N_5 = 1000
time_range = np.linspace(1, 10, 100)
Bo_range = [1,20,50,100,700,1000]

R0_1 = [1,10] #Ro = 0.1, beta = 1, gamma = 10
R0_2 = [1,5] #Ro = 0.2, beta = 1, gamma = 5
R0_3 = [1,1] #Ro = 1, beta = 1, gamma = 1
R0_4 = [3,2] #Ro = 1.5 beta = 3, gamma = 2
R0_5 = [2,1] #Ro = 2, beta = 2, gamma = 1
R0_6 = [5,2] #Ro = 2.5, beta = 5, gamma = 2
R0_7 = [3,1] #Ro = 3, beta = 3, gamma = 1
R0_8 = [7,2] #Ro = 3.5, beta = 7, gamma = 2
R0_9 = [4,1] #Ro = 4, beta = 4, gamma = 1
R0_10 = [5,1] #Ro = 5, beta = 5, gamma = 1

R0_range = {'0.1':R0_1, 
            '0.2':R0_2, 
            '1':R0_3, 
            '1.5':R0_4, 
            '2':R0_5, 
            '2.5':R0_6, 
            '3':R0_7, 
            '3.5':R0_8, 
            '4':R0_9,
            '5':R0_10}

plt.subplots(3,2, figsize=(15,15))

#B0_1 data
B0_v1 = Bo_range[0]
Bt_sols1 = {}
for key1, r001 in R0_range.items():
    Bt_sols1list = []
    for t in time_range:
        Bt_sol1 = func_B(B0_v1,r001[0],r001[1],N_5,t)
        Bt_sols1list.append(Bt_sol1)
    Bt_sols1[key1] = Bt_sols1list
    
plt.subplot(3,2,1)

plt.plot(time_range,Bt_sols1['0.1'],c='yellow', label='Ro = 0.1')
plt.plot(time_range,Bt_sols1['0.2'],c='orchid', label='Ro = 0.2')
plt.plot(time_range,Bt_sols1['1'],c='salmon', label='Ro = 1')
plt.plot(time_range,Bt_sols1['1.5'],c='sienna', label='Ro = 1.5')
plt.plot(time_range,Bt_sols1['2'],c='tomato', label='Ro = 2')
plt.plot(time_range,Bt_sols1['2.5'],c='lightblue', label='Ro = 2.5')
plt.plot(time_range,Bt_sols1['3'],c='palegreen', label='Ro = 3')
plt.plot(time_range,Bt_sols1['3.5'],c='mediumaquamarine', label='Ro = 3.5')
plt.plot(time_range,Bt_sols1['4'],c='tan', label='Ro = 4')
plt.plot(time_range,Bt_sols1['5'],c='darkgray', label='Ro = 5')

plt.xlabel("time")
plt.ylabel("B(t)")
plt.legend(loc=1)
plt.title("B0 = 1")

#B0_2 data
B0_v2 = Bo_range[1]
Bt_sols2 = {}
for key2, r002 in R0_range.items():
    Bt_sols2list = []
    for t in time_range:
        Bt_sol2 = func_B(B0_v2,r002[0],r002[1],N_5,t)
        Bt_sols2list.append(Bt_sol2)
    Bt_sols2[key2] = Bt_sols2list
    
plt.subplot(3,2,2)

plt.plot(time_range,Bt_sols2['0.1'],c='yellow', label='Ro = 0.1')
plt.plot(time_range,Bt_sols2['0.2'],c='orchid', label='Ro = 0.2')
plt.plot(time_range,Bt_sols2['1'],c='salmon', label='Ro = 1')
plt.plot(time_range,Bt_sols2['1.5'],c='sienna', label='Ro = 1.5')
plt.plot(time_range,Bt_sols2['2'],c='tomato', label='Ro = 2')
plt.plot(time_range,Bt_sols2['2.5'],c='lightblue', label='Ro = 2.5')
plt.plot(time_range,Bt_sols2['3'],c='palegreen', label='Ro = 3')
plt.plot(time_range,Bt_sols2['3.5'],c='mediumaquamarine', label='Ro = 3.5')
plt.plot(time_range,Bt_sols2['4'],c='tan', label='Ro = 4')
plt.plot(time_range,Bt_sols2['5'],c='darkgray', label='Ro = 5')

plt.xlabel("time")
plt.ylabel("B(t)")
plt.legend(loc=1)
plt.title("B0 = 20")

#B0_3 data
B0_v3 = Bo_range[2]
Bt_sols3 = {}
for key3, r003 in R0_range.items():
    Bt_sols3list = []
    for t in time_range:
        Bt_sol3 = func_B(B0_v3,r003[0],r003[1],N_5,t)
        Bt_sols3list.append(Bt_sol3)
    Bt_sols3[key3] = Bt_sols3list
    
plt.subplot(3,2,3)
plt.suptitle("B(t) shown for N = 1000",fontsize=16)

plt.plot(time_range,Bt_sols3['0.1'],c='yellow', label='Ro = 0.1')
plt.plot(time_range,Bt_sols3['0.2'],c='orchid', label='Ro = 0.2')
plt.plot(time_range,Bt_sols3['1'],c='salmon', label='Ro = 1')
plt.plot(time_range,Bt_sols3['1.5'],c='sienna', label='Ro = 1.5')
plt.plot(time_range,Bt_sols3['2'],c='tomato', label='Ro = 2')
plt.plot(time_range,Bt_sols3['2.5'],c='lightblue', label='Ro = 2.5')
plt.plot(time_range,Bt_sols3['3'],c='palegreen', label='Ro = 3')
plt.plot(time_range,Bt_sols3['3.5'],c='mediumaquamarine', label='Ro = 3.5')
plt.plot(time_range,Bt_sols3['4'],c='tan', label='Ro = 4')
plt.plot(time_range,Bt_sols3['5'],c='darkgray', label='Ro = 5')

plt.xlabel("time")
plt.ylabel("B(t)")
plt.legend(loc=1)
plt.title("B0 = 50")

#B0_4 data
B0_v4 = Bo_range[3]
Bt_sols4 = {}
for key4, r004 in R0_range.items():
    Bt_sols4list = []
    for t in time_range:
        Bt_sol4 = func_B(B0_v4,r004[0],r004[1],N_5,t)
        Bt_sols4list.append(Bt_sol4)
    Bt_sols4[key4] = Bt_sols4list
    
plt.subplot(3,2,4)

plt.plot(time_range,Bt_sols4['0.1'],c='yellow', label='Ro = 0.1')
plt.plot(time_range,Bt_sols4['0.2'],c='orchid', label='Ro = 0.2')
plt.plot(time_range,Bt_sols4['1'],c='salmon', label='Ro = 1')
plt.plot(time_range,Bt_sols4['1.5'],c='sienna', label='Ro = 1.5')
plt.plot(time_range,Bt_sols4['2'],c='tomato', label='Ro = 2')
plt.plot(time_range,Bt_sols4['2.5'],c='lightblue', label='Ro = 2.5')
plt.plot(time_range,Bt_sols4['3'],c='palegreen', label='Ro = 3')
plt.plot(time_range,Bt_sols4['3.5'],c='mediumaquamarine', label='Ro = 3.5')
plt.plot(time_range,Bt_sols4['4'],c='tan', label='Ro = 4')
plt.plot(time_range,Bt_sols4['5'],c='darkgray', label='Ro = 5')

plt.xlabel("time")
plt.ylabel("B(t)")
plt.legend(loc=1)
plt.title("B0 = 100")

#B0_5 data
B0_v5 = Bo_range[4]
Bt_sols5 = {}
for key5, r005 in R0_range.items():
    Bt_sols5list = []
    for t in time_range:
        Bt_sol5 = func_B(B0_v5,r005[0],r005[1],N_5,t)
        Bt_sols5list.append(Bt_sol5)
    Bt_sols5[key5] = Bt_sols5list
    
plt.subplot(3,2,5)

plt.plot(time_range,Bt_sols5['0.1'],c='yellow', label='Ro = 0.1')
plt.plot(time_range,Bt_sols5['0.2'],c='orchid', label='Ro = 0.2')
plt.plot(time_range,Bt_sols5['1'],c='salmon', label='Ro = 1')
plt.plot(time_range,Bt_sols5['1.5'],c='sienna', label='Ro = 1.5')
plt.plot(time_range,Bt_sols5['2'],c='tomato', label='Ro = 2')
plt.plot(time_range,Bt_sols5['2.5'],c='lightblue', label='Ro = 2.5')
plt.plot(time_range,Bt_sols5['3'],c='palegreen', label='Ro = 3')
plt.plot(time_range,Bt_sols5['3.5'],c='mediumaquamarine', label='Ro = 3.5')
plt.plot(time_range,Bt_sols5['4'],c='tan', label='Ro = 4')
plt.plot(time_range,Bt_sols5['5'],c='darkgray', label='Ro = 5')

plt.xlabel("time")
plt.ylabel("B(t)")
plt.legend(loc=1)
plt.title("B0 = 700")

#B0_6 data
B0_v6 = Bo_range[5]
Bt_sols6 = {}
for key6, r006 in R0_range.items():
    Bt_sols6list = []
    for t in time_range:
        Bt_sol6 = func_B(B0_v6,r006[0],r006[1],N_5,t)
        Bt_sols6list.append(Bt_sol6)
    Bt_sols6[key6] = Bt_sols6list
    
plt.subplot(3,2,6)

plt.plot(time_range,Bt_sols6['0.1'],c='yellow', label='Ro = 0.1')
plt.plot(time_range,Bt_sols6['0.2'],c='orchid', label='Ro = 0.2')
plt.plot(time_range,Bt_sols6['1'],c='salmon', label='Ro = 1')
plt.plot(time_range,Bt_sols6['1.5'],c='sienna', label='Ro = 1.5')
plt.plot(time_range,Bt_sols6['2'],c='tomato', label='Ro = 2')
plt.plot(time_range,Bt_sols6['2.5'],c='lightblue', label='Ro = 2.5')
plt.plot(time_range,Bt_sols6['3'],c='palegreen', label='Ro = 3')
plt.plot(time_range,Bt_sols6['3.5'],c='mediumaquamarine', label='Ro = 3.5')
plt.plot(time_range,Bt_sols6['4'],c='tan', label='Ro = 4')
plt.plot(time_range,Bt_sols6['5'],c='darkgray', label='Ro = 5')

plt.xlabel("time")
plt.ylabel("B(t)")
plt.legend(loc=1)
plt.title("B0 = 1000")


plt.tight_layout()
plt.show()


# The diagrams above are shown to converge at $B*$ $(N-N/R_0)$, for example:
# - On the first subplot $B_0 = 1$, for $R_0 = 1.5$, $B* = 1000 - 1000/1.5 = 333.33$ and the brown line for $Ro = 1.5$ reaches 333 for B(t) at larger time.
# - When $R_0 = 1$, B(t) = 0. It is likely that for this value of $R_0$, the B(t) remains in the other equilibria ($B*$) (N,0)
# - For any given value of $R_0$, if $\gamma$  increases, $R_0$ will be smaller whereas $R_0$ will increase with smaller values of $\gamma$. Thus, with smaller $R_0$ and  larger $\gamma$, the B(t) is will converge to the alterntive ($B*$) (N,0). Contrastingly, larger $R_0$ and  smaller $\gamma$ gamma means B(t) will converge to $(N-N/R_0)$.

# ### Simulation Work

# 1. Explore the behaviour of the system when considering suitably chosen scenarios, i.e., focus on the limit cases (e.g., small $R_0$, large $R_0$ and $R_0 = 1$; small $N$, large $N$; small $B_0$, large $B_0$). For each scenario, use the code provided to generate many realisations of the stochastic process. Plot all realisations on a single plot. Make relevant qualitative observations. (10 marks)

# In[4]:


# sorry in advance for long code, there wasn't enough time for me to optimize with loops 

# gillespie
def gillespie_ABA(N,B0,beta,gamma,Tmax):

    A=[N-B0] # We cannot predict how many elements there will be unfortunately
    B=[B0]
    T=[0] 
    state = np.random.permutation([0]*(N-B0)+[1]*B0) # Randomly allocate B0 individuals to have state B (state=1), A (state=0) otherwise 
    B_contacts = np.where(state==1)[0] # Index of individuals in state B (state=1).
    rate_vector = B0*beta*np.ones((N,1))/N # Set rates to be B0*beta/N (rate for individuals in state A) to all individuals (initialisation). 
    rate_vector[B_contacts] = gamma # Update rate of B_contacts to be gamma (the rate for individuals in state B)
    
    time = 0
    while time<=Tmax+0.5: # some (arbitrary) buffer after Tmax
        rate = np.sum(rate_vector) # Total rate (refer to Gillespie algorithm for details)
        cumrate = np.cumsum(rate_vector) # Cumulated sum of rates
        if rate > 0.000001: # if rate is sufficiently large
            tstep = expovariate(rate) # Pick an exponentially distributed time. Beware of difference with exprnd in Matlab where it is 1/rate
            T.append(T[-1]+tstep) # Time of next event
            event = np.where(cumrate>np.random.rand()*rate)[0][0] # Find which individual will see its state change 
            if state[event]==0: # individual is in state A 
                A.append(A[-1]-1) # this state A individual becomes state B so number of state A individuals is decreased
                B.append(B[-1]+1) # obviously, number of state B individuals is increased 
                state[event] = 1 # Update state vector
                rate_vector[event] = gamma # Change rate of individual to B->A rate, namely gamma
                A_contacts = np.where(state==0)[0] # List of state A individuals after change
                rate_vector[A_contacts] += beta/N # Update rate of state A individuals to account for the extra state B individual
            else: # individual is in state B
                B.append(B[-1]-1) # this state B individual becomes state A so number of state B individuals is decreased
                A.append(A[-1]+1) # obviously, number of state A individuals is increased
                state[event] = 0 # Update state vector
                A_contacts = np.where(state==0)[0] # List of state A individuals after changes                                
                rate_vector[A_contacts] = beta*len(np.where(state==1)[0])/N # Update rate of state A individuals based on number of B individuals  
        else: # Nothing will happen from now on so we can accelerate the process
            time = T[-1] # current time
            while time <= Tmax + 0.5:
                A.append(A[-1]) # Just keep things as they are
                B.append(B[-1])
                T.append(T[-1]+0.5) # arbitrarily add 0.5 to clock
                time = T[-1]
        # Update time and proceed with loop 
        time = T[-1]         

    return T,A,B    

#values for small R0
small_Ro = 0.01 #beta is 1, gamma is 100
small_Robeta = 1
small_Rogamma = 100

#values for big Ro
big_Ro = 2 #beta is 2, gamma is 1
big_Robeta = 2
big_Rogamma = 1

#value for R = 1
Ro_one = 1 #beta is 1, gamma is 1
one_Robeta = 1
one_Rogamma = 1

#values for N
small_N = 100
big_N = 1000

#values for Bo
small_Bo = 10
big_Bo_1 = 25
big_Bo_2 = 300

# Sim 1: N = 100, R0 = 0.01, Bo = 10
sim_1 = gillespie_ABA(small_N, small_Bo, small_Robeta, small_Rogamma, 60)

# Sim 2: N = 100, R0 = 2, Bo = 10
sim_2 = gillespie_ABA(small_N, small_Bo, big_Robeta, big_Rogamma, 60)

# Sim 3: N = 100, R0 = 1, Bo = 10
sim_3 = gillespie_ABA(small_N, small_Bo, one_Robeta, one_Rogamma, 60)

# Sim 4: N = 100, R0 = 0.01, Bo = 25
sim_4 = gillespie_ABA(small_N, big_Bo_1, small_Robeta, small_Rogamma, 60)

# Sim 5: N = 100, R0 = 2, Bo = 25
sim_5 = gillespie_ABA(small_N, big_Bo_1, big_Robeta, big_Rogamma, 60)

# Sim 6: N = 100, R0 = 1, Bo = 25
sim_6 = gillespie_ABA(small_N, big_Bo_1, one_Robeta, one_Rogamma, 60)

# Sim 7: N = 1000, R0 = 0.01, Bo = 10
sim_7 = gillespie_ABA(big_N, small_Bo, small_Robeta, small_Rogamma, 60)

# Sim 8: N = 1000, R0 = 2, Bo = 10
sim_8 = gillespie_ABA(big_N, small_Bo, big_Robeta, big_Rogamma, 60)

# Sim 9: N = 1000, R0 = 1, Bo = 10
sim_9 = gillespie_ABA(big_N, small_Bo, one_Robeta, one_Rogamma, 60)

# Sim 10: N = 1000, R0 = 0.01, Bo = 300
sim_10 = gillespie_ABA(big_N, big_Bo_2, small_Robeta, small_Rogamma, 60)

# Sim 11: N = 1000, R0 = 2, Bo = 300
sim_11 = gillespie_ABA(big_N, big_Bo_2, big_Robeta, big_Rogamma, 60)

# Sim 12: N = 1000, R0 = 1, Bo = 300
sim_12 = gillespie_ABA(big_N, big_Bo_2, one_Robeta, one_Rogamma, 60)

plt.subplots(2,1, figsize=(15,15))

#Plot 1 N=100
plt.subplot(2,1,1)
plt.plot(sim_1[0],sim_1[1],c='purple')
plt.plot(sim_1[0],sim_1[2],scaley=True,c='darkmagenta')
plt.plot(sim_2[0],sim_2[1],scaley=True,c='dodgerblue')
plt.plot(sim_2[0],sim_2[2],scaley=True,c='forestgreen')
plt.plot(sim_3[0],sim_3[1],scaley=True,c='steelblue')
plt.plot(sim_3[0],sim_3[2],scaley=True,c='teal')
plt.plot(sim_4[0],sim_4[1],scaley=True,c='yellowgreen')
plt.plot(sim_4[0],sim_4[2],scaley=True,c='mediumslateblue')
plt.plot(sim_5[0],sim_5[1],scaley=True,c='gold')
plt.plot(sim_5[0],sim_5[2],scaley=True,c='wheat')
plt.plot(sim_6[0],sim_6[1],scaley=True,c='gold')
plt.plot(sim_6[0],sim_6[2],scaley=True,c='wheat')
plt.legend(["State A, R0 = 0.01, B0 = 10",
                     "State B, R0 = 0.01, B0 = 10",
                     "State A, R0 = 2.0, B0 = 10",
                     "State B, R0 = 2.0, B0 = 10",
                     "State A, R0 = 1.0, B0 = 10",
                     "State B, R0 = 1.0, B0 = 10",
                     "State A, R0 = 0.01, B0 = 25",
                     "State B, R0 = 0.01, B0 = 25",
                     "State A, R0 = 2.0, B0 = 25",
                     "State B, R0 = 2.0, B0 = 25",
                     "State A, R0 = 1.0, B0 = 25",
                     "State B, R0 = 1.0, B0 = 25"
                    ],loc='upper right',fancybox=True)

plt.title("Gillespie for System: small N, N = 100")
plt.xlabel("Time")
plt.ylabel("No. of Individuals")

#Plot 2 N=1000
plt.subplot(2,1,2)

plt.plot(sim_7[0],sim_7[1],c='yellow')
plt.plot(sim_7[0],sim_7[2],c='sandybrown')
plt.plot(sim_8[0],sim_8[1],c='salmon')
plt.plot(sim_8[0],sim_8[2],c='burlywood')
plt.plot(sim_9[0],sim_9[1],c='lightpink')
plt.plot(sim_9[0],sim_9[2],c='darkred')
plt.plot(sim_10[0],sim_10[1],c='darkorange')
plt.plot(sim_11[0],sim_11[1],c='pink')
plt.plot(sim_11[0],sim_11[2],c='orange')
plt.plot(sim_12[0],sim_12[1],c='lightblue')
plt.plot(sim_12[0],sim_12[2],c='peachpuff')
plt.legend(["State A, R0 = 0.01, B0 = 10",
                     "State B, R0 = 0.01, B0 = 10",
                     "State A, R0 = 2.0, B0 = 10",
                     "State B, R0 = 2.0, B0 = 10",
                     "State A, R0 = 1.0, B0 = 10",
                     "State B, R0 = 1.0, B0 = 10",
                     "State A, R0 = 0.01, B0 = 300",
                     "State B, R0 = 0.01, B0 = 300",
                     "State A, R0 = 2.0, B0 = 300",
                     "State B, R0 = 2.0, B0 = 300",
                     "State A, R0 = 1.0, B0 = 300",
                     "State B, R0 = 1.0, B0 = 300"
                    ],loc='upper right',fancybox=True)

plt.title("Gillespie for System: large N, N = 1000")
plt.xlabel("Time")
plt.ylabel("No. of Individuals")


plt.show()


# The plots above show the behaviour, in terms of number of individuals in either state $A$ or $B$ over time, as an average from the gillespie algorithm which runs 10 repeats per time point. The total number of individuals $N$ remains constant, but for the top plot $N=100$ and for the bottom plot $N=1000$. <br/>
# These plots show: </br>
# - the size of $N$ does not appear to affect the behaviour of the system since the trajectories have similar shapes for the same variety of parameters. Parameters values leading to oscillatory behaviour take longer to reach such behaviour for $N = 1000$ but this is down to the increased scale of the system.
# - In both systems, small $R_0$ (0.001) leads to a fast equilibrium $ ([A],[B]) = (N,0)$
# - In both systems, $R_0$ (1.0) leads to a a slower transition to equilibrium $ ([A],[B]) = (N,0)$
# - These equilibria are reached with both small and large values of $B_0$, therefore in this system it can be speculated that when $ 0 < R_0$ ≤  $1 $ the trajectories will inevitably reach an $(N,0)$ equilibria, regardless of the starting number of individuals in state $B$.
# - Simulating a higher $R_0$ (2.0), the number of individuals in states $A$ and $B$ continue to oscillate.

# 2. For each scenario, calculate the average (and standard deviation) of the realisations. Here, you are going to face a problem linked with the nota bene from the introductory paragraph. You will need to think of a solution and implement it. Superimpose the average (and error bars) to the realisations. Use a larger line width for visibility. (11 marks)

# In[7]:


from scipy.interpolate import interp1d
from scipy.integrate import quad

common_time = np.linspace(1, 50, 500)

def repeat_gillespie(repeats,N,B0,beta,gamma,Tmax):
    
    interpolated_A = []
    interpolated_B = []
    repeat_no = 1
    
    while repeat_no < repeats:
        run = gillespie_ABA(N,B0,beta,gamma,Tmax)
        run_A = interp1d(run[0],run[1], 'linear', fill_value='extrapolate')
        run_B = interp1d(run[0],run[2], 'linear', fill_value='extrapolate')
        
        new_A = run_A(common_time)
        new_B = run_B(common_time)
        interpolated_A.append(new_A)
        interpolated_B.append(new_B)
        
        repeat_no += 1
    
    interpolated_A_np = np.array(interpolated_A)
    interpolated_B_np = np.array(interpolated_B)
    
  
    
    mean_A, std_A = np.mean(interpolated_A_np, axis=0), np.std(interpolated_A_np,axis=0)
    mean_B, std_B = np.mean(interpolated_B_np,axis=0), np.std(interpolated_B_np,axis=0)

    return [mean_A,std_A],[mean_B,std_B]

plt.subplots(4,1, figsize=(20,30))

sim1_int = repeat_gillespie(10, small_N, small_Bo, small_Robeta, small_Rogamma, 60)
sim1_A_int, sim1_B_int = sim1_int[0],sim1_int[1]

sim2_int = repeat_gillespie(10, small_N, small_Bo, big_Robeta, big_Rogamma, 60)
sim2_A_int, sim2_B_int = sim2_int[0],sim2_int[1]

sim3_int = repeat_gillespie(10, small_N, small_Bo, one_Robeta, one_Rogamma, 60)
sim3_A_int, sim3_B_int = sim3_int[0],sim3_int[1]

sim4_int = repeat_gillespie(10, small_N, big_Bo_1, small_Robeta, small_Rogamma, 60)
sim4_A_int, sim4_B_int = sim4_int[0],sim4_int[1]

sim5_int = repeat_gillespie(10, small_N, big_Bo_1, big_Robeta, big_Rogamma, 60)
sim5_A_int, sim5_B_int = sim5_int[0],sim5_int[1]

sim6_int = repeat_gillespie(10, small_N, big_Bo_1, one_Robeta, one_Rogamma, 60)
sim6_A_int, sim6_B_int = sim6_int[0],sim6_int[1]

sim7_int = repeat_gillespie(10, big_N, small_Bo, small_Robeta, small_Rogamma, 60)
sim7_A_int, sim7_B_int = sim7_int[0],sim7_int[1]

sim8_int = repeat_gillespie(10, big_N, small_Bo, big_Robeta, big_Rogamma, 60)
sim8_A_int, sim8_B_int = sim8_int[0],sim8_int[1]

sim9_int = repeat_gillespie(10, big_N, small_Bo, one_Robeta, one_Rogamma, 60)
sim9_A_int, sim9_B_int = sim9_int[0],sim9_int[1]

sim10_int = repeat_gillespie(10, big_N, big_Bo_2, small_Robeta, small_Rogamma, 60)
sim10_A_int, sim10_B_int = sim10_int[0],sim10_int[1]

sim11_int = repeat_gillespie(10, big_N, big_Bo_2, big_Robeta, big_Rogamma, 60)
sim11_A_int, sim11_B_int = sim11_int[0],sim11_int[1]

sim12_int = repeat_gillespie(10, big_N, big_Bo_2, one_Robeta, one_Rogamma, 60)
sim12_A_int, sim12_B_int = sim12_int[0],sim12_int[1]


#Subplot 1: State As N = 100
plt.subplot(4,1,1)
plt.errorbar(common_time,sim1_A_int[0],sim1_A_int[1],label="R0 = 0.01, Bo = 10",color='black',ecolor='lightgray')
plt.errorbar(common_time,sim2_A_int[0],sim2_A_int[1],label="R0 = 2.0, Bo = 10",color='black',ecolor='blue')
plt.errorbar(common_time,sim3_A_int[0],sim3_A_int[1],label="R0 = 1.0, Bo = 10",color='black',ecolor='forestgreen')
plt.errorbar(common_time,sim4_A_int[0],sim4_A_int[1],label="R0 = 0.01, Bo = 25",color='black',ecolor='deepskyblue')
plt.errorbar(common_time,sim5_A_int[0],sim5_A_int[1],label="R0 = 2.0, Bo = 25",color='black',ecolor='lightsteelblue')
plt.errorbar(common_time,sim6_A_int[0],sim6_A_int[1],label="R0 = 1.0, Bo = 25",color='black',ecolor='mediumseagreen')

plt.xlabel("time")
plt.ylabel("no. of individuals")
plt.legend(loc=1)
plt.title("State A, N=100")

#Subplot 2: State Bs N = 100
plt.subplot(4,1,2)
plt.errorbar(common_time,sim1_B_int[0],sim1_B_int[1],label="R0 = 0.01, Bo = 10",color='black',ecolor='pink')
plt.errorbar(common_time,sim2_B_int[0],sim2_B_int[1],label="R0 = 2.0, Bo = 10",color='black',ecolor='red')
plt.errorbar(common_time,sim3_B_int[0],sim3_B_int[1],label="R0 = 1.0, Bo = 10",color='black',ecolor='tomato')
plt.errorbar(common_time,sim4_B_int[0],sim4_B_int[1],label="R0 = 0.01, Bo = 25",color='black',ecolor='orange')
plt.errorbar(common_time,sim5_B_int[0],sim5_B_int[1],label="R0 = 2.0, Bo = 25",color='black',ecolor='lightcoral')
plt.errorbar(common_time,sim6_B_int[0],sim6_B_int[1],label="R0 = 1.0, Bo = 25",color='black',ecolor='deeppink')

plt.xlabel("time")
plt.ylabel("no. of individuals")
plt.legend(loc=1)
plt.title("State B, N=100")


#Subplot 3: State As N = 1000
plt.subplot(4,1,3)
plt.errorbar(common_time,sim7_A_int[0],sim7_A_int[1],label="R0 = 0.01, Bo = 10",color='black',ecolor='yellow')
plt.errorbar(common_time,sim8_A_int[0],sim8_A_int[1],label="R0 = 2.0, Bo = 10",color='black',ecolor='orange')
plt.errorbar(common_time,sim9_A_int[0],sim9_A_int[1],label="R0 = 1.0, Bo = 10",color='black',ecolor='burlywood')
plt.errorbar(common_time,sim10_A_int[0],sim10_A_int[1],label="R0 = 0.01, Bo = 300",color='black',ecolor='sienna')
plt.errorbar(common_time,sim11_A_int[0],sim11_A_int[1],label="R0 = 2.0, Bo = 300",color='black',ecolor='saddlebrown')
plt.errorbar(common_time,sim12_A_int[0],sim12_A_int[1],label="R0 = 1.0, Bo = 300",color='black',ecolor='indianred')

plt.xlabel("time")
plt.ylabel("no. of individuals")
plt.legend(loc=1)
plt.title("State A, N=1000")


#Subplot 3: State Bs N = 1000
plt.subplot(4,1,4)
plt.errorbar(common_time,sim7_B_int[0],sim7_B_int[1],label="R0 = 0.01, Bo = 10",color='black',ecolor='purple')
plt.errorbar(common_time,sim8_B_int[0],sim8_B_int[1],label="R0 = 2.0, Bo = 10",color='black',ecolor='thistle')
plt.errorbar(common_time,sim9_B_int[0],sim9_B_int[1],label="R0 = 1.0, Bo = 10",color='black',ecolor='lightpink')
plt.errorbar(common_time,sim10_B_int[0],sim10_B_int[1],label="R0 = 0.01, Bo = 300",color='black',ecolor='orchid')
plt.errorbar(common_time,sim11_B_int[0],sim11_B_int[1],label="R0 = 2.0, Bo = 300",color='black',ecolor='mediumslateblue')
plt.errorbar(common_time,sim12_B_int[0],sim12_B_int[1],label="R0 = 1.0, Bo = 300",color='black',ecolor='darkmagenta')

plt.xlabel("time")
plt.ylabel("no. of individuals")
plt.legend(loc=1)
plt.title("State B, N=1000")

plt.show()


# 3. Finally, superimpose the mean-field solution. Again, use a larger line width and different colour for visibility. Describe and interpret agreement between average of stochastic realisations and mean-field in relation to the choice of parameters. In this question, using $B_0 = 1$ (i.e., only one individual in state B at $t = 0$) can help exacerbate the differences and help you think about what is happening. You may want to refer to your bifurcation plot. (8 marks)

# For the state $B$ diagrams above, the mean field equation will be superimposed for the same parameter and variable values used in the Gillespie.

# In[9]:



#function for mean_field
def mean_field(B0,t,parameters): # parameters [N,beta,gamma]
    N = parameters[0]
    beta = parameters[1]
    gamma = parameters[2]
    
    B_dot = beta * (B0/N) * (N - B0) - (gamma*B0)
    return B_dot 

#Gillespie and meanfield run in one function for same paramters
def repeat_gillespie_mf(repeats,N,B0,beta,gamma,Tmax):
    
    interpolated_B = []
    mean_fields = []
    
    repeat_no = 1
    
    while repeat_no < repeats:
        run = gillespie_ABA(N,B0,beta,gamma,Tmax)
        run_B = interp1d(run[0],run[2], 'linear', fill_value='extrapolate')
        
        new_B = run_B(common_time)
        interpolated_B.append(new_B)
        
        repeat_no += 1
        
    params_mf = [N,beta,gamma]
    mean_field_B = odeint(mean_field,B0,common_time,args=(params_mf,))
    mean_field_B_result = mean_field_B[:,0]
    
    interpolated_B_np = np.array(interpolated_B)
    
    mean_B, std_B = np.mean(interpolated_B_np,axis=0), np.std(interpolated_B_np,axis=0)

    return [mean_B,std_B], [mean_field_B_result]

#Run and plot results
sim1_sup = repeat_gillespie_mf(10, small_N, small_Bo, small_Robeta, small_Rogamma, 60)
sim1_B_sup, sim1_B_mf = sim1_sup[0],sim1_sup[1]

sim2_sup = repeat_gillespie_mf(10, small_N, small_Bo, big_Robeta, big_Rogamma, 60)
sim2_B_sup, sim2_B_mf = sim2_sup[0],sim2_sup[1]

sim3_sup = repeat_gillespie_mf(10, small_N, small_Bo, one_Robeta, one_Rogamma, 60)
sim3_B_sup, sim3_B_mf = sim3_sup[0],sim3_sup[1]

sim4_sup = repeat_gillespie_mf(10, small_N, big_Bo_1, small_Robeta, small_Rogamma, 60)
sim4_B_sup, sim4_B_mf = sim4_sup[0],sim4_sup[1]

sim5_sup = repeat_gillespie_mf(10, small_N, big_Bo_1, big_Robeta, big_Rogamma, 60)
sim5_B_sup, sim5_B_mf = sim5_sup[0],sim5_sup[1]

sim6_sup = repeat_gillespie_mf(10, small_N, big_Bo_1, one_Robeta, one_Rogamma, 60)
sim6_B_sup, sim6_B_mf = sim6_sup[0],sim6_sup[1]

sim7_sup = repeat_gillespie_mf(10, big_N, small_Bo, small_Robeta, small_Rogamma, 60)
sim7_B_sup, sim7_B_mf = sim7_sup[0],sim7_sup[1]

sim8_sup = repeat_gillespie_mf(10, big_N, small_Bo, big_Robeta, big_Rogamma, 60)
sim8_B_sup, sim8_B_mf = sim8_int[0],sim8_sup[1]

sim9_sup = repeat_gillespie_mf(10, big_N, small_Bo, one_Robeta, one_Rogamma, 60)
sim9_B_sup, sim9_B_mf = sim9_sup[0],sim9_sup[1]

sim10_sup = repeat_gillespie_mf(10, big_N, big_Bo_2, small_Robeta, small_Rogamma, 60)
sim10_B_sup, sim10_B_mf = sim10_sup[0],sim10_sup[1]

sim11_sup = repeat_gillespie_mf(10, big_N, big_Bo_2, big_Robeta, big_Rogamma, 60)
sim11_B_sup, sim11_B_mf = sim11_sup[0],sim11_sup[1]

sim12_sup = repeat_gillespie_mf(10, big_N, big_Bo_2, one_Robeta, one_Rogamma, 60)
sim12_B_sup, sim12_B_mf = sim12_sup[0],sim12_sup[1]

#plot
plt.subplots(2,1, figsize=(20,20))

plt.subplot(2,1,1)

# N = 100
plt.errorbar(common_time,sim1_B_sup[0],sim1_B_sup[1],label="R0 = 0.01, Bo = 10",color='black',ecolor='pink')
plt.errorbar(common_time,sim2_B_sup[0],sim2_B_sup[1],label="R0 = 2.0, Bo = 10",color='black',ecolor='red')
plt.errorbar(common_time,sim3_B_sup[0],sim3_B_sup[1],label="R0 = 1.0, Bo = 10",color='black',ecolor='tomato')
plt.errorbar(common_time,sim4_B_sup[0],sim4_B_sup[1],label="R0 = 0.01, Bo = 25",color='black',ecolor='orange')
plt.errorbar(common_time,sim5_B_sup[0],sim5_B_sup[1],label="R0 = 2.0, Bo = 25",color='black',ecolor='lightcoral')
plt.errorbar(common_time,sim6_B_sup[0],sim6_B_sup[1],label="R0 = 1.0, Bo = 25",color='black',ecolor='deeppink')
plt.legend(loc=1)

plt.errorbar(common_time,sim1_B_mf[0],label="mean field",color='blue')
plt.errorbar(common_time,sim2_B_mf[0],color='blue')
plt.errorbar(common_time,sim3_B_mf[0],color='blue')
plt.errorbar(common_time,sim4_B_mf[0],color='blue')
plt.errorbar(common_time,sim5_B_mf[0],color='blue')
plt.errorbar(common_time,sim6_B_mf[0],color='blue')

plt.legend(loc=2)
plt.xlabel("time")
plt.ylabel("no. of individuals")
plt.title("State B, N=100")

plt.subplot(2,1,2)
# N = 100
plt.errorbar(common_time,sim7_B_sup[0],sim7_B_sup[1],label="R0 = 0.01, Bo = 10",color='black',ecolor='purple')
plt.errorbar(common_time,sim8_B_sup[0],sim8_B_sup[1],label="R0 = 2.0, Bo = 10",color='black',ecolor='thistle')
plt.errorbar(common_time,sim9_B_sup[0],sim9_B_sup[1],label="R0 = 1.0, Bo = 10",color='black',ecolor='lightpink')
plt.errorbar(common_time,sim10_B_sup[0],sim10_B_sup[1],label="R0 = 0.01, Bo = 300",color='black',ecolor='orchid')
plt.errorbar(common_time,sim11_B_sup[0],sim11_B_sup[1],label="R0 = 2.0, Bo = 300",color='black',ecolor='mediumslateblue')
plt.errorbar(common_time,sim12_B_sup[0],sim12_B_sup[1],label="R0 = 1.0, Bo = 300",color='black',ecolor='darkmagenta')
plt.legend(loc=1)

# N = 1000
plt.errorbar(common_time,sim7_B_mf[0],label="mean field",color='red')
plt.errorbar(common_time,sim8_B_mf[0],color='red')
plt.errorbar(common_time,sim9_B_mf[0],color='red')
plt.errorbar(common_time,sim10_B_mf[0],color='red')
plt.errorbar(common_time,sim11_B_mf[0],color='red')
plt.errorbar(common_time,sim12_B_mf[0],color='red')

plt.xlabel("time")
plt.ylabel("no. of individuals")
plt.legend(loc=2)
plt.title("State B, N=1000")
plt.show()


# The main difference seen with the Gillespie simualtion is that within the Gillespie there is a stochastic conversion of individuals from states $[A]$ and $[B]$ , known as a <i> jump process </i>, seen in the turbulence average lines across time. This is likely to be a better representation of real life scenarios and randomness of states. The mean field does not include the same randomness.

# 4. [Slightly challenging question]: Consider 100 replications for N = 1000, β = 0.51, γ = 0.5 and 100 replications for N = 1000, β = 0.95, γ = 0.5. You should notice a substantial difference in agreement between the mean-field and the av- erage of the stochastic realisations depending on which scenario is considered. How could you improve agreement for the scenario with the poorest agreement. Please note: The difference in B∗ is not the quantity of interest here. Rather you should think about why agreement is so poor. This does not actually involve analytical work. An excellent answer would see you implement your proposed solution and provide evidence of improved agreement. (12 marks)

# For β = 0.51, γ = 0.5, R0 would equal 0.98 , $R_0$ would be close to 1 so for the meanfield, the trajectory would reach the (N,0) equilbria. <br/>
# For β = 0.95, γ = 0.5, R0 would equal 1.9, $R_0$ would be close to 2 so for the meanfield, the trajectory would reaches the other $B*$, $N-N/Ro$.
#  <br/>
# The reason these two scenarios reach different $B*$ is that from the original equations the number of individuals in state $B$ is proportional to itself but there is always a consistent reduction in $[B]$ at the rate of gamma. Therefore, for β = 0.51, γ = 0.5, $R_0$ is not increasing the value of $[B]$ higher than the rate at which $[B]$  is lost. <br/>
# For β = 0.95, γ = 0.5,  $R_0$ is increasing $[B]$ at a faster rate, and since the value of gamma is proportionally smaller than beta, the increasing $[B]$ rate is faster than the rate $[B]$ is reduced due to gamma. Since N is constant, eventually the trajectory reaches the alternative equilibria $N-N/Ro$.

# In[13]:


#Run mean scenario 4.1  
q4_sim_1 = repeat_gillespie_mf(100, 1000, 300, 0.51, 0.5, 60)
q4_sim_1_B, q4_sim_1_mf = q4_sim_1[0],q4_sim_1[1]

#Run mean scenario 4.2 
q4_sim_2 = repeat_gillespie_mf(100, 1000, 300, 0.95, 0.5, 60)
q4_sim_2_B, q4_sim_2_mf = q4_sim_2[0],q4_sim_2[1]

plt.errorbar(common_time,q4_sim_1_B[0],q4_sim_1_B[1],label="β = 0.51, γ = 0.5",color='black',ecolor='slategrey')
plt.errorbar(common_time,q4_sim_2_B[0],q4_sim_2_B[1],label="β = 0.95, γ = 0.5",color='black',ecolor='palegreen')
plt.errorbar(common_time,q4_sim_1_mf[0],label="mean field: β = 0.51, γ = 0.5 ",color='blue')
plt.errorbar(common_time,q4_sim_2_mf[0],label="mean field: β = 0.95, γ = 0.5 ",color='green')

plt.legend(bbox_to_anchor=(1.1, 1.05))
plt.xlabel("time")
plt.ylabel("no. of individuals")
plt.title("Number of Individuals in State B, N=1000, R0=300")

plt.show()


# ## Critical Thinking

# 1. So far the brief has provided no information whatsoever regarding what the states $A$ and $B$ are and what the individuals are. Thinking about what is happening in this system, provide at least one example of real-world scenario to which this model could apply. Bonus points will be given for any answer that provides two examples, one in which the equilibria are of interest and the other in which the critical regime (when $R_0 = 1$) is of interest. Either way, what is the benefit of being able to model the system? (8 marks)

#   <b> Example 1) </b> <br/>
#    Individuals are neurons either in either in excitatory or non-polarized state.
#    - $N$ is the total number of neurons in the tissue of interest, for example part of the frontal cortex.
#    - $[B]$ is the number of neurons in the tissue in excitatory state.
#    - $[\dot{B}]$ is the rate of change of neurons into excitatory state.
#    - $[A]$ is the number of neurons in the tissue in non-polarized state.
#    - $[\dot{A}]$ is the rate of change of neurons into a non-polarized state. <br/>
#    As excitatory neurons excite other neurons, so $[B]$ would increase with respect to itself. <br/>
#    Neurons often fire spontaneously, so gamma $\gamma$ could represent rate of spontaneous firing. <br/>
#    Beta $\beta$ could represent intensity of a particular stimulus, for example a light stimulus to the eyes.
#    
#    Equibrilia in neuron excitability, e.g a certain number of of neurons remaining excited for a significant amount of time, could indicate something is happening within the brain e.g the subject is conscious or in a sleeping state. 
#    
#  <b> Example 2) </b> <br/>
#  Individuals are people either infected or not infected (recovered) from a transmissible disease.
#  - $N$ is the total number of people in the population
#  - $[B]$ is the number of people with the transmissible disease.
#  - $[\dot{B}]$ is the rate of change of people into having the disease.
#  - $[A]$ is the number of people not having the transmissible disease.
#  - $[\dot{A}]$ is the rate of recovery from the disease. 
# <br/>
# $\beta$ could represent some tranmission rate related to proximity of individuals in the system. <br/>
# $\gamma$ represents some given probability that a person will get the disease independent of population proximity.
#     
#   <br/><br/>
#   The benefits of modelling a system this way are to try and identify the underlying patterns in behaviour of signals or trajectories that are oscillating or have no clear signal or structure under a certain measurements overtime, and to find the relationships between parameters of a system. <br/>
#   Using the neuronal example, studying the system analytically and computationally can help identfy underlying brain activity signals and their behaviour when the initial signal measurement, for example EEG measurements, are oscillating overtime due to noise in the system, such as spontaneous firing. <br/> Furthermore, for a system which is simulated, we can adjust parameters values to see what results in the behaviour of the variables which can be vary valuable in forecasting or risk evaluation.
# 

# 2. [Very tough question]: The model provided implicitly assumes that all individuals are potentially in contact with each other. What would be a more likely scenario? What changes would have to be made to the code of the Gillespie algorithm in order to include such a scenario? If you are able to do this, do it. Then, speculate as to what could affect the results observed in the previous questions. If you feel so inclined, demonstrate it experimentally. NB: Only 10 marks have been given to this question. However, anyone managing it successfully would receive an extra 10 marks for the assignment (with the total mark capped to 100 obviously).

# A more likely scenario is that individuals in the population N associate in clusters. This would mean individuals within a cluster would have, for example, a high parameter representing close proximity (high beta) and an additional low proximity for individuals within another cluster. <br/>
# 
# - The same equations could be used, however the gillespie would have to take in an additional parameter $C$ where $C$ is the number of clusters in population $N$. 
# - For a simple model (althought different distributions could be examined) all clusters would have approximately the same number of individuals in. 
# - There would be an additional term in the equations $ +/-\omega[B] $ representing the number of individuals converting into state $[B]$ from another cluster. 
# - The Gillespie algorithm would have to include this term, then run averages for each cluster per timepoint and sum the total number of individuals in state $[B]$ (or $[A]$) over each the time point and average these.
# - The Gillespie algorithm would also allocate random jumping points both, within clusters and different numbers of clusters.
# - For advancing Gillespie algorithm further for this scenario, it could randomly also adjust the individuals in each cluster on every repeat which would better replicate real life simulations of changing groups of individuals.
# - Additionally, $\omega$ could be updated at every subsequent time point to represent the percentage of individuals in state $[B]$ multiplied by some constant intra-cluster contact rate.

# In[ ]:




