#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import math
from numpy.linalg import inv
from scipy.optimize import fsolve
from scipy.optimize import least_squares
from tabulate import tabulate

def get_sigVecInRollCS_from_sigPhi(sigma, deg): 

    # this function returns components of stress tensor
    # from uniaxial tensile test, in direction deg (in DEGREES) 
    # to stress components xx,yy,xy in main rolling direction
    
    #sigma -- uniaxial yield/flow stress (scalar) corresponding to the direction inclined at the given angle with respect to a rolling direction 
    #uniaxial stress transformed to the reference configuration/material's orthotropic frame
    
    rad = deg/180 * math.pi
    return np.array([[sigma * math.cos(rad)**2], [sigma * math.sin(rad)**2], [sigma * math.sin(rad) * math.cos(rad)]])


# In[2]:


def yld2000_EqvStr(alpha,a,sigma_vec):
    
    #alpha[0...7] , a=yield locus exponent, sigma_vec=stress componenets xx,yy,xy in vector/tensor form
    #this function returns effective(equivalent) stress at a stress point in yield locus
    M1 = np.array([[2/3,0,0],
                   [-1/3,0,0],
                   [0,-1/3,0],
                   [0,2/3,0],
                   [0,0,1]])
    M2 = np.array([[-2,2,8,-2,0],
                   [1,-4,-4,4,0],
                   [4,-4,-4,1,0],
                   [-2,8,2,-2,0],
                   [0,0,0,0,9]])
    M2 = 1/9 * M2
    alpha_for_l1 = np.array([[alpha[0]],[alpha[1]],[alpha[6]]])
    l1 = np.matmul(M1,alpha_for_l1)
    alpha_for_l2 = np.array([[alpha[2]],[alpha[3]],[alpha[4]],[alpha[5]],[alpha[7]]])
    l2 = np.matmul(M2,alpha_for_l2)
    #l(i) are rearranged in 2d form in L(i)
    L1 = np.array([[l1[0],l1[1],0], 
                   [l1[2],l1[3],0], 
                   [0,0,l1[4]]])
    L2 = np.array([[l2[0],l2[1],0], 
                   [l2[2],l2[3],0], 
                   [0,0,l2[4]]])
    X1 = np.matmul(L1,sigma_vec)                                             
    X2 = np.matmul(L2,sigma_vec)
    delta1 = (X1[0]-X1[1])**2 + 4*X1[2]**2
    delta2 = (X2[0]-X2[1])**2 + 4*X2[2]**2
    X1prin1 = 1/2*( X1[0] + X1[1] + math.sqrt(delta1)) #X'1
    X1prin2 = 1/2*( X1[0] + X1[1] - math.sqrt(delta1)) #X'2
    X2prin1 = 1/2*( X2[0] + X2[1] + math.sqrt(delta2))#X"1
    X2prin2 = 1/2*( X2[0] + X2[1] - math.sqrt(delta2)) #X"2
    Phi1 = abs(X1prin1 - X1prin2)**a
    Phi2 = abs(2*X2prin2 + X2prin1)**a + abs(2*X2prin1 + X2prin2)**a
    EqvStress = ((Phi1 + Phi2)/2)**(1/a)
    return float(EqvStress) #returns effective stress at a stress point in yield locus


# In[3]:

def sigmaVec_dist_to_unitYLD2000(alpha,a,sigma_vec): 
    
    #this function returns the value of (yield function error) distance between the UNIT yield locus and the sigma vector
        
    M1 = np.array([[2/3,0,0],
                   [-1/3,0,0],
                   [0,-1/3,0],
                   [0,2/3,0],
                   [0,0,1]])
    M2 = np.array([[-2,2,8,-2,0],
                   [1,-4,-4,4,0],
                   [4,-4,-4,1,0],
                   [-2,8,2,-2,0],
                   [0,0,0,0,9]])
    M2 = 1/9 * M2
    alpha_for_l1 = np.array([[alpha[0]],[alpha[1]],[alpha[6]]])
    l1 = np.matmul(M1,alpha_for_l1)
    alpha_for_l2 = np.array([[alpha[2]],[alpha[3]],[alpha[4]],[alpha[5]],[alpha[7]]])
    l2 = np.matmul(M2,alpha_for_l2)
    L1 = np.array([[l1[0],l1[1],0], [l1[2],l1[3],0], [0,0,l1[4]]])
    L2 = np.array([[l2[0],l2[1],0], [l2[2],l2[3],0], [0,0,l2[4]]])
    X1 = np.matmul(L1,sigma_vec)                                               
    X2 = np.matmul(L2,sigma_vec)
    delta1 = (X1[0]-X1[1])**2 + 4*X1[2]**2
    delta2 = (X2[0]-X2[1])**2 + 4*X2[2]**2
    X1prin1 = 1/2*( X1[0] + X1[1] + math.sqrt(delta1)) #X'1
    X1prin2 = 1/2*( X1[0] + X1[1] - math.sqrt(delta1)) #X'2
    X2prin1 = 1/2*( X2[0] + X2[1] + math.sqrt(delta2))#X"1
    X2prin2 = 1/2*( X2[0] + X2[1] - math.sqrt(delta2)) #X"2
    Phi1 = abs(X1prin1 - X1prin2)**a
    Phi2 = abs(2*X2prin2 + X2prin1)**a + abs(2*X2prin1 + X2prin2)**a
    yld2000_val = (Phi1 + Phi2) - 2      
    return float(yld2000_val) #returns distance between the UNIT yield locus and the sigma vector


# In[4]:


def yld2000_derivative(alpha,a,sigma_vec): 
    
    # this function gives the the derivates of yld2000-2d function for a given stress state sigma_vec and a given set of alpha and a
    # return type is a (1,3) numpy array: array([[array([-0.59769715]), array([0.57948825]), array([0.55403203])]], dtype=object)
    # return vector is normalized. So that the vector has a magnitude of 1.
    
    M1 = np.array([[2/3,0,0],[-1/3,0,0],[0,-1/3,0],[0,2/3,0],[0,0,1]])
    M2 = np.array([[-2,2,8,-2,0],[1,-4,-4,4,0],[4,-4,-4,1,0],[-2,8,2,-2,0],[0,0,0,0,9]])
    M2 = 1/9 * M2
    alpha_for_l1 = np.array([[alpha[0]],[alpha[1]],[alpha[6]]])
    l1 = np.matmul(M1,alpha_for_l1)
    alpha_for_l2 = np.array([[alpha[2]],[alpha[3]],[alpha[4]],[alpha[5]],[alpha[7]]])
    l2 = np.matmul(M2,alpha_for_l2)
    L1 = np.array([[l1[0],l1[1],0], [l1[2],l1[3],0], [0,0,l1[4]]])
    L2 = np.array([[l2[0],l2[1],0], [l2[2],l2[3],0], [0,0,l2[4]]])
    X1 = np.matmul(L1,sigma_vec)                                               
    X2 = np.matmul(L2,sigma_vec)

    # T = np.array([[2/3,-1/3,0],[-1/3,2/3,0],[0,0,1]])
    # Tinv = inv(T)
    # C1 = np.matmul(L1,Tinv)
    # C2 = np.matmul(L2,Tinv)

    delta1 = (X1[0]-X1[1])**2 + 4*X1[2]**2
    delta2 = (X2[0]-X2[1])**2 + 4*X2[2]**2
    X1prin1 = 1/2*( X1[0] + X1[1] + math.sqrt(delta1)) #X'1
    X1prin2 = 1/2*( X1[0] + X1[1] - math.sqrt(delta1)) #X'2
    X2prin1 = 1/2*( X2[0] + X2[1] + math.sqrt(delta2))#X"1
    X2prin2 = 1/2*( X2[0] + X2[1] - math.sqrt(delta2)) #X"2

    if delta1 < 1e-10 or abs(X1prin1 - X1prin2) < 1e-10:
        dPhi1_dX1 = np.zeros((3,1))

        dPhi1_dX1[0] = 0
        dPhi1_dX1[1] = 0
        dPhi1_dX1[2] = 0

    else:

        dPhi1_dX1 = np.zeros((3,1))
        dX1prin1_dX1 = np.zeros((3,1))
        dX1prin2_dX1 = np.zeros((3,1))

        dPhi1_dX1prin1 = a*(X1prin1 - X1prin2)**(a-1)
        dPhi1_dX1prin2 = -a*(X1prin1 - X1prin2)**(a-1)

        dX1prin1_dX1[0] = 1/2 * (1 + (X1[0]-X1[1])/math.sqrt(delta1))
        dX1prin1_dX1[1] = 1/2 * (1 - (X1[0]-X1[1])/math.sqrt(delta1))
        dX1prin1_dX1[2] = 2*X1[2]/math.sqrt(delta1)

        dX1prin2_dX1[0] = 1/2 * (1 - (X1[0]-X1[1])/math.sqrt(delta1))
        dX1prin2_dX1[1] = 1/2 * (1 + (X1[0]-X1[1])/math.sqrt(delta1))
        dX1prin2_dX1[2] = -2*X1[2]/math.sqrt(delta1)

        dPhi1_dX1 = dPhi1_dX1prin1 * dX1prin1_dX1 + dPhi1_dX1prin2 * dX1prin2_dX1

    if delta2 < 1e-10 or abs(X2prin1 - X2prin2) < 1e-10:

        dPhi2_dX2 = np.zeros((3,1))
        dPhi2_dX2[0] = a * abs((2*X2prin2 + X2prin1))**(a-1) * np.sign(2*X2prin2+X2prin1) + 2 * a * abs(2*X2prin1+X2prin2)**(a-1) * np.sign(2*X2prin1+X2prin2)
        dPhi2_dX2[1] = dPhi2_dX2[0]
        dPhi2_dX2[2] = 0

    else:

        dX2prin1_dX2 = np.zeros((3,1))
        dX2prin2_dX2 = np.zeros((3,1))
        dPhi2_dX2prin1 = a*abs(2*X2prin2+X2prin1)**(a-1)*np.sign(2*X2prin2+X2prin1) + 2*a*abs(2*X2prin1+X2prin2)**(a-1)*np.sign(2*X2prin1+X2prin2)
        dPhi2_dX2prin2 = 2*a*abs(2*X2prin2+X2prin1)**(a-1)*np.sign(2*X2prin2+X2prin1) + a*abs(2*X2prin1+X2prin2)**(a-1)*np.sign(2*X2prin1+X2prin2)

        dX2prin1_dX2[0] = 1/2 * (1 + (X2[0]-X2[1])/math.sqrt(delta2))
        dX2prin1_dX2[1] = 1/2 * (1 - (X2[0]-X2[1])/math.sqrt(delta2))
        dX2prin1_dX2[2] = 2*X2[2]/math.sqrt(delta2)

        dX2prin2_dX2[0] = 1/2 * (1 - (X2[0]-X2[1])/math.sqrt(delta2))
        dX2prin2_dX2[1] = 1/2 * (1 + (X2[0]-X2[1])/math.sqrt(delta2))
        dX2prin2_dX2[2] = -2*X2[2]/math.sqrt(delta2)

        dPhi2_dX2 = dPhi2_dX2prin1 * dX2prin1_dX2 + dPhi2_dX2prin2* dX2prin2_dX2

    dPhi_dsig = np.matmul(dPhi1_dX1.transpose(),L1) + np.matmul(dPhi2_dX2.transpose(),L2)
    
    dPhi_dsig[0][2] = dPhi_dsig[0][2]/2
    #dPhi_dsig = dPhi_dsig.astype('int32')
    #dPhi_dsig = dPhi_dsig / float(np.linalg.norm(dPhi_dsig)) # this normalizes the output vector
    #dPhi_dsig = dPhi_dsig / np.linalg.norm(dPhi_dsig) # this normalizes the output vector
    #print(dPhi_dsig)
    dPhi_dsig = dPhi_dsig / float(np.linalg.norm(dPhi_dsig)[0])
    return dPhi_dsig   #this function returns 1*3 numpy array which has a magnitude of 1


# In[5]:


def yld2000_err(alpha_2_to_8,a,sig_0,sig_45,sig_90,sig_b,r0,r45,r90,rb):
    
    # this function is used by the optimizer for alpha parameters
    # only the first argument(here:alpha_2_to_8) is optimized while others are passed in to calculate residual/objective function inside yld2000_err function
    # this function returns the RMS-Error (between: numerically calculated  - experimentally determined/given) values for stresses(F) and Lankford anisotropy coefficients(G) for these cases 0, 45, 90 degrees and biaxial tension
    # calculation is done in the "pressure"-space not in the unit-space. Hence sigma_0 is also considered as the equivalent stress
    
    # yield locus intersecting the Sigma-1 axis at sig_0 yields a relation between 6 alpha values from a1 to a6: 
    # when all of the 8 parameters are varied simultaneously without any constraining equation,
    # the obtained yield loci do not necessarily reflect the yield stress of the material 
    # designated by the flow curve in rolling direction. In the graphical representation,
    # the predicted yield loci, at the horizontal axis in stress space, do not assume the
    # value of the yield stress obtained in the rolling direction. In order to avoid this
    # artificial scaling of the flow curve in rolling direction, a relation between the material 
    # parameters is given.By the help of this relation ONE parameter becomes dependent to others
    # and therefore only 7 alphas are varied in the parameter identification scheme.
    # (Reference: Güner et al, https://doi.org/10.1016/j.ijsolstr.2012.05.001)
 
    # regarding the dependency between the first six alpha parameters,
    # the first alpha is calculated from the following five using the relation:
    # alpha_1 = ( ( (2 - abs((2*alpha_2_to_8[1]-2*alpha_2_to_8[2])/3.0)**a - abs((4*alpha_2_to_8[3]-alpha_2_to_8[4])/3)**a)**(1/a) )  * 3.0 - alpha_2_to_8[0] ) / 2.0

    # it is taken that: alpha_1 = 1.0  before the yield locus normalization
    factor = ( (abs((2.0+alpha_2_to_8[0])/3.0))**a + (abs((2.0*alpha_2_to_8[1]-2.0*alpha_2_to_8[2])/3.0))**a + (abs((4.0*alpha_2_to_8[3]-alpha_2_to_8[4])/3.0))**a  ) / 2.0
    factor = factor ** (1.0/a) # this factor is now used to normalize the yield locus by dividing all alpha values with the constrain related factor
    alpha_2_to_8 = [i/factor for i in alpha_2_to_8]  # here the normalized values
    alpha_1 = 1.0 / factor
    
    
    #the calculated dependent alpha is inserted to the list of remaining 7 alphas in position 0: altogther we have 8 values now
    alpha = np.insert(alpha_2_to_8, 0, alpha_1, axis=0)
          
    sig0_vec = get_sigVecInRollCS_from_sigPhi(sig_0,0)      #np.array([[sig_0],[0],[0]])
    sig45_vec = get_sigVecInRollCS_from_sigPhi(sig_45, 45)
    sig90_vec = get_sigVecInRollCS_from_sigPhi(sig_90, 90)  #np.array([[0],[sig_90],[0]])

    #biaxial stress so doesnt have to be rotated
    sigb_vec = np.array([[sig_b],[sig_b],[0]])

    #Errors(diffrence from equivalent stress) for stress points are calculated:
    F_0  = yld2000_EqvStr(alpha,a,sig0_vec) - sig_0      # sig_0 is the equivalent stress in yld2000 definition. 
    F_90 = yld2000_EqvStr(alpha,a,sig90_vec) - sig_0
    F_b  = yld2000_EqvStr(alpha,a,sigb_vec) - sig_0  
    F_45  = yld2000_EqvStr(alpha,a,sig45_vec) - sig_0
    #print(F_0)
    #print(F

    dPhi_dsig_0 = yld2000_derivative(alpha,a,sig0_vec)  #this function returns 1*3 numpy array
    dPhi_dsig_90 = yld2000_derivative(alpha,a,sig90_vec)
    dPhi_dsig_b = yld2000_derivative(alpha,a,sigb_vec)
    dPhi_dsig_45 = yld2000_derivative(alpha,a,sig45_vec)
    
    #print(dPhi_dsig_0)
    #print(dPhi_dsig_90)
    #print(dPhi_dsig_b)
    #print(dPhi_dsig_45)
        
    #Errors(difference from experimental anisotropy parameters) for lankford anisotropy parameters are calculated:
    
    G_0 = -(math.sin(0/180 * math.pi)**2*dPhi_dsig_0[0][0] - math.sin(2*0)*dPhi_dsig_0[0][2] + math.cos(0)**2*dPhi_dsig_0[0][1]) / (dPhi_dsig_0[0][0] + dPhi_dsig_0[0][1]) - r0
    G_45 = -(math.sin(45/180 * math.pi)**2*dPhi_dsig_45[0][0] - math.sin(2*45/180 * math.pi)*dPhi_dsig_45[0][2] + math.cos(45/180 * math.pi)**2*dPhi_dsig_45[0][1])/(dPhi_dsig_45[0][0] + dPhi_dsig_45[0][1]) - r45
    G_90 = -(math.sin(90/180 * math.pi)**2*dPhi_dsig_90[0][0] - math.sin(2*90/180 * math.pi)*dPhi_dsig_90[0][2] + math.cos(90/180 * math.pi)**2*dPhi_dsig_90[0][1])/(dPhi_dsig_90[0][0] + dPhi_dsig_90[0][1]) - r90
    G_b = dPhi_dsig_b[0][1] / dPhi_dsig_b[0][0] - rb
    
    #yld2000_err function returns an array of differences between the numerically calculated R and Stress values and the given ones
    #the differences can be negative and positive, and are considered by least squares method during optimization
    #print(np.array([F_0, F_45,F_90,F_b,G_0,G_45,G_90,G_b]))
    return np.array([F_0, F_45,F_90,F_b,G_0[0],G_45[0],G_90[0],G_b[0]]) 


# ??? toDo: CK: Rename this function get_RvsTheta_Rb_values
def  get_R_vs_Theta_Rb_values(alpha,a):
    
    # function returns the lankford anisotropy coefficients for uniaxial tension tests in direction ranging from 0 to 90 degrees with respect to the rolling direction  (FORWARD COMPUTATION)
    # function returns a 1x90 array, indexes being R[0][0..89]
    
    yield_stress_guess = 1 # 1 is the exact guess for an isotropic material
    deg = np.array([list(range(0,91))])
    rad = deg/180*math.pi
    R = np.zeros((1, np.shape(rad)[1]))
        
    for i in range(np.shape(rad)[1]):
        
        sig_vec = get_sigVecInRollCS_from_sigPhi(yield_stress_guess, deg[0][i])
        dPhi_dsig = yld2000_derivative(alpha,a,sig_vec)

        R[0][i] = -(math.sin(rad[0][i])**2*dPhi_dsig[0][0] - math.sin(2*rad[0][i])*dPhi_dsig[0][2] + math.cos(rad[0][i])**2*dPhi_dsig[0][1]) / (dPhi_dsig[0][0] + dPhi_dsig[0][1])
   
    sigb_vec = np.array([[1.0],[1.0],[0]])
    dPhi_dsig_b = yld2000_derivative(alpha,a,sigb_vec)
    Rb = dPhi_dsig_b[0][1] / dPhi_dsig_b[0][0]
    
    return R, Rb


# In[5]:

def get_UniaxYieldStress_vs_Theta_values(alpha, a, Y_0=1.0, NormalizeAlpha=True):
    
    SigmaInput = 1.0
    if NormalizeAlpha == True:
        factor = ( (abs((2.0*alpha[0]+alpha[1])/3.0))**a + (abs((2.0*alpha[2]-2.0*alpha[3])/3.0))**a + (abs((4.0*alpha[4]-alpha[5])/3.0))**a  ) / 2.0
        factor = factor ** (1.0/a)
        alpha = [i/factor for i in alpha]
        
        # alpha1 = ( ( (2 - abs((2*alpha[2]-2*alpha[3])/3.0)**a - abs((4*alpha[4]-alpha[5])/3)**a)**(1/a) )  * 3.0 - alpha[1] ) / 2.0
        
        # if np.iscomplex(alpha1)==True:
        #     print('Complex number encountered!Taking only the real part!')
        # alpha[0] = alpha1.real
        
    deg = np.array([list(range(0,91))])
    stress = np.zeros((1, np.shape(deg)[1]))
    
    for i in range(np.shape(deg)[1]):
        stress[0][i] = SigmaInput * Y_0 / yld2000_EqvStr(alpha,a,get_sigVecInRollCS_from_sigPhi(SigmaInput, deg[0][i]))
    
    return stress


# In[ ]:
def get_magnitudeSIGMA_in_directionDEG_for_unitYL(alpha,a,shear,deg):
    # used by the function: get_stress_points_in_Yieldlocus() in case of OVERLAID SHEAR  !!!!
    # this functions returns the sigma(scalar) that minimizes the (yield function error) distance between the UNIT yield locus and the sigma vector. This is used for plotting a UNIT yield locus.
        
    sigmaGuess = 1.0 # sigmaGuess is used as the initial guess.
    rad = (deg/180)*math.pi
        
    #distError returns the distance between the stress point(rad,sigmaGuess = magnitude) and the UNIT yield locus
    def distError(sigmaGuess,alpha,a,shear,rad): 
        sig_dir =np.array([[sigmaGuess * math.cos(rad)], [sigmaGuess * math.sin(rad)], [shear]])
        return sigmaVec_dist_to_unitYLD2000(alpha,a,sig_dir)

    return fsolve(distError,sigmaGuess,args=(alpha,a,shear,rad))

    

def get_stress_points_in_Yieldlocus(alpha,a,shear):
 
    # this function returns values for yield locus: plot_x and plot_y ,and sigmas(stresses): sigma_00, sigma_45, sigma_90, sigma_biax, shear1, shear2, sigmaplanestrain_1 and sigmaplanestrain_2 for a given set of alpha and a (FORWARD COMPUTATION)
    
    Y_0 = 1.0 # yield stress in rolling direction    
    deg = np.array([list(range(0,361))])
 
    plot_x = np.zeros((1, np.shape(deg)[1])) #x-components of stresses in yield locus
    plot_y = np.zeros((1, np.shape(deg)[1])) #y-components of stresses in yield locus
  
    # there are some special points on yield locus: shear1, shear2, plane strain 1, plane strain 2, etc. 
    # shear point at second quadrant (135 degrees in Yield Locus)   |   shear point at fourth quadrant (-45 degrees in Yield Locus)
    if -1e-15 <= shear <= 1e-15: # if shear is zero, stress points can be calculated directly
        
        shear_x_input, shear_y_input = -1.0, 1.0
        #shear1_x = shear_x_input * Y_0 / yld2000_EqvStr(alpha,a,[shear_x_input, shear_y_input, 0.0])
        shear1_y = shear_y_input * Y_0 / yld2000_EqvStr(alpha,a,[shear_x_input, shear_y_input, 0.0]) #x and y components have same absolute values
        shear_x_input, shear_y_input = 1.0, -1.0 
        shear2_x = shear_x_input * Y_0 / yld2000_EqvStr(alpha,a,[shear_x_input, shear_y_input, 0.0])#x and y components have same absolute values
        #shear2_y = shear_y_input * Y_0 / yld2000_EqvStr(alpha,a,[shear_x_input, shear_y_input, 0.0])
        
    else: # if overlaid shear is not zero, stress points has to be calculated iteratively!
        
        sigma_shear1 = get_magnitudeSIGMA_in_directionDEG_for_unitYL(alpha,a,shear,deg=deg[0][135])
        sigma_shear2 = get_magnitudeSIGMA_in_directionDEG_for_unitYL(alpha,a,shear,deg=-deg[0][45])
        shear1_y = sigma_shear1 * math.sin(deg[0][135]/180 * math.pi) #x and y components have same absolute values
        shear2_x = sigma_shear2 * math.cos(-deg[0][45]/180 * math.pi) #x and y components have same absolute values
        
  
    # tensile test at 45 degrees to the rolling direction. This does not consider a further shear component AND can not be directly demonstrated on the principal stress space because of the existing intrinsic shear component
    # sigm_45 tests ignore shear components in principal directions(for plotting purposes)
    cos45 = 1.0 / math.sqrt(2)
    sin45 = cos45
    sigma_45_x_input, sigma_45_y_input, sigma_45_xy_input = cos45**2, sin45**2, cos45 * sin45
    BaseYieldStress = yld2000_EqvStr(alpha,a,[1.0, 0.0, 0.0])
    sigma_45_x  =  sigma_45_x_input * BaseYieldStress / yld2000_EqvStr(alpha,a,[sigma_45_x_input, sigma_45_y_input, sigma_45_xy_input])
    sigma_45 = sigma_45_x / (cos45**2)
      
    # loop below calculates the stress points on the Yield Locus for each angular direction in principal (rolling) yield stress space       
    for i in range(np.shape(deg)[1]):
 
        if -1e-15 <= shear <= 1e-15: # if shear is zero, stress points can be calculated directly 
        
            sigmaXinput ,  sigmaYinput = math.cos(deg[0][i]/180.0 * math.pi) ,  math.sin(deg[0][i]/180.0 * math.pi)
            plot_x[0][i] = sigmaXinput * Y_0 / yld2000_EqvStr(alpha,a,[sigmaXinput, sigmaYinput, 0.0])
            plot_y[0][i] = sigmaYinput * Y_0 / yld2000_EqvStr(alpha,a,[sigmaXinput, sigmaYinput, 0.0])  
            
        else: # if overlaid shear is not zero, stress points have to be calculated iteratively!
            
            sigma = get_magnitudeSIGMA_in_directionDEG_for_unitYL(alpha,a,shear,deg=deg[0][i])
            plot_x[0][i], plot_y[0][i] = sigma * math.cos(deg[0][i]/180 * math.pi), sigma * math.sin(deg[0][i]/180 * math.pi)
            

        if -1e-15 <= plot_x[0][i] <= 1e-15 and plot_y[0][i] > 0:  #get sigma_90
            sigma_90 = plot_y[0][i]
            #sigma1_for_sigma_90 = plot_x[0][i]
            
        if -1e-15 <= plot_y[0][i] <= 1e-15 and plot_x[0][i] > 0:  #get sigma_00
            sigma_00 = plot_x[0][i]
            #sigma2_for_sigma_00 = plot_y[0][i]
            
        if abs(plot_x[0][i] - plot_y[0][i]) <= 1e-5 and np.sign(plot_x[0][i]) > 0.0 and np.sign(plot_y[0][i]) > 0.0: #get sigma_biax_x,                                                                                                                               sigma_biax_y
            #sigma_biax = plot_x[0][i]                                                                 #such that they have to be equal and 
            sigma_biax_x = plot_x[0][i]                                                                 #also have + sign -> first quadrant
            sigma_biax_y = plot_y[0][i]                                                               
            
    PlaneStrain_1x = np.amax(plot_x) #stress component that gives highest equivalent stress point in yield locus in xx direction
    index_PlaneStrain_1 = np.argmax(plot_x, axis=1)
    PlaneStrain_1y = plot_y[0][index_PlaneStrain_1]
    
    PlaneStrain_2y = np.amax(plot_y) #stress component that gives highest equivalent stress point in yield locus yy direction
    index_PlaneStrain_2 = np.argmax(plot_y, axis=1)
    PlaneStrain_2x = plot_x[0][index_PlaneStrain_2]  

    angle_plane_strain_1 = ((math.atan(PlaneStrain_1y / PlaneStrain_1x))*180)/math.pi
    angle_plane_strain_2 = ((math.atan(PlaneStrain_2y / PlaneStrain_2x))*180)/math.pi
    angle_plane_strain_2 =(( math.atan(PlaneStrain_2y / PlaneStrain_2x))*180)/math.pi
    angle_sigma_biax = ((math.atan(sigma_biax_y / sigma_biax_x))*180)/math.pi
	               
    return np.array([[plot_x, plot_y, sigma_00, sigma_90, sigma_45, sigma_biax_x, shear1_y, shear2_x, PlaneStrain_1x, PlaneStrain_1y, PlaneStrain_2x, PlaneStrain_2y,
	angle_plane_strain_1, angle_plane_strain_2, angle_sigma_biax]])


# In[ ]:

def calculate_optimized_alphas(a,sig_0,sig_45,sig_90,sig_b,r0,r45,r90,rb):
    
    #this function calculates the optimized alpha values from given stress and lankford anisotropy values (REVERSE COMPUTATION)
    # a certain relation between the alpha values have to be satisfied: Refer to yld2000_err function for more information. 
    #alpha0 is a list of alphas from alpha_2 to alpha_8, alpha_1 is calculated from the relation of alpha_2 to alpha_6 (PS: the indexing in python starts from 0)
    
    alpha_init_2_to_8=[1.0,1.0,1.0,1.0,1.0,1.0,1.0] # this is the initial guess necessary for the optimization below
    # alpha_init_2_to_8=[0.85,1.1,0.85,1.2,1.2,0.85,1.2] # this is the initial guess necessary for the optimization below
    
    # in function yld2000_err it is taken that alpha_1=1.0 as reference before the normalization of the yield locus.
    alpha_lm = least_squares(yld2000_err, alpha_init_2_to_8, method='lm', ftol=1.49012e-08, xtol=1.49012e-08,  args=(a,sig_0,sig_45,sig_90,sig_b,r0,r45,r90,rb))
    #yld2000_err function returns an array of differences between the numerically calculated R and Stress values and the given ones
    #the differences can be negative and positive, and are considered by least squares method during optimization
    
    #only seven values are optimized so the eighth one is calculated with the help of following relation. It is a constraint to have yieldlocus intersect at sigma_11 = 1

    factor = ( (abs((2.0+alpha_lm.x[0])/3.0))**a + (abs((2.0*alpha_lm.x[1]-2.0*alpha_lm.x[2])/3.0))**a + (abs((4.0*alpha_lm.x[3]-alpha_lm.x[4])/3.0))**a  ) / 2.0
    factor = factor ** (1.0/a)
    alpha_lm.x = [i/factor for i in alpha_lm.x]
    alpha_lm_1 = 1.0 / factor

    # alpha_lm_1 =  ( ( (2 - abs((2*alpha_lm.x[1]-2*alpha_lm.x[2])/3.0)**a - abs((4*alpha_lm.x[3]-alpha_lm.x[4])/3)**a)**(1/a) )  * 3.0 - alpha_lm.x[0] ) / 2.0     
    # if np.iscomplex(alpha_lm_1)==True:
    #     print('Complex number encountered!Taking only the real part!')
    #     alpha_lm_1 = alpha_lm_1.real
       
    alpha = np.insert(alpha_lm.x, 0, alpha_lm_1, axis=0)
    
    return alpha  # function returns all the 8 alpha values such that the yield locus intersects sigma1 axis with the equivalent yield stress (or 1 in case of normalized yield locus, but doesn't intersect at 1 if alpha_lm_1 comes out to be complex number and only real part is considered) 


def compare_given_and_calculated_Sigmas_Rs(a,sig_0,sig_45,sig_90,sig_b,r0,r45,r90,rb,printResult=False):
    
    #Normalization
    normalizing_factor = sig_0
    sig_0_n = sig_0/normalizing_factor
    sig_45_n = sig_45/normalizing_factor
    sig_90_n = sig_90/normalizing_factor
    sig_b_n = sig_b/normalizing_factor
    
    # function calculates first the optimized alphas (reverse comnputation) from the Given-Sigmas-&-Rs
    # then Stresses and R values are calculated using the found alpha values
    # then calculate the error between the initially given stress and R values for RC, and the computed after FC
    
    #Stresses(un-normalized) and R values for reverse computation of optimized alphas
    stress_for_rc= np.array([[sig_0, sig_45, sig_90, sig_b]])
    R_for_rc = np.array([[r0, r45, r90, rb]])
    
    # alpha_init_2_to_8=[1.0,1.0,1.0,1.0,1.0,1.0,1.0] # this is the initial guess necessary for the optimization below
    alpha = calculate_optimized_alphas(a,sig_0_n,sig_45_n,sig_90_n,sig_b_n,r0,r45,r90,rb)
    alpha_and_a =np.insert(alpha,len(alpha),a,axis=0)
    #print(alpha)
        
    #forward computation of Stresses and R values from optimized alphas and given exponent a
    S = get_stress_points_in_Yieldlocus(alpha,a,shear=0)  
    R,Rb_est = get_R_vs_Theta_Rb_values(alpha,a)
   
    #De-normalization before error calculation
    
    stress_after_fc = np.array([[S[0][2], S[0][4], S[0][3], S[0][5]]]) * normalizing_factor

    R_after_fc = np.array([[R[0][0], R[0][45], R[0][90], Rb_est]])

    Absolute_Error_for_stress = (abs(stress_for_rc - stress_after_fc))
    Absolute_Error_for_R_value = (abs(R_for_rc - R_after_fc))
    
    Percentage_Error_for_stress = (abs(stress_for_rc - stress_after_fc) / stress_for_rc)*100
    Percentage_Error_for_R_value = (abs(R_for_rc - R_after_fc)/ R_for_rc )*100
    
    if printResult:
        
        table_stress = [["Given Stresses:", stress_for_rc[0][0], stress_for_rc[0][1], stress_for_rc[0][2], stress_for_rc[0][3] ],
                        ["Calculated Stresses:", stress_after_fc[0][0], stress_after_fc[0][1], stress_after_fc[0][2], stress_after_fc[0][3] ],
                        ["Absolute Error for Stresses:",  Absolute_Error_for_stress[0][0],Absolute_Error_for_stress[0][1],Absolute_Error_for_stress[0][2],Absolute_Error_for_stress[0][3]],
                        ["Percentage_Error for Stresses:", Percentage_Error_for_stress[0][0],Percentage_Error_for_stress[0][1], Percentage_Error_for_stress[0][2],Percentage_Error_for_stress[0][3]  ]  ]
           
        print(tabulate(table_stress, headers=['Name','Sigma_00','Sigma_45','Sigma_90','Sigma_biax']))
        print('')
        print('')
        print('')
        print('')
        table_R_value = [["Given R values:", R_for_rc[0][0], R_for_rc[0][1], R_for_rc[0][2], R_for_rc[0][3] ],
                         ["Calculated R values:", R_after_fc[0][0], R_after_fc[0][1], R_after_fc[0][2], R_after_fc[0][3] ],
                         ["Absolute Error for R values:", Absolute_Error_for_R_value[0][0], Absolute_Error_for_R_value[0][1],
                                                           Absolute_Error_for_R_value[0][2] , Absolute_Error_for_R_value[0][3]   ],
                         ["Percentage Error for R values:", Percentage_Error_for_R_value[0][0],Percentage_Error_for_R_value[0][1],Percentage_Error_for_R_value[0][2],Percentage_Error_for_R_value[0][3]  ]  ]
           
        print(tabulate(table_R_value, headers=['Name','R_0','R_45','R_90','R_b']))
        
        print(alpha_and_a)
        
    # return np.array([alpha_and_a, stress_for_rc, R_for_rc, stress_after_fc, R_after_fc, Absolute_Error_for_stress, Absolute_Error_for_R_value, Percentage_Error_for_stress, Percentage_Error_for_R_value], dtype=object)  


def compare_given_and_calculated_alphas(alpha,a,printResult=False):
    
    #Normalization
    factor = ( (abs((2.0*alpha[0]+alpha[1])/3.0))**a + (abs((2.0*alpha[2]-2.0*alpha[3])/3.0))**a + (abs((4.0*alpha[4]-alpha[5])/3.0))**a  ) / 2.0
    factor = factor ** (1.0/a)
    alpha = [i/factor for i in alpha]
    
    
    #forward computation of Stresses and R values from normalized alphas and given exponent a
    S = get_stress_points_in_Yieldlocus(alpha,a,shear=0)  
    R,Rb = get_R_vs_Theta_Rb_values(alpha,a)
        
    sig_0, sig_45, sig_90, sig_b = S[0][2], S[0][4], S[0][3], S[0][5]
    r0, r45, r90, rb = R[0][0], R[0][45], R[0][90], Rb
    
    stress_direct = np.array([[sig_0, sig_45, sig_90, sig_b]])
    R_direct = np.array([[r0, r45, r90, rb]])
    
    alpha_fit = calculate_optimized_alphas(a,sig_0,sig_45,sig_90,sig_b,r0,r45,r90,rb)    
    #the calculated alphas are already normalized during the alpha-fitting process --reverse computation
    
    #recalculation of Stresses and R values from optimized and normalized (fitted) alphas and given exponent a
    S = get_stress_points_in_Yieldlocus(alpha_fit,a,shear=0)  
    R,Rb = get_R_vs_Theta_Rb_values(alpha_fit,a)
        
    stress_fit = np.array([[S[0][2], S[0][4], S[0][3], S[0][5]]])
    R_fit = np.array([[R[0][0], R[0][45], R[0][90], Rb]])

    Absolute_Error_for_stress = (abs(stress_fit - stress_direct))
    Absolute_Error_for_R_value = (abs(R_fit - R_direct))
    Percentage_Error_for_stress = (abs(stress_fit - stress_direct) / stress_fit)*100
    Percentage_Error_for_R_value = (abs(R_fit - R_direct)/ R_fit )*100
    
    Absolute_Error_for_alpha = (abs(alpha_fit - alpha))
    Percentage_Error_for_alpha = (abs(alpha_fit - alpha) / alpha_fit)*100
    
    if printResult:
        
        table_stress = [["Direct Stresses:", stress_direct[0][0], stress_direct[0][1], stress_direct[0][2], stress_direct[0][3] ],
                        ["Fit Stresses:", stress_fit[0][0], stress_fit[0][1], stress_fit[0][2], stress_fit[0][3] ],
                        ["Absolute Error for Stresses:",  Absolute_Error_for_stress[0][0],Absolute_Error_for_stress[0][1],Absolute_Error_for_stress[0][2],Absolute_Error_for_stress[0][3]],
                        ["Percentage_Error for Stresses:", Percentage_Error_for_stress[0][0],Percentage_Error_for_stress[0][1], Percentage_Error_for_stress[0][2],Percentage_Error_for_stress[0][3]  ]  ]
           
        print(tabulate(table_stress, headers=['Name','Sigma_00','Sigma_45','Sigma_90','Sigma_biax']))
        print('')
        print('')
        print('')
        print('')
        table_R_value = [["Direct R values:", R_direct[0][0], R_direct[0][1], R_direct[0][2], R_direct[0][3] ],
                         ["Fit R values:", R_fit[0][0], R_fit[0][1], R_fit[0][2], R_fit[0][3] ],
                         ["Absolute Error for R values:", Absolute_Error_for_R_value[0][0], Absolute_Error_for_R_value[0][1],
                                                           Absolute_Error_for_R_value[0][2] , Absolute_Error_for_R_value[0][3]   ],
                         ["Percentage Error for R values:", Percentage_Error_for_R_value[0][0],Percentage_Error_for_R_value[0][1],Percentage_Error_for_R_value[0][2],Percentage_Error_for_R_value[0][3]  ]  ]
           
        print(tabulate(table_R_value, headers=['Name','R_0','R_45','R_90','R_b']))
        print('')
        print('')
        print('')
        print('')

        table_alpha = [["Given alphas:", alpha[0], alpha[1], alpha[2], alpha[3], alpha[4], alpha[5], alpha[6], alpha[7] ],
                         ["Fit alphas:", alpha_fit[0], alpha_fit[1], alpha_fit[2], alpha_fit[3], alpha_fit[4], alpha_fit[5], alpha_fit[6], alpha_fit[7]  ],
                         ["Absolute Error for alphas:", Absolute_Error_for_alpha[0], Absolute_Error_for_alpha[1], Absolute_Error_for_alpha[2], Absolute_Error_for_alpha[3], Absolute_Error_for_alpha[4], Absolute_Error_for_alpha[5], Absolute_Error_for_alpha[6], Absolute_Error_for_alpha[7] ],
                         ["Percentage Error for alphas:", Percentage_Error_for_alpha[0], Percentage_Error_for_alpha[1], Percentage_Error_for_alpha[2], Percentage_Error_for_alpha[3], Percentage_Error_for_alpha[4], Percentage_Error_for_alpha[5], Percentage_Error_for_alpha[6], Percentage_Error_for_alpha[7] ]]
           
        print(tabulate(table_alpha, headers=['Name','alpha1','alpha2','alpha3','alpha4','alpha5','alpha6','alpha7','alpha8'], floatfmt=".6f"))

    # return np.array([alpha, alpha_fit, stress_direct, R_direct, stress_fit, R_fit, Absolute_Error_for_stress, Absolute_Error_for_R_value, Absolute_Error_for_alpha, Percentage_Error_for_stress, Percentage_Error_for_R_value, Percentage_Error_for_alpha],  dtype=object) 
    