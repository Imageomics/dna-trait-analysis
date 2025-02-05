"""
Methods suffixed with _imm were adapted from https://github.com/EpistasisLab/epistasis_detection based on the paper:

Sandra Batista, Vered Senderovich Madar, Philip J. Freda, Priyanka Bhandary, Attri Ghosh, 
Nick Matsumoto, Apurva S. Chitre, Abraham A. Palmer, Jason H. Moore. 
Interaction models matter: an efficient, flexible computational framework for model-specific investigation of epistasis. 
BioData Mining 2024; doi: https://doi.org/10.1186/s13040-024-00358-0
"""

import pandas as pd
import numpy as np
import scipy.stats
import math
from time import process_time_ns
import statsmodels.stats.multitest as ssm

# Algorithm for Interaction Coefficient for Pairwise Epistasis
def detect_epistasis_pair_imm(snp1, snp2, phenotype, df):
    """Parameters for the method:
        snp1 - First snp in the pair to be checked for interaction
        snp2 - Second snp in the pair to be checked for interaction
        phenotype - Phenotype vector
    """
    try:
        # removing the intercept by mean centering
        snp1_tilde = snp1 - snp1.mean()
        snp2_tilde = snp2 - snp2.mean()
        phenotype_tilde = phenotype - phenotype.mean()

        # declaring the interaction vector - will contain the interaction term of snp1 and snp2
        interaction_vector = pd.Series(dtype='float64')
    
        # defining the interaction vector - either using cartesian product or XOR
        interaction_vector = snp1.mul(snp2) # using cartesian product
        # interaction_vector = (snp1%2 + snp2%2)%2 # using XOR penetrance
    
        # print(interaction_vector)
        # mean centering the interaction vector 
        interaction_vector_tilde = interaction_vector - interaction_vector.mean()
   
        # computing the dot products as explained in the algorithm
        x = (snp1_tilde.dot(interaction_vector_tilde)/snp1_tilde.dot(snp1_tilde)) * (snp1_tilde) # 2nd term in the v variable, using to breakdown the formula
        v = interaction_vector_tilde - x
        q2 = snp2_tilde - (((snp1_tilde.dot(snp2_tilde)) / (snp1_tilde.dot(snp1_tilde))) * snp1_tilde)
        v = v - ((interaction_vector_tilde.dot(q2)/q2.dot(q2))*q2)
        b3 = (v.dot(phenotype_tilde)) / (v.dot(v)) # interactionn coefficient, referred as beta_3 in the paper


        # residual calculation
        residual = phenotype_tilde - (snp1_tilde.dot(phenotype_tilde)/snp1_tilde.dot(snp1_tilde))*snp1_tilde
        residual = residual - (phenotype_tilde.dot(q2)/q2.dot(q2))*q2
        residual = residual - b3 * v

        # print(b3)

        v = pd.Series(np.squeeze(np.asarray(v)), dtype='float64')
        residual = pd.Series(np.squeeze(np.asarray(residual)), dtype='float64') 


        t_test = np.sqrt(snp1.shape[0]-4)*np.sqrt(v.dot(v))*b3/(np.sqrt(residual.dot(residual)))

        # t_test = np.sqrt(df)*b3/(np.sqrt(1-b3*b3))
        p_val = scipy.stats.t.sf(abs(t_test), df) * 2
    except Exception as e:
        print("Error pair detected with error: ", e)
        b3 = 0
        t_test = 0
        p_val = 1
        
    return b3, t_test, p_val # returning the interaction coefficient,, t test value and p value

def detect_epistasis_imm(data, phenotype, alpha=0.05):
    
    n = data.shape[0] # Number of samples
    df = n-4 # Degress of freedom
    critical_value = scipy.stats.t.ppf(q=1-alpha/2,df=df)
    
    p_value_locus = [] # list of tuples containing the p_value and the two interacting loci
    for i in range(0, data.shape[1]-1):
        for j in range(i+1, data.shape[1]-1):
            interacting_snp_1 = data.iloc[:,i]
            interacting_snp_2 = data.iloc[:,j]
            b3, t_test, p_val = detect_epistasis_pair_imm(interacting_snp_1, interacting_snp_2, phenotype, df=df)
            p_value_locus.append((p_val, data.columns[i], data.columns[j], b3, t_test))

            #print(" Interacting SNP 1 = {0} \t  Interacting SNP 2 ={1} \t  Beta Coefficient(r) = {2}  \t t_test = {3} \t p_val = {4} ".format(data.columns[i], data.columns[j], b3, t_test, p_val))
            
    # performing fdr correction to correct for multiple tests

    all_p_values = [] # list of all the p_values of the pairwise combinations
    for i in range(0, len(p_value_locus)):
        all_p_values.append(p_value_locus[i][0])

    accept, corrected_pvals = ssm.fdrcorrection(np.asarray(all_p_values).flatten(), alpha=0.001)

    adjusted_p_value_locus = []
    for i in range(0, len(corrected_pvals)):
        # printing the significant pairs after p value correction
        if corrected_pvals[i] < 0.001:
            print(" p-value = {0} \t adjusted p-value = {1} \t loci 1 = {2} \t loci 2 = {3} ".format( p_value_locus[i][0],  corrected_pvals[i], p_value_locus[i][1], p_value_locus[i][2]))
            adjusted_p_value_locus.append([p_value_locus[i][0],  corrected_pvals[i], p_value_locus[i][1], p_value_locus[i][2], True])
        else:
            adjusted_p_value_locus.append([p_value_locus[i][0],  corrected_pvals[i], p_value_locus[i][1], p_value_locus[i][2], False])
            
    return p_value_locus, adjusted_p_value_locus
    
