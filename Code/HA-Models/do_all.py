# filename: do_all.py

# Import the exec function
from builtins import exec
import os

#%%
# Step 1: Estimation of the splurge factor: 
# This file replicates the results from paper section 3.1, creates Figure 1 (in Target_AggMPCX_LiquWealth/Figures),
# and saves results in Target_AggMPCX_LiquWealth as .txt files to be used in following steps. For the robustness checks, 
# script also estimates the splurge factor for CRRA = 1 and 3.
print('Running Step 1 ... \n')
script_path = "Target_AggMPCX_LiquWealth/Estimation_BetaNablaSplurge.py"
exec(open(script_path).read())
print('Concluded Step 1. \n')



#%%
# Step 2: Estimating the discount factor distributions: This file replicates the results from paper section 3.4, creates Figure 2, 7, 8





#%%
# Step 3: Comparing fiscal stimulus policies: This file replicates the results from paper section 4, 
# creates Figures 3-6 (located in FromPandemicCode/Figures), creates tables (located in FromPandemicCode/Tables)
# and creates robustness results
print('Running Step 3 ... \n')
script_path = "AggFiscalMAIN.py"
os.chdir('FromPandemicCode')
exec(open(script_path).read())
os.chdir('../')
print('Concluded Step 3. \n')