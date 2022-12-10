'''
This is the main script for the paper
'''

#from Parameters import returnParameters
from Simulate import Simulate
from Output_Results import Output_Results


#%%


Run_Main                = False
Run_EqualPVs            = False
Run_ADElas_robustness   = False
Run_CRRA1_robustness    = False
Run_CRRA3_robustness    = False
Run_Rfree_robustness    = True
Run_Rspell_robustness   = False
Run_LowerUBnoB          = False


Run_Dict = dict()
Run_Dict['Run_Baseline']            = True
Run_Dict['Run_Recession ']          = True
Run_Dict['Run_Check_Recession']     = True
Run_Dict['Run_UB_Ext_Recession']    = True
Run_Dict['Run_TaxCut_Recession']    = True
Run_Dict['Run_Check']               = True
Run_Dict['Run_UB_Ext']              = True
Run_Dict['Run_TaxCut']              = True
Run_Dict['Run_AD ']                 = True
Run_Dict['Run_1stRoundAD']          = True
Run_Dict['Run_NonAD']               = True

#%% Execute main Simulation

if Run_Main:

    figs_dir = './Figures/CRRA2/'    
    Simulate(Run_Dict,figs_dir,Parametrization='Baseline')    
    Output_Results('./Figures/CRRA2/','./Figures/','./Tables/CRRA2/',Parametrization='Baseline')
    
if Run_EqualPVs:
        
    Run_Dict['Run_1stRoundAD']          = False   
    figs_dir = './Figures/CRRA2_PVSame/'
    Simulate(Run_Dict,figs_dir,Parametrization='CRRA2_PVSame')
    Output_Results('./Figures/CRRA2_PVSame/','./Figures/CRRA2_PVSame/','./Tables/',Parametrization='CRRA2_PVSame')







#%% 
if Run_ADElas_robustness:
    
    Run_Dict['Run_1stRoundAD']          = False

    figs_dir = './Figures/ADElas/'
    Simulate(Run_Dict,figs_dir,Parametrization='ADElas')
    Output_Results('./Figures/ADElas/','./Figures/ADElas/','./Tables/ADElas/',Parametrization='ADElas')
     
    figs_dir = './Figures/ADElas_PVSame/'
    Simulate(Run_Dict,figs_dir,Parametrization='ADElas_PVSame')
    Output_Results('./Figures/ADElas_PVSame/','./Figures/ADElas_PVSame/','./Tables/ADElas_PVSame/',Parametrization='ADElas_PVSame')


    
#%% Execute robustness run
        
if Run_CRRA1_robustness:
    
    Run_Dict['Run_1stRoundAD']          = False

    figs_dir = './Figures/CRRA1/'
    Simulate(Run_Dict,figs_dir,Parametrization='CRRA1')
    Output_Results('./Figures/CRRA1/','./Figures/CRRA1/','./Tables/CRRA1/',Parametrization='CRRA1')
     
    figs_dir = './Figures/CRRA1_PVSame/'
    Simulate(Run_Dict,figs_dir,Parametrization='CRRA1_PVSame')
    Output_Results('./Figures/CRRA1_PVSame/','./Figures/CRRA1_PVSame/','./Tables/CRRA1_PVSame/',Parametrization='CRRA1_PVSame')

if Run_CRRA3_robustness:
    
    Run_Dict['Run_1stRoundAD']          = False

    figs_dir = './Figures/CRRA3/'
    Simulate(Run_Dict,figs_dir,Parametrization='CRRA3')
    Output_Results('./Figures/CRRA3/','./Figures/CRRA3/','./Tables/CRRA3/',Parametrization='CRRA3')
     
    figs_dir = './Figures/CRRA3_PVSame/'
    Simulate(Run_Dict,figs_dir,Parametrization='CRRA3_PVSame')
    Output_Results('./Figures/CRRA3_PVSame/','./Figures/CRRA3_PVSame/','./Tables/CRRA3_PVSame/',Parametrization='CRRA3_PVSame')



#%%

if Run_Rfree_robustness:
    
    Run_Dict['Run_1stRoundAD']          = False

    figs_dir = './Figures/Rfree_1005/'
    Simulate(Run_Dict,figs_dir,Parametrization='Rfree_1005')
    Output_Results('./Figures/Rfree_1005/','./Figures/Rfree_1005/','./Tables/Rfree_1005/',Parametrization='Rfree_1005')
     
    figs_dir = './Figures/Rfree_1005_PVSame/'
    Simulate(Run_Dict,figs_dir,Parametrization='Rfree_1005_PVSame')
    Output_Results('./Figures/Rfree_1005_PVSame/','./Figures/Rfree_1005_PVsame/','./Tables/Rfree_1005_PVsame/',Parametrization='Rfree_1005_PVSame')

    figs_dir = './Figures/Rfree_1015/'
    Simulate(Run_Dict,figs_dir,Parametrization='Rfree_1015')
    Output_Results('./Figures/Rfree_1015/','./Figures/Rfree_1015/','./Tables/Rfree_1015/',Parametrization='Rfree_1015')
     
    figs_dir = './Figures/Rfree_1015_PVSame/'
    Simulate(Run_Dict,figs_dir,Parametrization='Rfree_1015_PVSame')
    Output_Results('./Figures/Rfree_1015_PVSame/','./Figures/Rfree_1015_PVSame/','./Tables/Rfree_1015_PVSame/',Parametrization='Rfree_1015_PVSame')


if Run_Rspell_robustness:
    
    Run_Dict['Run_1stRoundAD']          = False

    figs_dir = './Figures/Rspell_4/'
    Simulate(Run_Dict,figs_dir,Parametrization='Rspell_4')
    Output_Results('./Figures/Rspell_4/','./Figures/Rspell_4/','./Tables/Rspell_4/',Parametrization='Rspell_4')
     
    figs_dir = './Figures/Rspell_4_PVSame/'
    Simulate(Run_Dict,figs_dir,Parametrization='Rspell_4_PVSame')
    Output_Results('./Figures/Rspell_4_PVSame/','./Figures/Rspell_4_PVSame/','./Tables/Rspell_4_PVSame/',Parametrization='Rspell_4_PVSame')


if Run_LowerUBnoB:

    Run_Dict['Run_1stRoundAD']          = False

    figs_dir = './Figures/LowerUBnoB/'
    Simulate(Run_Dict,figs_dir,Parametrization='LowerUBnoB')
    Output_Results('./Figures/LowerUBnoB/','./Figures/LowerUBnoB/','./Tables/LowerUBnoB/',Parametrization='LowerUBnoB')
   
    figs_dir = './Figures/LowerUBnoB_PVSame/'
    Simulate(Run_Dict,figs_dir,Parametrization='LowerUBnoB_PVSame')
    Output_Results('./Figures/LowerUBnoB_PVSame/','./Figures/LowerUBnoB_PVSame/','./Tables/LowerUBnoB_PVSame/',Parametrization='LowerUBnoB_PVSame')
 