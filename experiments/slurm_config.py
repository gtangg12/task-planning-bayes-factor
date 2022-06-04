''' Configuration file for MIT Satori slurm resources

    Update file with your own Satori settings and preferences
'''
import os


# maximum number of jobs allocated to user
SLURM_MAX_CONCURRENT_RUNNING_JOBS = 4 

# template file for running experiments
SLURM_TEMPLATE_PATH = os.path.dirname(__file__) + '/template.sh' 
