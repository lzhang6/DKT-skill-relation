import pandas as pd
import numpy as np
import code0_parameter as code0
import code1_data as code1
import aux
np.set_printoptions(threshold=np.inf)


filename = './data/sim_problem_data.csv'
data = pd.read_csv(filename)
data.rename(columns={'Student': 'user_id','Correctness': 'correct'}, inplace=True)
"""
k = aux.getUserQuesNumList(data['user_id'])
print (k)
print (len(k))
"""

print (np.unique(data['skill_id']))
print (len(data))

