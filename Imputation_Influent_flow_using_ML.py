import os
import sys
import pandas as pd
import wwtp_model
import utils as ut
import numpy as np
from scipy.interpolate import CubicSpline
import datetime
#import statsmodels.api as sm

print("Read in Input Data")
input_data = pd.read_csv('SVCW_merged_v4_input.csv')
output_data = pd.read_csv('SVCW_merged_v4_input.csv')

# Import wwtp_model
currentdir = os.getcwd()
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)


# Load model parameters and create model selection class instance
root_dir = "C:/Users/HP/IdeaProjects/Summer RA/Latest Pull/energy_inflows/energy_inflows"
parameters_path = os.path.join(root_dir,'input_parameters.xls')
#charts_dir = os.path.join(root_dir,'Project Output','Meetings','6-2-21')
parameters = ut.set_parameters(parameters_path)
ms = wwtp_model.model_selection(parameters)


influent_flow_data = input_data['influent_flow_m3pd']
idx_missing_influent_flow = influent_flow_data[influent_flow_data.isnull()].index.tolist()

print("No of missing values of Influent Flow")
print(len(idx_missing_influent_flow))


fit_data_1 = input_data[input_data['influent_flow_m3pd'].notnull()].reset_index()

x_fit_1 = np.array(fit_data_1['index'])
y_fit_1 = np.array(fit_data_1['influent_flow_m3pd'])


# calculate natural cubic spline polynomials
cs1 = CubicSpline(x_fit_1,y_fit_1,bc_type='natural')


for i in range(len(idx_missing_influent_flow)):
    idx = idx_missing_influent_flow[i]
    output_data.at[idx,'influent_flow_m3pd'] = cs1(idx)

influent_flow_data_after_imp = output_data['influent_flow_m3pd']
idx_missing_influent_flow_after_imp = influent_flow_data_after_imp[influent_flow_data_after_imp.isnull()].index.tolist()

print("missing val now is :")
print(len(idx_missing_influent_flow_after_imp))

output_data.to_csv('new_output_1.csv', header=True, index=False)

print("################################################################################")