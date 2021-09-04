import os
import sys
import pandas as pd
import wwtp_model
import utils as ut
import numpy as np
import statsmodels.api as sm


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



# Purchased kW

# Read in Input data after Imputation of Power Demand, Solids Variables,
# Biogas production and Gross Generation

print("Read in Input data")
input_data = pd.read_csv('new_output_7.csv')[0:76033]
#output_data = pd.read_csv('output_6.csv')

# using Imputation Strategy:
# Purchased kW = Power Demand - Gross Generation

# Get Power Demand, Gross Generation and Purchased_kW data
purchased_kw_data = input_data['purchased_kW']
power_demand_kw_data = np.array(input_data['power_demand_kW'])
gross_gen_data = np.array(input_data['gross_generation_kW'])


# Get indices of missing Purchased_kW values

idx_missing_purchaded_kw = purchased_kw_data[purchased_kw_data.isnull()].index.tolist()


print("Starting Imputation Process")

# create copy of input imputed dataset, and impute missing values of
# Purchased_kW in this dataset

output_data = input_data


for i in range(input_data.shape[0]):
    if i in idx_missing_purchaded_kw:
        input_data.at[i,'purchased_kW'] = power_demand_kw_data[i] - gross_gen_data[i]


print("Imputation process completed")

# Checking if all missing values of Purchased_kW have been imputed

output_data['purchased_kW'] = input_data['purchased_kW']

purchased_kw_data_after_imputation = output_data['purchased_kW']

idx_missing_purchased_kw_after_imputation = \
    purchased_kw_data_after_imputation[purchased_kw_data_after_imputation.isnull()].index.tolist()


print("No of missing value for Purchased kW now is : ")
print(len(idx_missing_purchased_kw_after_imputation))

output_data.to_csv('new_output_8.csv',header=True)


print("#############################################################################################")
