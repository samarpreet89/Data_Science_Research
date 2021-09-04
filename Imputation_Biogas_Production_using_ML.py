import os
import sys
import pandas as pd
import wwtp_model
import utils as ut
import numpy as np
#import statsmodels.api as sm


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


# Biogas production

# Read in Imputed data after imputation of Power Demand and Solids variables
print("Read in Input Data")
input_data = pd.read_csv('new_output_5.csv')[0:76033]
output_data = pd.read_csv('new_output_5.csv')[0:76033]

# Create copy of earlier Imputed dataset, and impute missing values for
# Biogas Production in this dataset

solids_variables = ['thickened_was_to_dig_m3pd','thickened_primary_to_dig_m3pd','fog_to_dig_m3pd']

print("Starting Task: Impute missing values for Biogas Production")

#model_path = os.path.join(root_dir,'Data','Prepped', 'biogas_production_model.pkl')
timevars_5 = ut.get_quarter_varnames() + ut.get_weekday_varnames() + ut.get_hour_varnames()
print("hi")
prepped_data_5 = ms.reprep_data(
    input_data= input_data,
    lagged_variables = ['influent_flow_m3pd','biogas_production_m3pd'],
    daily_lagged_variables = solids_variables,
    other_variables = ['biogas_production_m3pd']
)

print("Training Model")

model_5 = ms.run_model_selection(
    data = prepped_data_5,
    outcome = 'biogas_production_m3pd',
    lagged_variables = [],
    daily_lagged_variables = solids_variables,
    other_variables = timevars_5
)

print("Model Trained")


# Identify Indices of missing Biogas Production values
biogas_prod_data = input_data['biogas_production_m3pd']

idx_missing_biogas_prod =biogas_prod_data[biogas_prod_data.isnull()].index.tolist()

no_missing = len(idx_missing_biogas_prod)
print("No of missing values are")
print(no_missing)

all_data_after_imputation_5 = input_data

x_new = model_5['X']
x_data_5 = np.array(x_new)
print(x_new)
m = x_new.index[0]
n = x_new.shape[1]
print("m and n")
print(m)
print(n)

print("Starting Imputation Process")

j = 1
for i in range(0,76000):
    if i in idx_missing_biogas_prod:
        #print(j)
        # x_data starts from index 96
        biogas_prod_val= model_5['model'].predict(x_data_5[i - m - 1].reshape(1,n))
        #print(biogas_prod_val)
        all_data_after_imputation_5.at[i,'biogas_production_m3pd'] = biogas_prod_val
        new_prepped_data_5 = ms.reprep_data(input_data = all_data_after_imputation_5,
                                            lagged_variables = ['influent_flow_m3pd','biogas_production_m3pd'],
                                            daily_lagged_variables = solids_variables,
                                            other_variables = ['biogas_production_m3pd'])
        x_new_5,_ = ms.get_model_input_arrays(data = new_prepped_data_5,
                                              outcome = 'biogas_production_m3pd',
                                              lagged_variables = [],
                                              daily_lagged_variables = [],
                                              other_variables = [])
        m = x_new_5.index[0]
        n = x_new_5.shape[1]
        x_data_5 = np.array(x_new_5)
        all_data_after_imputation_5.at[i,'biogas_production_m3pd'] = biogas_prod_val
        #biogas_prod_data_now = all_data_after_imputation_5['biogas_production_m3pd']
        #idx_missing_biogas_prod_now = \
        #            biogas_prod_data_now[biogas_prod_data_now.isnull()].index.tolist()
        #k = idx_missing_biogas_prod_now
        print(j, i, x_data_5.shape[0])
        j += 1

print("Imputation process completed")

output_data['biogas_production_m3pd'] = all_data_after_imputation_5['biogas_production_m3pd']

# Check if all missing values of Biogas Production have been imputed

biogas_prod_data_after_imputation = output_data['biogas_production_m3pd']

idx_missing_biogas_prod_after_imputation = \
    biogas_prod_data_after_imputation[biogas_prod_data_after_imputation.isnull()].index.tolist()


print("No of missing value for Biogas Production now is : ")
print(len(idx_missing_biogas_prod_after_imputation))

output_data.to_csv('new_output_6.csv',header=True, index=False)

print("#############################################################################################")

print("Next variable")
