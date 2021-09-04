import os
import sys
import pandas as pd
import wwtp_model
import utils as ut
import numpy as np
#import statsmodels.api as sm

print("Read in Input Data")
input_data = pd.read_csv('new_output_1.csv')
output_data = pd.read_csv('new_output_1.csv')


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


# Impute missing values for Gross Demand using Machine Learning (ElasticNet)
print("Starting Task 1: Impute missing values for Gross Demand")


timevars1 = ut.get_quarter_varnames() + ut.get_weekday_varnames() + ut.get_hour_varnames()
prepped_data1 = ms.reprep_data(
    input_data= input_data,
    lagged_variables = ['influent_flow_m3pd','power_demand_kW'],
    other_variables = ['power_demand_kW']
)

print("Training model")

model1 = ms.run_model_selection(
    data = prepped_data1,
    outcome = 'power_demand_kW',
    lagged_variables = ['influent_flow_m3pd'],
    other_variables = timevars1,
)

print("Model trained")


# Predicting missing values


print("Starting Imputation Process")

# Determine indices of missing values of Power Demand in dataset

gross_demand_data = input_data['power_demand_kW']
idx_missing_gross_demand = gross_demand_data[gross_demand_data.isnull()].index.tolist()

# create copy of given dataset and impute missing values in this dataset
all_data_after_imputation = input_data


x_new = model1['X']
print(x_new)
print(x_new.index[0])
x_data = np.array(x_new)

# X

#power_demand_vals = []

#for i in range(all_data.shape[0]):
#    if i not in idx_missing_power_demand:
#        continue
#        #power_demand_vals.append(all_data['power_demand_kW'].loc[i])
#    else:
#        if i <= 315:
#            continue
#        else:
#            power_val= model['model'].predict(x_data.loc[i-1:i])
#            all_data_after_imputation.at[i,'power_demand_kW'] = power_val
#            new_prepped_data = ms.reprep_data(input_data = all_data_after_imputation,
#                lagged_variables = ['influent_flow_m3pd','power_demand_kW'],
#                                            other_variables = ['power_demand_kW'])
#            x_data,_ = ms.get_model_input_arrays(data = new_prepped_data,
#                                               outcome = 'power_demand_kW',
#                                               lagged_variables = ['influent_flow_m3pd'],
#                                               other_variables = [])


for i in range(input_data.shape[0]):
    if i in idx_missing_gross_demand:
        gross_demand_val= model1['model'].predict(x_data[i-97].reshape(1,193))
        all_data_after_imputation.at[i,'power_demand_kW'] = gross_demand_val
        #output_data.at[i,'power_demand_kW'] = power_val
        new_prepped_data = ms.reprep_data(input_data = all_data_after_imputation,
                                          lagged_variables = ['influent_flow_m3pd','power_demand_kW'],
                                          other_variables = ['power_demand_kW'])
        x_new,_ = ms.get_model_input_arrays(data = new_prepped_data,
                                             outcome = 'power_demand_kW',
                                             lagged_variables = ['influent_flow_m3pd'],
                                             other_variables = [])
        x_data = np.array(x_new)
        print(i)


print("Imputation process completed")

output_data['power_demand_kW'] = all_data_after_imputation['power_demand_kW']

# Checking if all missing values of Power Demand have been imputed
gross_demand_data_after_imputation = output_data['power_demand_kW']

idx_missing_gross_demand_after_imputation = \
    gross_demand_data_after_imputation[gross_demand_data_after_imputation.isnull()].index.tolist()

#print(idx_missing_power_demand_after_imputation)

print("No of missing value for Power Demand now is : ")
print(len(idx_missing_gross_demand_after_imputation))

#output_data.to_csv('new_output_2.csv',header=True, index=False)

print("#############################################################################################")

print("Next variable")



#####################################################################################################


