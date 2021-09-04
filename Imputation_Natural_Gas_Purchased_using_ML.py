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

# Natural gas purchased

# Read in input data after imputation of Power Demand, Solids variables,
# Biogas Production,Gross generation, Purchased kW
print("Read in Input Data")
input_data = pd.read_csv('new_output_8.csv')
output_data = pd.read_csv('new_output_8.csv')

# Create copy of earlier imputed dataset, and impute missing values
# of Natural Gas Production in this dataset

#all_data_after_imp_8 = pd.read_csv('data_after_imputation_7.csv')


print("Starting Task: Impute missing values for Natural Gas Purchased")

timevars8 = ut.get_quarter_varnames() + ut.get_weekday_varnames() + ut.get_hour_varnames()
# Train 2 models based on whether one or both generators were running

lagged_variables = ['power_demand_kW','natural_gas_purchased_m3pd','biogas_production_m3pd']
prepped_data_8 = ms.reprep_data(
    input_data= input_data,
    lagged_variables = lagged_variables,
    other_variables = ['natural_gas_purchased_m3pd','gross_generation_kW']
)

#list1 = list(prepped_data_8.columns)

# create subset when both generators were running
clean_bool_81 = (prepped_data_8['gross_generation_kW'] >= 650) & \
                (prepped_data_8['natural_gas_purchased_m3pd'] > 0)

# create subset when only 1 generator was running
clean_bool_82 = (prepped_data_8['gross_generation_kW'] < 650) & \
                (prepped_data_8['natural_gas_purchased_m3pd'] > 0)


prepped_data_81 = ms.reprep_data(
    input_data= input_data,
    lagged_variables = lagged_variables,
   other_variables = ['natural_gas_purchased_m3pd'],
    clean_bool= clean_bool_81
)


prepped_data_82 = ms.reprep_data(
    input_data= input_data,
    lagged_variables = lagged_variables,
    other_variables = ['natural_gas_purchased_m3pd'],
    clean_bool= clean_bool_82,
)

print("Training Models")

model_81 = ms.run_model_selection(
    prepped_data_81,
    outcome = 'natural_gas_purchased_m3pd',
    lagged_variables = ['power_demand_kW','biogas_production_m3pd'],
    other_variables = timevars8
)

model_82 = ms.run_model_selection(
    prepped_data_82,
    outcome = 'natural_gas_purchased_m3pd',
    lagged_variables = ['power_demand_kW','biogas_production_m3pd'],
    other_variables = timevars8
)

print("Models Trained")


# Identify Indices of missing 'Natural Gas Purchased' variable

nat_gas_purchased_data = input_data['natural_gas_purchased_m3pd']

#idx_81 = nat_gas_purchased_data[clean_bool_81].index

idx_missing_nat_gas_purchased = nat_gas_purchased_data[nat_gas_purchased_data.isnull()].index.tolist()

# identify indices in dataset when Gross Generation >= 650 and < 650 - used at prediction stage

list_81 = list(clean_bool_81)
list_82 = list(clean_bool_82)

idx_81 = []
idx_82 = []

for i in range(len(list_81)):
    if list_81[i] == True:
        idx_81.append(i)

for i in range(len(list_82)):
    if list_82[i] == True:
        idx_82.append(i)

x_new_81 = model_81['X']
print("m is")
print(x_new_81.shape[1])
x_data_81 = np.array(x_new_81)
print(x_new_81)

x_new_82 = model_82['X']
x_data_82 = np.array(x_new_82)
print(x_new_82)

# create array of gross generation values - used to decide which of the
# 2 models is to be used while predicting for a particular missing values
# of natural gas purchased
gross_gen_vals = np.array(input_data['gross_generation_kW'])

print("Starting Imputation Process")

# X, Y/outcome, idx, covariate names
# returns imputed Y and/or updated daraframe and lag terms

all_data_after_imp_8 = input_data


for i in range(input_data.shape[0]):
    if i in idx_missing_nat_gas_purchased:
        if gross_gen_vals[i] >= 650:
            # Use model 1 for prediction: Model 1 trained on datapoints for which gross gen >= 650
            # identify preceeding entry in dataset that is to be used to predict this missing value
            k1 = -1
            for j in range(len(idx_81)):
                if idx_81[j] < i:
                    k1 += 1
                else:
                    break
            print(i,k1)
            #m = x_new_81.index[0]
            n1 = x_new_81.shape[1]
            nat_gas_purchased_val = model_81['model'].predict(x_data_81[k1].reshape(1,n1))
            all_data_after_imp_8.at[i,'natural_gas_purchased_m3pd'] = nat_gas_purchased_val
            prepped_data_8_1 = ms.reprep_data(
                input_data= all_data_after_imp_8,
                lagged_variables = lagged_variables,
                other_variables = ['natural_gas_purchased_m3pd','gross_generation_kW']
            )
            #list2 = list(prepped_data_7_1.columns)
            clean_bool_81_2 = (prepped_data_8_1['gross_generation_kW'] >= 650) & \
                          (prepped_data_8_1['natural_gas_purchased_m3pd'] > 0)
            new_prepped_data_81 = ms.reprep_data(input_data = all_data_after_imp_8,
                                             lagged_variables = lagged_variables,
                                             other_variables = ['natural_gas_purchased_m3pd'],
                                             clean_bool= clean_bool_81_2
                                             )
            x_new_81,_ = ms.get_model_input_arrays(data = new_prepped_data_81,
                                                outcome = 'natural_gas_purchased_m3pd',
                                                lagged_variables = ['power_demand_kW','biogas_production_m3pd'],
                                                other_variables = [])
            x_data_81 = np.array(x_new_81)
            list_81 = list(clean_bool_81_2)
            idx_81 = []
            for i in range(len(list_81)):
                if list_81[i] == True:
                    idx_81.append(i)
            print("done")
        else:
            # Use model 2  : trained on datapoints where gross generation < 650
            ##
            # Use model 1 for prediction: Model 1 trained on datapoints for which gross gen >= 650
            # identify preceeding entry in dataset that is to be used to predict this missing value
            k2 = -1
            for j in range(len(idx_82)):
                if idx_82[j] < i:
                    k2 += 1
                else:
                    break
            print(i,k2)
            n2 = x_new_82.shape[1]
            nat_gas_purchased_val = model_82['model'].predict(x_data_82[k2].reshape(1,n2))
            all_data_after_imp_8.at[i,'natural_gas_purchased_m3pd'] = nat_gas_purchased_val
            prepped_data_8_2 = ms.reprep_data(
                input_data= all_data_after_imp_8,
                lagged_variables = lagged_variables,
                other_variables = ['natural_gas_purchased_m3pd','gross_generation_kW']
            )
            #list2 = list(prepped_data_7_2.columns)
            clean_bool_82_2 = (prepped_data_8_2['gross_generation_kW'] < 650) & \
                              (prepped_data_8_2['natural_gas_purchased_m3pd'] > 0)
            new_prepped_data_82 = ms.reprep_data(input_data = all_data_after_imp_8,
                                                 lagged_variables = lagged_variables,
                                                 other_variables = ['natural_gas_purchased_m3pd'],
                                                 clean_bool= clean_bool_82_2
                                                 )
            x_new_82,_ = ms.get_model_input_arrays(data = new_prepped_data_82,
                                                    outcome = 'natural_gas_purchased_m3pd',
                                                    lagged_variables = ['power_demand_kW','biogas_production_m3pd'],
                                                    other_variables = [])
            x_data_82 = np.array(x_new_82)
            list_82 = list(clean_bool_82_2)
            idx_82 = []
            for i in range(len(list_82)):
                if list_82[i] == True:
                    idx_82.append(i)
            print("done 2")


print("Imputation process completed")

output_data['natural_gas_purchased_m3pd'] = all_data_after_imp_8['natural_gas_purchased_m3pd']

# Checking if all missing values of Natural Gas Purchased have been imputed
nat_gas_pur_m3pd_data_after_imputation = output_data['natural_gas_purchased_m3pd']

idx_missing_nat_gas_pur_m3pd_after_imputation = \
    nat_gas_pur_m3pd_data_after_imputation[nat_gas_pur_m3pd_data_after_imputation.isnull()].index.tolist()


print("No of missing value for Natural Gas Generation now is : ")
print(len(idx_missing_nat_gas_pur_m3pd_after_imputation))

# Generate csv file of FINAL imputed complete dataset !!!! :-D
output_data.to_csv('new_output_9.csv', header=True, index=False)

print("#############################################################################################")

