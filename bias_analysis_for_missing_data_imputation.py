import os
import sys
import pandas as pd
import wwtp_model
import utils as ut
import numpy as np
import random
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
#import statsmodels.api as sm
random.seed(25)

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

print("Read in Input Data")
input_data = pd.read_csv('SVCW_merged_v4_input.csv')[0:76033]
idx1 = list(range(input_data.shape[0]))

#output_data = pd.read_csv('SVCW_merged_v3_input.csv')

imputed_data = pd.read_csv('new_output_9.csv')
test_data = pd.read_csv('new_output_9.csv')

power_demand_data = input_data['power_demand_kW']
idx2 = power_demand_data[power_demand_data.isnull()].index.tolist()
print(idx2)

print("No of missing values was: ")
print(len(idx2))

for i in range(len(idx2)):
    idx1.remove(idx2[i])

idx_test = sorted(random.sample(idx1, 76))

print(idx_test)

y_true = []
for i in range(len(idx_test)):
    idx = idx_test[i]
    y_true.append(test_data.at[idx,'power_demand_kW'])
    test_data.at[idx,'power_demand_kW'] = None


#power_demand_data_2 = imputed_data['power_demand_kW']
#idx3 = power_demand_data_2[power_demand_data_2.isnull()].index.tolist()
#print(len(idx3))
#print(idx3)



# Impute missing values for Power Demand using Machine Learning (ElasticNet)
print("Starting Task 1: Impute missing values for Power Demand")


timevars1 = ut.get_quarter_varnames() + ut.get_weekday_varnames() + ut.get_hour_varnames()
prepped_data1 = ms.reprep_data(
    input_data = test_data,
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

#power_demand_data = input_data['power_demand_kW']
#idx_missing_power_demand = power_demand_data[power_demand_data.isnull()].index.tolist()

# create copy of given dataset and impute missing values in this dataset
all_data_after_imputation = test_data

x_new = model1['X']
#print(x_new)
x_data = np.array(x_new)

y_pred = []

for i in range(input_data.shape[0]):
    if i in idx_test:
        m = x_new.index[0]
        n = x_new.shape[1]
        power_val= model1['model'].predict(x_data[i-m-1].reshape(1,n))
        y_pred.append(power_val.item())
        all_data_after_imputation.at[i,'power_demand_kW'] = power_val
        #output_data.at[i,'power_demand_kW'] = power_val
        new_prepped_data = ms.reprep_data(input_data = all_data_after_imputation,
                                          lagged_variables = ['influent_flow_m3pd','power_demand_kW'],
                                          other_variables = ['power_demand_kW'])
        x_new,_ = ms.get_model_input_arrays(data = new_prepped_data,
                                            outcome = 'power_demand_kW',
                                            lagged_variables = ['influent_flow_m3pd'],
                                            other_variables = [])
        x_data = np.array(x_new)
        #print(i)



print("Imputation process completed")

test_data['power_demand_kW'] = all_data_after_imputation['power_demand_kW']

# Checking if all missing values of Power Demand have been imputed
power_demand_data_after_imputation = test_data['power_demand_kW']

idx_missing_power_demand_after_imputation = \
    power_demand_data_after_imputation[power_demand_data_after_imputation.isnull()].index.tolist()

#print(idx_missing_power_demand_after_imputation)

print("No of missing value for Power Demand now is : ")
print(len(idx_missing_power_demand_after_imputation))

print(y_true)
print(y_pred)

#print("Mean Squared error")
#print(mean_squared_error(y_true,y_pred))


def percentage_error(y_true, y_pred):
    per_err = []
    #res = np.empty(actual.shape)
    for i in range(len(y_true)):
        if y_true[i] != 0:
            per_err.append((y_true[i] - y_pred[i]) / y_true[i])
        else:
            per_err.append(y_true[i] / np.mean(y_true))
    return per_err


def MAPE(y_true,y_pred):
    mape = np.mean(np.abs(percentage_error(y_true, y_pred))) * 100
    return mape

def MAPE_CI(y_true,y_pred):
    ape = np.abs(percentage_error(y_true, y_pred))
    mape = np.mean(ape) * 100
    se = (np.std(ape)) * 100 /len(ape)
    mape_ci_lower = mape + 1.96*se
    mape_ci_higher = mape - 1.96*se
    return (mape_ci_lower,mape_ci_higher)

def MPE_CI(y_true, y_pred):
    pe = percentage_error(y_true, y_pred)
    mpe = np.mean(pe) * 100
    se = (np.std(pe)) * 100 /len(pe)
    mpe_ci_lower = mpe + 1.96*se
    mpe_ci_higher = mpe - 1.96*se
    return (mpe_ci_lower,mpe_ci_higher)


def MPE(y_true, y_pred):
    mpe = np.mean(percentage_error(y_true,y_pred)) * 100
    return mpe

print("Mean Absolute Percentage error")
print(MAPE(y_true,y_pred))

print("Mean Percentage error")
print(MPE(y_true,y_pred))

print("95% CI for MAPE")
print(MAPE_CI(y_true,y_pred))

print("95% CI for MPE")
print(MPE_CI(y_true,y_pred))




print("#############################################################################################")

print("Next variable")

#####################################################################################################

