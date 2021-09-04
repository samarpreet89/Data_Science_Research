import os
import sys
import pandas as pd
import wwtp_model
import utils as ut
import numpy as np
#import statsmodels.api as sm

print("Read in Input Data")
input_data = pd.read_csv('new_output_2.csv')
output_data = pd.read_csv('new_output_2.csv')


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


# Solids variables

# Create copy of earlier imputed dataset and impute missing values in this dataset

print("Starting Task: Impute missing values for Solids Variables")

all_data_after_imputation = input_data

solids_variables = ['thickened_was_to_dig_m3pd','thickened_primary_to_dig_m3pd','fog_to_dig_m3pd']
for idx,outcome in enumerate(solids_variables):
    prepped_data = ms.reprep_data(
        input_data= all_data_after_imputation,
        lagged_variables = ['influent_flow_m3pd'] + [outcome],
        other_variables = [outcome],
        #daily = True
    )
    if outcome != 'fog_to_dig_m3pd':
        print("Starting Task: Impute missing values for " + str(outcome))
        clean_bool = prepped_data[outcome] > 0.5
        prepped_data2 = ms.reprep_data(
            input_data=all_data_after_imputation,
            lagged_variables = ['influent_flow_m3pd'] + [outcome],
            other_variables = [outcome],
            clean_bool = clean_bool,
            #    daily = True
        )
        print("Training Model")
        timevars2 = ut.get_quarter_varnames() + ut.get_weekday_varnames() + ut.get_hour_varnames()
        model2 = ms.run_model_selection(
            prepped_data2,
            outcome = outcome,
            lagged_variables = ['influent_flow_m3pd'],
            other_variables = timevars2,
            #daily = True
        )
        print("Model Trained")
        print("Starting Imputation Process for " + str(outcome))
        # Determine Indices of Missing values for Solids Variables
        solids_data = input_data[outcome]
        idx_missing_solids_data = solids_data[solids_data.isnull()].index.tolist()
        print("No of missing values for " + str(outcome) + " is: ")
        print(len(idx_missing_solids_data))
        x_new = model2['X']
        print("x_new is")
        print(x_new)
        m = x_new.index[0]
        n = x_new.shape[1]
        print("The value of m is : ")
        print(m)
        print("Value of n is : ")
        print(n)
        x_data = np.array(x_new)
        for i in range(all_data_after_imputation.shape[0]):
            if i in idx_missing_solids_data:
                print(i)
                solids_val= model2['model'].predict(x_data[i-m-1].reshape(1,n))
                all_data_after_imputation.at[i,outcome] = solids_val
                new_prepped_data_2 = ms.reprep_data(input_data = all_data_after_imputation,
                                                    lagged_variables = ['influent_flow_m3pd',outcome],
                                                    other_variables = [outcome])
                x_new,_ = ms.get_model_input_arrays(data = new_prepped_data_2,
                                                     outcome = outcome,
                                                     lagged_variables = ['influent_flow_m3pd'],
                                                     other_variables = [])
                x_data = np.array(x_new)
        print("Imputation process completed for " + str(outcome))
        # Checking if all missing values have been imputed
        output_data[outcome] = all_data_after_imputation[outcome]
        solid_data_after_imputation = output_data[outcome]
        idx_missing_solids_after_imputation = \
            solid_data_after_imputation[solid_data_after_imputation.isnull()].index.tolist()
        print("No of missing value for " + str(outcome) + " now is : ")
        print(len(idx_missing_solids_after_imputation))
        print("Next solid variable")


print("#############################################################################################")
print("Next variable")


print("Starting Task: Impute missing values for FOG to Digestor")

# Determine indices of missing values for FOG to Digestor
fog_values = input_data['fog_to_dig_m3pd']
idx_missing_fog_values = fog_values[fog_values.isnull()].index.tolist()

print("No of missing values for FOG to Digestor are: ")
print(len(idx_missing_fog_values))

# 'FOG to Digestor' has only 97 missing values -  0.13% of total values

# Trying using Cubic Spline to impute missing values : upto Index 45000
# After index 45000 overwhelmingly zero values for 'FOG to Digestor' - impute all these to zero

#x_train = []
#y_train = []

#for i in range(45000):
#    if i not in idx_3:
#        x_train.append(i)
#        y_train.append(fog.loc[i])

#print("Training Model")

#fit_model = sm.GLM(y_train, x_train).fit()

#print("Model trained")

#preds = fit_model.predict(idx_3)

#for i in range(input_data.shape[0]):
#    j = 0
#    if i < 45000:
#        if i not in idx_missing_fog_values:
#            continue
#        else:
#            all_data_after_imputation_4.at[i,'fog_to_dig_m3pd'] = preds[j]
#            j += 1
#    else:
#        if i not in idx_missing_fog_values:
#            continue
#        else:
#            all_data_after_imputation_4.at[i,'fog_to_dig_m3pd'] = 0


# However, since 93.9% of non-missing 'FOG to Digestor' values are zero, using cubic
# spline to impute missing values generates very small predictions (< 0.5) for these
# missing values, when non-zero non-missing values generally lie between 4 and 7

# Hence, instead of using Cubic Spline to Impute missing values, we set all missing
# values (0.13% of all values) for 'FOG to Digestor' to be zero.

# create copy for earlier Imputed Dataset and impute missing values in this dataset.

#all_data_after_imputation_2 = all_data_after_imputation

print("Starting imputation process for FOG to Digestor")

for i in range(input_data.shape[0]):
    if i in idx_missing_fog_values:
        output_data.at[i,'fog_to_dig_m3pd'] = 0


print("Imputation completed for FOG to Digestor")

# Checking if all missing values of 'FOG to Digestor' have been imputed


fog_data_after_imputation = output_data['fog_to_dig_m3pd']
idx_missing_fog_after_imputation = \
    fog_data_after_imputation[fog_data_after_imputation.isnull()].index.tolist()


print("No of missing value for FOG to Digestor now is : ")
print(len(idx_missing_fog_after_imputation))

output_data.to_csv('new_output_5.csv',header=True, index=False)


print("#############################################################################################")
print("Next variable")


