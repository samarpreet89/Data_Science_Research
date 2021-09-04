import pandas as pd

wwtp_data = pd.read_csv('CWNS Processed_csv.csv')


flow_list = list(wwtp_data['Total_Flow'])

# initialize N
N = 100

# Indices of N largest elements in list
# using sorted() + lambda + list slicing
res = sorted(range(len(flow_list)), key = lambda sub: flow_list[sub])[-N:]

top_100 = pd.DataFrame()

for i in range(len(res)):
    top_100 = top_100.append(wwtp_data.loc[res[i]])


top_100.to_csv("100_largest_WWTP_data.csv")

