import pandas as pd
 
xls_file = pd.read_excel(io='./2021-10-29-210422-김진영-intra.xlsx',sheet_name='Sheet2')


new_data = pd.DataFrame(xls_file.iloc[18:29,1:9])

new_data = new_data.reset_index(drop=True,inplace=False)

# new_data = new_data.set_index('Unnamed: 1', drop=False)

print(new_data)
# new_data = new_data.drop(['index'], axis = 1)

# print(new_data)

# # print(new_data)

# new_data.to_csv("asd.csv", mode='w', index =None)






