import wbdata
import pandas as pd
import numpy as np
import datetime
import linear_regress

turkey = [i['id'] for i in wbdata.get_country(country_id=("TUR"))]
# I could not get rid of the loop above. If I take get_country() function outside
# of the loop and delete the loop, it returns an error because of non-iterable object.
variables = {"SP.DYN.LE00.IN": "Life expectancy at birth, total (years)", 
             "per_sa_allsa.avt_pop_tot": "Average per capita transfer - All Social Assistance"}
dataframe = wbdata.get_dataframe(variables, country=turkey, convert_date=True)
# In order to get rid of the missing data, I have used pd.dropna() function here.
dataframe.dropna(inplace=True)
dataframe.to_csv('RegressionData_HW2.csv')
# In order to extract arrays for the linear_regress function, I have used .iloc function of Pandas.
ind_var = dataframe.iloc[:,0]
dep_var = dataframe.iloc[:,1:]

linear_regress.linear_regress(ind_var, dep_var)