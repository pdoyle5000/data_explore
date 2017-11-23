'''This is a script that imports an xlsx file and performs unvariate analysis.
pip3 install pandas xlrd numpy'''

from __future__ import print_function
import pandas as pd
import numpy as np

"""
Cannot force these types to integer.  Coming in as floats due to blanks.
FORCE_TYPES = {
    'HouseholdSize': np.int64,
    'NumberPets': np.int64,
    'NumberCats': np.int64,
    'NumberDogs': np.int64,
    'HomeOwner': np.int64}
"""
def main():
    ''' main function that executs on python3 univariate_analysis.py'''
    customer_db = pd.read_excel('CustomerData_Merrimack.xlsx', na_values="NA")

    # Split the db into three different data-types for clairty
    continuous_vars = customer_db.select_dtypes(include=[np.float])
    integer_vars = customer_db.select_dtypes(include=[np.int64])
    categorical_vars = customer_db.select_dtypes(include=[np.object])

    print("Categorical Variables:")
    print(categorical_vars.describe())

    print("Continuous Variables:")
    print(continuous_vars.describe())

    print("Integer Variables:")
    print(integer_vars.describe())

    for key in continuous_vars:
        print(continuous_vars[key].kurt())

    continuous_vars.CreditDebt.hist()

if __name__ == "__main__":
    main()
