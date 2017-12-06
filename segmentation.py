#!/usr/bin/env python3
from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from univariate_analysis import FORCE_TYPES as forced_types
from univariate_analysis import OUTPUT_CSV as FROM_ANALYSIS

# Force the rest to categorical
def import_data():
    forced_types['Gender'] = 'category'
    forced_types['JobCategory'] = 'category'
    forced_types['HouseholdIncome'] = 'category'
    return pd.read_csv(FROM_ANALYSIS, dtype=forced_types)

def main():
    df = import_data()
    print(list(df))
    print(df.dtypes)

if __name__ == "__main__":
    main()
