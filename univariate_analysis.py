#!/usr/bin/env python3
from __future__ import print_function
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

INPUT_FILE = "CustomerData_Merrimack.xlsx"
OUTPUT_FILE = "data_output/import_summary.txt"
PK = "CustomerID"
FORCE_TYPES = {
        'UnionMember': 'category',
        'Region': 'category',
        }

def import_original_data():
    customer_db = pd.read_excel(INPUT_FILE, dtype=FORCE_TYPES)
    return customer_db

def print_summary(db):
    f = open(OUTPUT_FILE, 'w')
    num_attributes = 1
    for attribute in db:
        f.write(str(num_attributes) + ': ' + db[attribute].name + ', ' + str(db[attribute].dtype) + '\n')
        f.write(db[attribute].describe().to_string() + '\n\n')

        dt = db[attribute].dtype
        if str(dt) == "category":
            tmp_df = db.groupby(attribute)[PK].nunique()
            chisq = categorical_independence(db, attribute)
            f.write("Categorical Breakdown:\n")
            f.write(str(tmp_df))
            f.write("\n\nIndependence Issues:\n")
            for k, v in chisq.items():
                f.write(k + ": p-value ->" + str(v))

        elif str(dt) == "int64":
            mean = db[attribute].mean()
            median = db[attribute].median()
            skew = db[attribute].skew()
            kurt = db[attribute].kurt()
            f.write("Normality Snapshot:\n")
            f.write("Mean: " + str(mean) + '\n')
            f.write("Median: " + str(median) + '\n')
            f.write("Skew: " + str(skew) + '\n')
            f.write("Kurtosis: " + str(kurt) + '\n')

        f.write('\n---------------------\n\n')
        num_attributes += 1

    f.write(str(db.dtypes))
    f.close()

def clean_attributes(db):
    # Address null values in town size and convert type
    db['TownSize'].fillna("NA", inplace=True)
    db['TownSize'] = db['TownSize'].astype('category')

    # Address null values in gender and convert type
    db['Gender'].fillna("NA", inplace=True)
    db['Gender'] = db['Gender'].astype('category')

    return db

def categorical_independence(db, att):
    result_dict = {}

    for attribute in db.select_dtypes('category'):
        if attribute != att:
            xtab = pd.crosstab(db[att], db[attribute])
            chisq, p, _, _ = chi2_contingency(xtab)
            if p < 0.05:
                result_dict[attribute] = p

    return result_dict


def main():
    db = import_original_data()
    clean_db = clean_attributes(db)
    print_summary(clean_db)

    

if __name__ == "__main__":
    main()
