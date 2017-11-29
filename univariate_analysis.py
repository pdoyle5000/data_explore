#!/usr/bin/env python3
from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import scipy.stats as stats

INPUT_FILE = "CustomerData_Merrimack.xlsx"
OUTPUT_FILE = "data_output/summary.txt"
OUTPUT_CSV = "data_output/final.csv"
PK = "CustomerID"
FORCE_TYPES = {
        'UnionMember': 'category',
        'Region': 'category',
        'Retired': 'category',
        'LoanDefault': 'category',
        'MaritalStatus': 'category',
        'CarsOwned': 'category',
        'CarOwnership': 'category',
        'CarBrand': 'category',
        'PoliticalPartyMem': 'category',
        'Votes': 'category',
        'CreditCard': 'category',
        'ActiveLifestyle': 'category',
        'EquipmentRental': 'category',
        'CallingCard': 'category',
        'WirelessData': 'category',
        'Multiline': 'category',
        'Pager': 'category',
        'Internet': 'category',
        'VM': 'category',
        'CallerID': 'category',
        'CallWait': 'category',
        'CallForward': 'category',
        'ThreeWayCalling': 'category',
        'EBilling': 'category',
        'OwnsPC': 'category',
        'OwnsMobileDevice': 'category',
        'OwnsGameSystem': 'category',
        'OwnsFax': 'category',
        'NewsSubscriber': 'category'
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
            chisq = categorical_association(db, attribute)
            f.write("Categorical Breakdown:\n")
            f.write(str(tmp_df))
            f.write("\n\nPossible Associations:\n")
            for k, v in chisq.items():
                f.write(k + ": P-value (Chi-Square-Test) ->" + str(v) + '\n')

        elif str(dt) == "int64" or str(dt) == "float64":
            mean = db[attribute].mean()
            median = db[attribute].median()
            skew = db[attribute].skew()
            kurt = db[attribute].kurt()
            f.write("Normality Snapshot:\n")
            f.write("Mean: " + str(mean) + '\n')
            f.write("Median: " + str(median) + '\n')
            f.write("Skew: " + str(skew) + '\n')
            f.write("Kurtosis: " + str(kurt) + '\n')
            f.write("\nPossible Associations:\n")
            corvals = correlation(db, attribute)
            for k,v in corvals.items():
                f.write(k + ": R-Squared (Correlation) ->" + str(v) + '\n')

            # Compare against categorical
            anova_results = anova(db, attribute)
            for k, v in anova_results.items():
                f.write(k + ": P-value (ANOVA) ->" + str(v) + '\n')
            

        f.write('\n-----------------------------------------------------\n\n')
        num_attributes += 1

    f.close()

def clean_attributes(db):
    # Address null values in TownSize and convert type
    db['TownSize'].fillna("NA", inplace=True)
    db['TownSize'] = db['TownSize'].astype('category')

    # Address null values in HomeOwner and convert type
    db['HomeOwner'].fillna("NA", inplace=True)
    db['HomeOwner'] = db['HomeOwner'].astype('category')

    # Address null values in Gender and convert type
    db['Gender'].fillna("NA", inplace=True)
    db['Gender'] = db['Gender'].astype('category')

    # Address null values in JobCategory and convert type
    db['JobCategory'].fillna("Unknown", inplace=True)
    db['JobCategory'] = db['JobCategory'].astype('category')

    # Replace all the null values in CommuteTime with 0
    db['CommuteTime'].fillna(0, inplace=True)
    db['CommuteTime'] = db['CommuteTime'].astype(np.int64)

    # Replace all null values in VoiceOverTenure with 0
    db['VoiceOverTenure'].fillna(0, inplace=True)
    db['VoiceOverTenure'] = db['VoiceOverTenure'].astype(np.float64)

    # Create categorical variable from HHIncome
    db['HouseholdIncome'] = hhincome_to_cat(db['HHIncome'])

    # Create categorical variable from EducationYears
    db['Education'] = educationyrs_to_cat(db['EducationYears'])

    # Create categorical variable from DebtToIncomeRatio
    db['InHeavyDebt'] = debt_ratio_categorical(db['DebtToIncomeRatio'])

    # Address null values in HouseholdSize
    db['HouseholdSize'] = fill_null_household_size(db)

    # Create categorical variable HasPets
    db['HasPets'] = has_pets(db)

    # Replace all the -$1000 values in CarValue with NaN
    db['CarValue'].replace(float(-1000), np.nan, inplace=True)
    
    # Create AvgPhoneBill variable
    db['AvgPhoneBill'] = db['VoiceOverTenure'].divide(db['PhoneCoTenure'])

    cols_to_delete = [
            'HHIncome', 
            'EducationYears', 
            'DebtToIncomeRatio', 
            'CreditDebt', 
            'OtherDebt', 
            'NumberCats', 
            'NumberDogs', 
            'NumberBirds', 
            'NumberPets',
            'CarBrand']

    db.drop(cols_to_delete, inplace=True, axis=1)
    return db

def hhincome_to_cat(hhincome):
    newincome = []
    for val in hhincome:
        if val < 20000:
            newincome.append("0 to 20")
        elif val >= 20000 and val < 40000:
            newincome.append("20 to 39")
        elif val >= 40000 and val < 80000:
            newincome.append("40 to 79")
        elif val >= 80000 and val < 200000:
            newincome.append("80 to 200")
        else:
            newincome.append("above 200")
    return pd.Series(newincome, dtype='category')

def educationyrs_to_cat(yrs):
    edu = []
    for val in yrs:
        if val < 12:
            edu.append(0)
        elif val >= 12 and val < 17:
            edu.append(1)
        else:
            edu.append(2)
    return pd.Series(edu, dtype='category')

def debt_ratio_categorical(debt):
    d = []
    for val in debt:
        if val <= 15.0:
            d.append(0)
        else:
            d.append(1)
    return pd.Series(d, dtype='category')

def fill_null_household_size(db):
    new_sizes = []
    for index, row in db.iterrows():
        if pd.isnull(row['HouseholdSize']):
            if row['MaritalStatus'] == 'Married':
                new_sizes.append(2)
            else:
                new_sizes.append(1)
        else:
            new_sizes.append(row['HouseholdSize'])
    return pd.Series(new_sizes, dtype='int64')

def has_pets(db):
    haspets = []
    for index, row in db.iterrows():
        if row['NumberPets'] > 0 or row['NumberCats'] > 0 or row['NumberDogs'] > 0 or row['NumberBirds'] > 0:
            haspets.append(1)
        else:
            haspets.append(0)
    return pd.Series(haspets, dtype='category')

def categorical_association(db, att):
    result_dict = {}

    for attribute in db.select_dtypes('category'):
        if attribute != att:
            xtab = pd.crosstab(db[att], db[attribute])
            chisq, p, _, _ = chi2_contingency(xtab)
            # only show me p values that indicate association.
            if p < 0.05:
                result_dict[attribute] = p

    return result_dict

def correlation(db, att):
    corr_dict = {}

    for attribute in db.select_dtypes(np.number):
        if attribute != att:
            rsq = db[att].corr(db[attribute])
            if rsq > 0.8:
                corr_dict[attribute] = rsq

    return corr_dict

def anova(db, att):
    anova_dict = {}
    for attribute in db.select_dtypes('category'):
        grouped_data = db.groupby(attribute)[att].apply(list)
        _, anova_val = stats.f_oneway(*grouped_data)
        if anova_val < 0.05:
            anova_dict[attribute] = anova_val
    return anova_dict

def main():
    db = import_original_data()
    clean_db = clean_attributes(db)
    print_summary(clean_db)
    clean_db.to_csv(OUTPUT_CSV)

if __name__ == "__main__":
    main()
