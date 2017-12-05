#!/usr/bin/env python3
from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from univariate_analysis import FORCE_TYPES
from univariate_analysis import OUTPUT_CSV as FROM_ANALYSIS

def import_data():
    return pd.read_csv(FROM_ANALYSIS, dtype=FORCE_TYPES)

def main():
    db = import_data()
    print(list(db))

if __name__ == "__main__":
    main()
