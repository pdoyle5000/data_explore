```pip3 dependencies: pandas xlrd numpy scipy sklearn matplotlib```

To run:
`./univariate_analysis.py`

This script pulls in CustomerData_Merrimack.xlsx, scrubs the data, outputs a `final.csv` of the final scrub into the data_output folder and a `summary.txt` file.

Summary.txt

Each variable was cleaned, summarized and compared against other variables using the appropriate statistical test (ANOVA, correlation, Chi-Square) dependant on the data types being compared.

`./segmentation.py`

This is a script that takes the output of univariate_analysis `final.csv` and performs a few different clustering algorithms, outputs raw data, ANOVA statistical testing and visualizations of the tested attributes.  The script must have comments removed to produce the visualizations.  the k-means or wards algorithm `.show()` functions, `input()` calls and function call itself need to be uncommented in order to display the scatter plots.
