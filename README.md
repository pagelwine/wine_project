# Project Goal

Find drivers of wine quality and create a model that can accurately predict wine quality while performing better than baseline.

# Project Description

Using the dataset for wine quality from Data World, look for any effects of physicochemical properties of wine on its resulting quality.

# Initial Questions/Thoughts

We believe that density, pH, alcohol, and sulphates will be the best predictors of wine quality.

# Data Dictionary

Column Name | Description | Key
--- | --- | ---
fixed_acidity | the difference between total acidity and volatile acidity | milliequivalents per liter (int)
volatile_acidity | measure of volatile acids (acids that readily evaporate) | grams of acetic acid per liter (int)
citric_acid | weak organic acid naturally occurring in fruits | 
residual_sugar | amount of sugar remaining after fermentation | grams per liter (int)
chlorides | amount of salt in wine | 
free_sulfur_dioxide | free form of SO$\2$, used as preservative | 
total_sulfur_dioxide | total of free SO$\2$ and SO$\2$ bound to other chemicals | 
density | mass per unit volume of wine at 20°C | 
pH | measure of wine from basic to acidic | [0-14], wines are usually between [3-4]
sulphates | additive that contributes to SO$\2$ levels | 
alcohol | percent alcohol content of wine | percentage (float) 
quality | quality of wine (numeric) | [0-9]
wine_type | type of wine | [Red, White]
quality_bin | quality of wine (category) | [Good, Bad]

# Steps to Reproduce

Data was acquired from: https://data.world/food/wine-quality.
Clone the repo and run through the final report.

# Project Plan

## Acquire

- Acquire dataset from link
- Add additional column for wine_type
- Merge two dfs into one to create full df
- Cache full df for future use
- View data .info, .describe, .shape

## Prepare

- Adjust column names to be Python-readable
- View/correct datatypes
- Handle nulls (potential imputing)
- Visualize full dataset for univariate exploration (histograms and boxplots)
    - Handle outliers
- Get rid of unnecessary columns
- Splitting data

### Pre-processing

- Scaling data on train
- Encoding any necessary columns 
- Document how we're changing the data

## Exploration

- Use unscaled data for multivariate exploration
    - Hypothesize
    - Visualize
    - Run stats tests
        - Run chi-squared test on clusters vs. target
    - Summarize
- Potential Feature Engineering

### Exploration Summary

### Initial Questions/Thoughts

# Modeling

- Use scaled/encoded data
- Split into X_variables and y_variables
- Determine evaluation metrics
    - Establishing baseline
- Run different models on train/validate
- Pick best model and evaluate on test

# Conclusion

# Recommendations