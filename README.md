# Project Goal

Find drivers of wine quality and create a model that can accurately predict wine quality while performing better than baseline.

# Project Description

Using the dataset for wine quality from Data World, look for any effects of physicochemical properties of wine on its resulting quality.

# Initial Questions/Thoughts

We believe that density, pH, alcohol, and sulphates will be the best predictors of wine quality.

# Data Dictionary

TBD

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