```python
import pandas as pd

# Step 1: Read the binary parquet file created by our pipeline
saved_cohort_data = pd.read_parquet("../data/extract_patient_cohort.parquet")

# Step 2: Print out the internal properties of our data frame for verification
print(f"Total Rows Extracted: {len(saved_cohort_data)}")
print(f"Data Columns Present: {list(saved_cohort_data.columns)}")

# Step 3: Render the clear dataframe spreadsheet directly inside the notebook window
saved_cohort_data.head(10)
```

    Total Rows Extracted: 364627
    Data Columns Present: ['subject_id', 'gender', 'anchor_age', 'anchor_year', 'dod']





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>subject_id</th>
      <th>gender</th>
      <th>anchor_age</th>
      <th>anchor_year</th>
      <th>dod</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10000032</td>
      <td>F</td>
      <td>52</td>
      <td>2180</td>
      <td>2180-09-09</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10000048</td>
      <td>F</td>
      <td>23</td>
      <td>2126</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10000058</td>
      <td>F</td>
      <td>33</td>
      <td>2168</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10000068</td>
      <td>F</td>
      <td>19</td>
      <td>2160</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10000084</td>
      <td>M</td>
      <td>72</td>
      <td>2160</td>
      <td>2161-02-13</td>
    </tr>
    <tr>
      <th>5</th>
      <td>10000102</td>
      <td>F</td>
      <td>27</td>
      <td>2136</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>10000108</td>
      <td>M</td>
      <td>25</td>
      <td>2163</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>10000115</td>
      <td>M</td>
      <td>24</td>
      <td>2154</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>10000117</td>
      <td>F</td>
      <td>48</td>
      <td>2174</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10000161</td>
      <td>M</td>
      <td>60</td>
      <td>2163</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
import pandas as pd

# 1. Read the newly generated diagnosis cohort file
diabetes_cohort = pd.read_parquet("../data/diabetes_cohort_list.parquet")

# 2. Display basic dimensions (Rows and Columns)
print("STUDY COHORT DIMENSIONS ")
print(f"Total Rows (Patients): {diabetes_cohort.shape[0]:,}")
print(f"Total Columns: {diabetes_cohort.shape[1]}")
print(f"Column Names: {list(diabetes_cohort.columns)}")

# 3. Render the first few rows as an interactive visual table
print("First 5 rows of diabetes cohort:")
diabetes_cohort.head(5)
```

    STUDY COHORT DIMENSIONS 
    Total Rows (Patients): 100,585
    Total Columns: 4
    Column Names: ['subject_id', 'hadm_id', 'icd_code', 'icd_version']
    First 5 rows of diabetes cohort:





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>subject_id</th>
      <th>hadm_id</th>
      <th>icd_code</th>
      <th>icd_version</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10000635</td>
      <td>20642640</td>
      <td>E119</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10000635</td>
      <td>26134563</td>
      <td>25000</td>
      <td>9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10000980</td>
      <td>20897796</td>
      <td>E1122</td>
      <td>10</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10000980</td>
      <td>24947999</td>
      <td>25000</td>
      <td>9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10000980</td>
      <td>25911675</td>
      <td>E118</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>




```python

```


```python
import pandas as pd
import os

# Point directly to your processed directory
PROCESSED_DATA_PATH = "../data/processed/kdigo_labeled_features.parquet"

# Check if the file exists from the notebook's relative directory
if not os.path.exists(PROCESSED_DATA_PATH):
    print(f"ERROR: Cannot find {PROCESSED_DATA_PATH}. Make sure your pipeline ran successfully.")
else:
    # Read the master analytical matrix
    master_matrix = pd.read_parquet(PROCESSED_DATA_PATH)
    
    # 1. Print out the matrix properties
    print(f"SUCCESS! Master Matrix Loaded Perfectly.")
    print(f"Final Matrix Size: {master_matrix.shape[0]:,} rows x {master_matrix.shape[1]} columns")
    
    # 2. Display Class Balance Target Breakdown
    print("Target Class Breakdown (KDIGO Stages):")
    print(master_matrix['kdigo_stage'].value_counts().sort_index())
    
    # 3. Render the spreadsheet view inside the notebook window
    print("Feature Matrix Preview:")
    display(master_matrix.head(5))
```

    SUCCESS! Master Matrix Loaded Perfectly.
    Final Matrix Size: 62,831 rows x 10 columns
    Target Class Breakdown (KDIGO Stages):
    kdigo_stage
    0    53393
    1     7065
    2      631
    3     1742
    Name: count, dtype: int64
    Feature Matrix Preview:



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>subject_id</th>
      <th>hadm_id</th>
      <th>icd_code</th>
      <th>icd_version</th>
      <th>glucose_mean</th>
      <th>glucose_cv</th>
      <th>glucose_time_in_range</th>
      <th>creatinine_slope_24h</th>
      <th>baseline_creatinine</th>
      <th>kdigo_stage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10000980</td>
      <td>20897796</td>
      <td>E1122</td>
      <td>10</td>
      <td>159.2</td>
      <td>4.434816</td>
      <td>75.0</td>
      <td>0.2</td>
      <td>2.4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10000980</td>
      <td>25911675</td>
      <td>E118</td>
      <td>10</td>
      <td>159.2</td>
      <td>4.434816</td>
      <td>75.0</td>
      <td>0.0</td>
      <td>2.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10000980</td>
      <td>26913865</td>
      <td>25000</td>
      <td>9</td>
      <td>159.2</td>
      <td>4.434816</td>
      <td>75.0</td>
      <td>0.0</td>
      <td>2.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10000980</td>
      <td>29654838</td>
      <td>25000</td>
      <td>9</td>
      <td>159.2</td>
      <td>4.434816</td>
      <td>75.0</td>
      <td>0.0</td>
      <td>1.4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10000980</td>
      <td>29659838</td>
      <td>E1121</td>
      <td>10</td>
      <td>159.2</td>
      <td>4.434816</td>
      <td>75.0</td>
      <td>0.1</td>
      <td>2.2</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>

