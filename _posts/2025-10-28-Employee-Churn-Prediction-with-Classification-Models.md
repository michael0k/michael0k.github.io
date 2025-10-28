---
title: Employee Churn Prediction with Classification Models
date: 2025-10-28 10:00:00 -0500
toc: true
toc_sticky: true
categories: [PYTHON , DEMO, VISUALIZATION]
tags: [python,machine learning, pandas, scikit-learn ,sql, demo, predictive modeling, BigQuery, looker studio, google colab]
comments:
  provider: "utterances"
  utterances:
    theme: "github-dark" # "github-dark"
    issue_term: "pathname"
    label: "comment" # Optional - must be existing label.
---

## Scenario 

We are aiding a company that is having problems retaining employees. We are here to proactively identify employees that are of high churn risk. 

>**Objective:** Build a machine learning model trained on previous data that can predict a new employee's likelihood of leaving 
{: .notice--primary}


Deliverable form: 
A report/dashboard

### Analysis Questions 
1. What is causing employees to leave ? 
2. Who is predicted to leave ?
3. Are employees satisfied ? 
4. What departments have the most churn ? 



## Data Sources 


Training data: [`tbl_hr_data.csv`](https://github.com/michael0k/projects-and-demos/blob/75b7b32de11739c2b3b8f22e95f153f84ed3acae/employee_churn/tbl_hr_data.csv)

Data for analysis: [`tbl_new_employees.csv`](https://github.com/michael0k/projects-and-demos/blob/75b7b32de11739c2b3b8f22e95f153f84ed3acae/employee_churn/tbl_new_employees.csv)




## Tools Utilized 

1. [Google BigQuery](https://cloud.google.com/bigquery?hl=en)
2. [Google Colab](https://colab.research.google.com/)
3. [Google Looker Studio](https://lookerstudio.google.com/)


![image-center](/assets/images/2025-10-28-Employee-Churn-Prediction-with-Classification-Models/mermaid_chart.png){: .align-center}

### Required Libraries (Python) 

Since we will be developing our model in Google Colab, we have a number of libraries that we will import and utilize in the project. 

* `pandas`: For data manipulation and analysis.
* `numpy`: For numerical operations, especially for handling arrays and mathematical functions.
* `scikit-learn`: For machine learning models (linear regression & multiple linear regression), data splitting, and evaluation metrics.
* `seaborn`: For advanced statistical data visualization.
* `matplotlib.pyplot`: For creating static, interactive, and animated visualizations.



## 1. Google BigQuery: Creating a table view 

We began by uploading the following CSVs to Google BigQuery and importing their data into tables. 

`tbl_hr_data.csv` - The historical data that we will use to train and test our models on. 

`tbl_new_employees.csv` - The actual data which our model will run its predictions on for our client. 


```sql
SELECT * ,"Original" as Type FROM `my-project-echurn102025.employeedata.tbl_hr_data`
UNION ALL 
SELECT * , "Pilot" as Type FROM `my-project-echurn102025.employeedata.tbl_new_employees`
```

We proceeded to save the results of this query as a [view](https://docs.cloud.google.com/bigquery/docs/views) in Google BigQuery named `tbl_full_data` 




## 2. Google Colab: Connect to BigQuery

**Remark.** You can confirm the soundness of the code snippets below by downloading a copy of the [Jupyter notebook](https://github.com/michael0k/projects-and-demos/blob/7201cba0bd8705215fde9659df787a1529d950c4/employee_churn/Employee_Churn_Analysis.ipynb) that contains all of the code below. 
{: .notice--info}

```python
from google.cloud import bigquery
from google.colab import auth
import numpy as np
import pandas as pd

auth.authenticate_user()

project_id = 'my-project-echurn102025' #project id from BigQuery
client = bigquery.Client(project = project_id , location = 'US')
```


```python
datatset_reference = client.dataset('employeedata' , project = project_id)
dataset = client.get_dataset(datatset_reference)
table_reference = dataset.table('tbl_hr_data')
table = client.get_table(table_reference)
table.schema
```




    [SchemaField('satisfaction_level', 'FLOAT', 'NULLABLE', None, None, (), None),
     SchemaField('last_evaluation', 'FLOAT', 'NULLABLE', None, None, (), None),
     SchemaField('number_project', 'INTEGER', 'NULLABLE', None, None, (), None),
     SchemaField('average_montly_hours', 'INTEGER', 'NULLABLE', None, None, (), None),
     SchemaField('time_spend_company', 'INTEGER', 'NULLABLE', None, None, (), None),
     SchemaField('Work_accident', 'INTEGER', 'NULLABLE', None, None, (), None),
     SchemaField('Quit_the_Company', 'INTEGER', 'NULLABLE', None, None, (), None),
     SchemaField('promotion_last_5years', 'INTEGER', 'NULLABLE', None, None, (), None),
     SchemaField('Departments', 'STRING', 'NULLABLE', None, None, (), None),
     SchemaField('salary', 'STRING', 'NULLABLE', None, None, (), None),
     SchemaField('employee_id', 'STRING', 'NULLABLE', None, None, (), None)]




```python
table_reference_new = dataset.table('tbl_new_employees')
table_new = client.get_table(table_reference_new)
```


```python
#create dataframes from BigQuery tables
df = client.list_rows(table = table).to_dataframe()
df2 = client.list_rows(table = table_new).to_dataframe() #contains unseen final data to be used for predictions
```


```python
df.info()
df2.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 15004 entries, 0 to 15003
    Data columns (total 11 columns):
     #   Column                 Non-Null Count  Dtype  
    ---  ------                 --------------  -----  
     0   satisfaction_level     15004 non-null  float64
     1   last_evaluation        15004 non-null  float64
     2   number_project         14999 non-null  Int64  
     3   average_montly_hours   15004 non-null  Int64  
     4   time_spend_company     14999 non-null  Int64  
     5   Work_accident          15000 non-null  Int64  
     6   Quit_the_Company       15004 non-null  Int64  
     7   promotion_last_5years  15004 non-null  Int64  
     8   Departments            15004 non-null  object 
     9   salary                 15004 non-null  object 
     10  employee_id            15004 non-null  object 
    dtypes: Int64(6), float64(2), object(3)
    memory usage: 1.3+ MB
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 100 entries, 0 to 99
    Data columns (total 11 columns):
     #   Column                 Non-Null Count  Dtype  
    ---  ------                 --------------  -----  
     0   satisfaction_level     100 non-null    float64
     1   last_evaluation        100 non-null    float64
     2   number_project         100 non-null    Int64  
     3   average_montly_hours   100 non-null    Int64  
     4   time_spend_company     100 non-null    Int64  
     5   Work_accident          100 non-null    Int64  
     6   Quit_the_Company       100 non-null    Int64  
     7   promotion_last_5years  100 non-null    Int64  
     8   Departments            100 non-null    object 
     9   salary                 100 non-null    object 
     10  employee_id            100 non-null    object 
    dtypes: Int64(6), float64(2), object(3)
    memory usage: 9.3+ KB

Here you can see that our training data is quite large relative to the data in `df2` which we will be using for our final predictions. 


## 3. Data preprocessing

Tasks performed
- One-hot encoding categorical features (salary and department)
- Changing the case of features to lower case
- Fixing any typos or inconsistent feature names


```python
df_processed = pd.DataFrame(df)
df2_processed = pd.DataFrame(df2)

df_processed = pd.get_dummies(df, columns=['salary','Departments'],prefix=['salary','department'],prefix_sep='_' , dtype = int)
cols= df_processed.columns.tolist()
df_processed.columns = [x.lower() for x in cols]
df_processed.columns

#2nd dataframe

df2_processed = pd.get_dummies(df2, columns=['salary' , 'Departments'], prefix =['salary', 'department'] , prefix_sep='_' , dtype =int)
cols2 = df2_processed.columns.tolist()
df2_processed.columns = [x.lower() for x in cols2]
df2_processed.columns
```




    Index(['satisfaction_level', 'last_evaluation', 'number_project',
           'average_montly_hours', 'time_spend_company', 'work_accident',
           'quit_the_company', 'promotion_last_5years', 'employee_id',
           'salary_high', 'salary_low', 'salary_medium', 'department_it',
           'department_randd', 'department_accounting', 'department_hr',
           'department_management', 'department_marketing',
           'department_product_mng', 'department_sales', 'department_support',
           'department_technical'],
          dtype='object')




```python
df_processed.rename(columns = {"average_montly_hours" : "average_monthly_hours"}, inplace = True)
df2_processed.rename(columns = {"average_montly_hours" : "average_monthly_hours"},inplace = True)
```

## 4. Build Models

Now we will initialize three different classification models and assess which one is the best choice for our particular use.


Models:
- [K-Nearest Neighbor (KNN)](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier)
- [Support Vector Machine (SVM)](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC)
- [Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)



```python
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split , GridSearchCV , StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
```



```python
#create our data frames for training
X = df_processed.drop(columns = ['quit_the_company' , 'employee_id'])
y = df_processed['quit_the_company']

X_train , X_test, y_train, y_test = train_test_split(X , y , test_size = 0.2 , random_state = 42)

nan_total_train = X_train.isnull().sum() #tally NaN values in training data Before processing
nan_total_test = X_test.isnull().sum()


#tally of nan values in our dataframes
print(f"Original number of NaN values in our training dataframe is : {nan_total_train}\n")
print("______________________________________________________\n")
print(f"Original number of NaN values in our testing dataframe is: {nan_total_test}")
```

    Original number of NaN values in our training dataframe is: satisfaction_level        0
    last_evaluation           0
    number_project            4
    average_monthly_hours     0
    time_spend_company        4
    work_accident             3
    promotion_last_5years     0
    salary_high               0
    salary_low                0
    salary_medium             0
    department_it             0
    department_randd          0
    department_accounting     0
    department_hr             0
    department_management     0
    department_marketing      0
    department_product_mng    0
    department_sales          0
    department_support        0
    department_technical      0
    dtype: int64
    
    ______________________________________________________
    
    Original number of NaN values in our testing dataframe is : satisfaction_level        0
    last_evaluation           0
    number_project            1
    average_monthly_hours     0
    time_spend_company        1
    work_accident             1
    promotion_last_5years     0
    salary_high               0
    salary_low                0
    salary_medium             0
    department_it             0
    department_randd          0
    department_accounting     0
    department_hr             0
    department_management     0
    department_marketing      0
    department_product_mng    0
    department_sales          0
    department_support        0
    department_technical      0
    dtype: int64


Now we'll optimize our KNN model with the use of [pipelines](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html).


```python
from sklearn.impute import SimpleImputer

#initalize the pipeline and it's transformers / estimators
pipeline = Pipeline(steps=[
('imputer', SimpleImputer(strategy='mean')), #replaces missing values with the mean of the feature
('scaler' , StandardScaler()), #scale the feature values to ensure "equal" contributions to KNN model
('pca', PCA()), #Principal Component Analysis - to reduce features to the minimal set of orthogonal components
('knn', KNeighborsClassifier()) #our KNN model
])

#the PCA and KNN parameters that we will be assessing our model with
param_grid = {'pca__n_components' : [2,3],
              'knn__n_neighbors' : [2,3,4,5,6,7] #number of nearest points used to classify a given observation
              }

#cross validation via 5 Strafied Folds
cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
```

Iterate through the KNN parameters until we find the optimal combination, using [`GridSearchCV()`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html).


```python
best_model_knn = GridSearchCV(estimator = pipeline ,
                          param_grid = param_grid,
                          cv = cv ,
                          scoring = 'accuracy',
                          verbose = 2
                          )
```


```python
best_model_knn.fit(X_train, y_train)
```




<style>#sk-container-id-2 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: #000;
  --sklearn-color-text-muted: #666;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-2 {
  color: var(--sklearn-color-text);
}

#sk-container-id-2 pre {
  padding: 0;
}

#sk-container-id-2 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-2 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-2 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-2 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-2 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-2 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-2 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-2 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-2 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-2 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-2 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-2 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-2 label.sk-toggleable__label {
  cursor: pointer;
  display: flex;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
  align-items: start;
  justify-content: space-between;
  gap: 0.5em;
}

#sk-container-id-2 label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

#sk-container-id-2 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-2 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-2 div.sk-label label.sk-toggleable__label,
#sk-container-id-2 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-2 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-2 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-2 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-2 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-2 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 0.5em;
  text-align: center;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-2 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-2 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-2 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-2 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=StratifiedKFold(n_splits=5, random_state=42, shuffle=True),
             estimator=Pipeline(steps=[(&#x27;imputer&#x27;, SimpleImputer()),
                                       (&#x27;scaler&#x27;, StandardScaler()),
                                       (&#x27;pca&#x27;, PCA()),
                                       (&#x27;knn&#x27;, KNeighborsClassifier())]),
             param_grid={&#x27;knn__n_neighbors&#x27;: [2, 3, 4, 5, 6, 7],
                         &#x27;pca__n_components&#x27;: [2, 3]},
             scoring=&#x27;accuracy&#x27;, verbose=2)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-7" type="checkbox" ><label for="sk-estimator-id-7" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>GridSearchCV</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.model_selection.GridSearchCV.html">?<span>Documentation for GridSearchCV</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted"><pre>GridSearchCV(cv=StratifiedKFold(n_splits=5, random_state=42, shuffle=True),
             estimator=Pipeline(steps=[(&#x27;imputer&#x27;, SimpleImputer()),
                                       (&#x27;scaler&#x27;, StandardScaler()),
                                       (&#x27;pca&#x27;, PCA()),
                                       (&#x27;knn&#x27;, KNeighborsClassifier())]),
             param_grid={&#x27;knn__n_neighbors&#x27;: [2, 3, 4, 5, 6, 7],
                         &#x27;pca__n_components&#x27;: [2, 3]},
             scoring=&#x27;accuracy&#x27;, verbose=2)</pre></div> </div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-8" type="checkbox" ><label for="sk-estimator-id-8" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>best_estimator_: Pipeline</div></div></label><div class="sk-toggleable__content fitted"><pre>Pipeline(steps=[(&#x27;imputer&#x27;, SimpleImputer()), (&#x27;scaler&#x27;, StandardScaler()),
                (&#x27;pca&#x27;, PCA(n_components=3)),
                (&#x27;knn&#x27;, KNeighborsClassifier(n_neighbors=2))])</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-9" type="checkbox" ><label for="sk-estimator-id-9" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>SimpleImputer</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.impute.SimpleImputer.html">?<span>Documentation for SimpleImputer</span></a></div></label><div class="sk-toggleable__content fitted"><pre>SimpleImputer()</pre></div> </div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-10" type="checkbox" ><label for="sk-estimator-id-10" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>StandardScaler</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.preprocessing.StandardScaler.html">?<span>Documentation for StandardScaler</span></a></div></label><div class="sk-toggleable__content fitted"><pre>StandardScaler()</pre></div> </div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-11" type="checkbox" ><label for="sk-estimator-id-11" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>PCA</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.decomposition.PCA.html">?<span>Documentation for PCA</span></a></div></label><div class="sk-toggleable__content fitted"><pre>PCA(n_components=3)</pre></div> </div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-12" type="checkbox" ><label for="sk-estimator-id-12" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>KNeighborsClassifier</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.neighbors.KNeighborsClassifier.html">?<span>Documentation for KNeighborsClassifier</span></a></div></label><div class="sk-toggleable__content fitted"><pre>KNeighborsClassifier(n_neighbors=2)</pre></div> </div></div></div></div></div></div></div></div></div></div></div>



If we expand the information under the elements of our pipeline (in the `GridSearchCV` graphic produced in the ouput), we see that our selected parameters were 2 and 3 for the number of nearest neighbors and the PCA components respectively.


Next , we find the accuracy score of our KNN model using the test data.


```python
test_score = best_model_knn.score(X_test,y_test)
print(f"{test_score:.3f}")
```

    0.946



```python
best_model_knn.best_params_
```




    {'knn__n_neighbors': 2, 'pca__n_components': 3}



Now that we have our ideal KNN model. We'll return it's employee churn predictions up ahead. We will also assess how it performed with a Confusion Matrix and some useful metrics. 

Lets initialize pipelines for our other classifier models. Specifically a RandomForest Classifier and a Support Vector Machine model.


```python
pipeline_rf = Pipeline( steps=[
    ('imputer' , SimpleImputer(strategy = 'mean')) , # replace missing values with the mean of the column
    ('scaler' , StandardScaler()),
    ('rfc' , RandomForestClassifier(random_state = 42))
])

pipeline_svm = Pipeline ( steps=[
    ('imputer' , SimpleImputer(strategy = 'mean')) , # replace missing values with the mean of the column
    ('scaler' , StandardScaler()),
    ('svm' , SVC(kernel = 'linear', C=1 , random_state = 42))
])
```

Train the models


```python
model_rf = pipeline_rf.fit(X_train, y_train)
model_svm = pipeline_svm.fit(X_train, y_train)
```

## 5. Evaluating the Classification Models

Tools & metrics that we will use :
- Confusion Matrices
- Classification Reports :
  - Recall
  - Precision
  - F1-Score


### Interpreting Confusion Matrices

Some preliminaries. In this context, since we are assessing the risk of employee churn. A *positive* corresponds to 1 , i.e. employee churn , *negative* corresponds to 0, i.e. employee retained.

So a *true positive* is a correctly predicted case of employee churn while a *false positive* is the case where our model predicted that a given employee has left when they actually haven't. 

**True positive:** prediction $= 1 =$ actual value

**False positive:** prediction $= 1 $ and  actual value $= 0$

**True negative:** prediction $= 0 =$ actual value

**False negative:** prediction $= 0 $ and  actual value $= 1$



```python
y_pred_knn = best_model_knn.predict(X_test)
y_pred_svm = model_svm.predict(X_test)
y_pred_rf = model_rf.predict(X_test)

model_preds = [y_pred_knn, y_pred_svm , y_pred_rf]
class_algs = ['KNN' ,'SVM', 'Random Forest']

#give us the accuracy score and classification reports for our models
for i, j in zip(model_preds,class_algs) :
  print(f"The accuracy score for {j} is {accuracy_score(y_test, i):.3f}\n" )
  print(f"\n The classification report for {j} is: \n")
  print("\n" , classification_report(y_test, i))
  print(f"______________________________________________________\n")

```

    The accuracy score for KNN is 0.946
    
    
     The classification report for KNN is: 
    
    
                   precision    recall  f1-score   support
    
             0.0       0.95      0.98      0.96      2281
             1.0       0.92      0.85      0.88       720
    
        accuracy                           0.95      3001
       macro avg       0.94      0.91      0.92      3001
    weighted avg       0.95      0.95      0.94      3001
    
    ______________________________________________________
    
    The accuracy score for SVM is 0.770
    
    
     The classification report for SVM is: 
    
    
                   precision    recall  f1-score   support
    
             0.0       0.79      0.94      0.86      2281
             1.0       0.55      0.23      0.32       720
    
        accuracy                           0.77      3001
       macro avg       0.67      0.58      0.59      3001
    weighted avg       0.74      0.77      0.73      3001
    
    ______________________________________________________
    
    The accuracy score for Random Forest is 0.991
    
    
     The classification report for Random Forest is: 
    
    
                   precision    recall  f1-score   support
    
             0.0       0.99      1.00      0.99      2281
             1.0       0.99      0.97      0.98       720
    
        accuracy                           0.99      3001
       macro avg       0.99      0.98      0.99      3001
    weighted avg       0.99      0.99      0.99      3001
    
    ______________________________________________________
    


Based on these classification reports I am highly inclined to pick the Random Forest Classifier as our model of choice.

But let's carry on with the next assessment, the confusion matrices. Which will give us a closer look into the number of true or false positive/negatives in our models' predictions.

### Confusion Matrices

Using a combination of [confusion matrices](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html) and [heatmaps](https://seaborn.pydata.org/generated/seaborn.heatmap.html) we will visualize a comparison between our models' predictions.

First we initialize our confusion matrices and then we visualize them using heatmaps from the statstical data visualization library [Seaborn](https://seaborn.pydata.org/index.html).



```python
conf_matrix_knn = confusion_matrix(y_test , y_pred_knn)
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)


fig, axes = plt.subplots(1,3, figsize = (18,7))

#knn heatmap subplot for knn confusion matrix
sns.heatmap(conf_matrix_knn , annot=True , cmap='Blues' , fmt = 'd', ax = axes[0] , xticklabels = 'auto' , yticklabels = 'auto' )
axes[0].set_title("KNN Testing Confusion Matrix")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

#svm heatmap subplot

sns.heatmap(conf_matrix_svm, annot=True, cmap='Blues' , fmt = 'd' , ax = axes[1] , xticklabels = 'auto', yticklabels = 'auto')
axes[1].set_title("SVM Testing Confusion Matrix")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

#rf heatmap subplot

sns.heatmap(conf_matrix_rf, annot=True , cmap='Blues' , fmt = 'd' , ax = axes[2] , xticklabels = 'auto' , yticklabels = 'auto')
axes[2].set_title("Random Forest Testing Confusiion Matrix")
axes[2].set_xlabel("Predicted")
axes[2].set_ylabel("Actual")

plt.tight_layout()
plt.show()
```


    
![image-center](/assets/images/2025-10-28-Employee-Churn-Prediction-with-Classification-Models/output_32_0.png){: .align-center}



Here we can see that our Random Forest classifier is the better performing of the three models.

To further instill the information discussed prior. We can breakdown the predictions of our Random Forest classifier in the following manner:
- True Positives count: 700
- False Positives count: 7
- True Negatives count: 2274
- False Negatives count: 20

Now the results of our classification report should seem clearer.

Below are the formulas for the metrics in our classification report , in respect to employee churn (1).




$$\text{Precision }= \displaystyle\frac{\text{True positives}}{\text{True Positives} + \text{ False Positives}} = \frac{700}{707} \approx 0.99 \\ $$



$$\text{Recall }= \displaystyle\frac{\text{True positives}}{\text{True Positives} + \text{False Negatives}} \frac{700}{720} \approx 0.97 \\ $$




**F1 Score** - The harmonic mean of precison and recall

$$\text{F1 Score }= \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{(2 \cdot 0.99 \cdot 0.97)}{0.99 + 0.97} \approx 0.98 \\ $$




We could try to tune the parameters for our SVM model to see if we can improve it's performance. But "out of the box", with little to no configuration, our random forest classifier seems best suited for our dataset. We will proceed to use it's predictions in our upcoming Looker Studio dashboard.

## 6. Exploring Our Results 

### Which Features Contribute Most to Employee Churn ?

Now we wish to determine how much each feature in our dataset/dataframe contributes to the probability of us losing a given employee. It's clear to imagine why not every feature would have equal or similar contribution to the likelihood of an employee leaving an organization.

We will create a separate two column dataframe for this. Which we will eventually export as a table to BigQuery and incorporate into our Looker Studio dashboard.


```python
feat_imps = model_rf.named_steps['rfc'].feature_importances_
df_feats = pd.DataFrame(zip(X_train.columns, feat_imps), columns=['feature', 'importance'])
df_feats = df_feats.sort_values(by='importance' , ascending = False)
```


```python
#df_feats.set_index('feature')
df_feats
```





  <div id="df-080f1673-6f25-46e7-946f-0f01af98cf78" class="colab-df-container">
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
      <th>feature</th>
      <th>importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>satisfaction_level</td>
      <td>0.318813</td>
    </tr>
    <tr>
      <th>2</th>
      <td>number_project</td>
      <td>0.178543</td>
    </tr>
    <tr>
      <th>4</th>
      <td>time_spend_company</td>
      <td>0.174293</td>
    </tr>
    <tr>
      <th>3</th>
      <td>average_monthly_hours</td>
      <td>0.153096</td>
    </tr>
    <tr>
      <th>1</th>
      <td>last_evaluation</td>
      <td>0.125385</td>
    </tr>
    <tr>
      <th>5</th>
      <td>work_accident</td>
      <td>0.011894</td>
    </tr>
    <tr>
      <th>8</th>
      <td>salary_low</td>
      <td>0.006976</td>
    </tr>
    <tr>
      <th>7</th>
      <td>salary_high</td>
      <td>0.005463</td>
    </tr>
    <tr>
      <th>19</th>
      <td>department_technical</td>
      <td>0.003689</td>
    </tr>
    <tr>
      <th>17</th>
      <td>department_sales</td>
      <td>0.003318</td>
    </tr>
    <tr>
      <th>9</th>
      <td>salary_medium</td>
      <td>0.003255</td>
    </tr>
    <tr>
      <th>18</th>
      <td>department_support</td>
      <td>0.002683</td>
    </tr>
    <tr>
      <th>10</th>
      <td>department_it</td>
      <td>0.001881</td>
    </tr>
    <tr>
      <th>6</th>
      <td>promotion_last_5years</td>
      <td>0.001776</td>
    </tr>
    <tr>
      <th>11</th>
      <td>department_randd</td>
      <td>0.001758</td>
    </tr>
    <tr>
      <th>13</th>
      <td>department_hr</td>
      <td>0.001633</td>
    </tr>
    <tr>
      <th>14</th>
      <td>department_management</td>
      <td>0.001583</td>
    </tr>
    <tr>
      <th>15</th>
      <td>department_marketing</td>
      <td>0.001440</td>
    </tr>
    <tr>
      <th>12</th>
      <td>department_accounting</td>
      <td>0.001345</td>
    </tr>
    <tr>
      <th>16</th>
      <td>department_product_mng</td>
      <td>0.001177</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-080f1673-6f25-46e7-946f-0f01af98cf78')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-080f1673-6f25-46e7-946f-0f01af98cf78 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-080f1673-6f25-46e7-946f-0f01af98cf78');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-317bee2d-28c3-4b0b-bd44-6110cd2462f6">
      <button class="colab-df-quickchart" onclick="quickchart('df-317bee2d-28c3-4b0b-bd44-6110cd2462f6')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-317bee2d-28c3-4b0b-bd44-6110cd2462f6 button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

  <div id="id_c6279374-45fb-4d7d-9deb-c24d0944e0fd">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('df_feats')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_c6279374-45fb-4d7d-9deb-c24d0944e0fd button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('df_feats');
      }
      })();
    </script>
  </div>

    </div>
  </div>




Lets create a stem chart to better visualize how the importances of each feature compares to one another.


```python
plt.stem(df_feats['feature'] , df_feats['importance'] , orientation = 'horizontal', basefmt = '' )
plt.title('Quantified Importances of Features')
plt.ylabel('Features')
plt.xlabel('Feature Importance')
plt.gca().invert_yaxis() #reverse the default asecending order of the y-axis
plt.grid()
plt.show()
```


![image-center](/assets/images/2025-10-28-Employee-Churn-Prediction-with-Classification-Models/output_40_0.png){: .align-center}

As we can see, we have a sparse set of feature variables. In other words, very few of our features contribute to the outcome of our target variable.

$\approx$ 25% of the features (five out of twenty) are responsible for $\approx$ 95% of the total feature importance.

### Create a Dataframe of Our Churn Predictions

Finally, let's create a dataframe of our churn predictions. Which will mirror `df2_processed` except for the two additional columns on it's end. One for the model's churn prediction and the churn probability score for that given employee.

Then we will export this new dataframe , along with our feature importance dataframe, back to BigQuery. So that we can use them in our Looker Studio dashboard.


```python
#set up intermediary dataframes
X2 = df2_processed.drop(columns = ['quit_the_company', 'employee_id'])
df_ids = df2_processed[['quit_the_company' ,'employee_id']]

#X_train, y_train, id_train, X_test, y_test, id_test = train_test_split(X2, y2, df_ids, test_size=0.2 , random_state = 42 )


y_pred_final = model_rf.predict(X2)
y_pred_proba = model_rf.predict_proba(X2)[:,1]

#final df with quit_the_company + predictions + churn proba

df2_final = df2_processed
df2_final['churn_prediction'] = y_pred_final
df2_final['churn_probability'] = y_pred_proba


```






### Examining Our At-Risk Employees

We will create a horizontal bar chart that simply displays the number of employees that are likely to churn, by department. 


```python
df_churn = df2_final[df2_final['churn_prediction'] == 1]
churn_departments = [x for x in df_churn.columns if x.startswith('department_')]

churn_departments

for x in churn_departments: #rename department columns to omit "department_" prefix
  df_churn.rename( columns = {x : x.split("_")[1]} , inplace = True)

```
    Index(['satisfaction_level', 'last_evaluation', 'number_project',
           'average_monthly_hours', 'time_spend_company', 'work_accident',
           'quit_the_company', 'promotion_last_5years', 'employee_id',
           'salary_high', 'salary_low', 'salary_medium', 'it', 'randd',
           'accounting', 'hr', 'management', 'marketing', 'product', 'sales',
           'support', 'technical', 'churn_prediction', 'churn_probability'],
          dtype='object')




```python
df_churn

depart_names_truncated = [x.split("_")[1] for x in churn_departments]

depart_tally = dict()

for x in depart_names_truncated:
  if int(df_churn[x].sum() > 0 ): # only include departments with a nonzero count
    depart_tally[x] = int(df_churn[x].sum()) #sum the 1s from the departments column
```


```python
depart_tally
```




    {'it': 1, 'management': 2, 'sales': 2, 'support': 2, 'technical': 1}




```python
plt.barh(depart_tally.keys() , depart_tally.values() )
plt.title("At Risk Employees by Departments")
plt.ylabel("Department")
#plt.xlabel("Count")
plt.show()
```

    
![image-center](/assets/images/2025-10-28-Employee-Churn-Prediction-with-Classification-Models/output_49_0.png){: .align-center}


```python
df_depchurn = pd.DataFrame.from_dict(depart_tally, orient='index', columns=['count']).reset_index()
df_depchurn.rename(columns={'index': 'department'}, inplace=True)
df_depchurn
```



  <div id="df-5f7b4f2e-702b-43d1-b582-d268a8ec2e3e" class="colab-df-container">
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
      <th>department</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>it</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>management</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>sales</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>support</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>technical</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-5f7b4f2e-702b-43d1-b582-d268a8ec2e3e')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-5f7b4f2e-702b-43d1-b582-d268a8ec2e3e button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-5f7b4f2e-702b-43d1-b582-d268a8ec2e3e');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-161960f3-1581-4a9d-b57f-0741c8c01e91">
      <button class="colab-df-quickchart" onclick="quickchart('df-161960f3-1581-4a9d-b57f-0741c8c01e91')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-161960f3-1581-4a9d-b57f-0741c8c01e91 button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

  <div id="id_d368b466-3794-4887-a323-3395e333bdae">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('df_depchurn')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_d368b466-3794-4887-a323-3395e333bdae button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('df_depchurn');
      }
      })();
    </script>
  </div>

    </div>
  </div>



Determine the average values for each of the top 5 features for those at risk of churning


```python
means = dict()

for x in df_feats['feature'][0:5]: #iterate through the most important features
  means[x] = float(df_churn[x].mean())

means

df_churn_avgs = pd.DataFrame.from_dict(means, orient= 'index' , columns = ['average'] )
df_churn_avgs
```





  <div id="df-ad18ae7f-7cfd-4754-874a-31fca9a0af99" class="colab-df-container">
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
      <th>average</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>satisfaction_level</th>
      <td>0.275993</td>
    </tr>
    <tr>
      <th>number_project</th>
      <td>3.875000</td>
    </tr>
    <tr>
      <th>time_spend_company</th>
      <td>3.375000</td>
    </tr>
    <tr>
      <th>average_monthly_hours</th>
      <td>275.250000</td>
    </tr>
    <tr>
      <th>last_evaluation</th>
      <td>0.504686</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-ad18ae7f-7cfd-4754-874a-31fca9a0af99')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-ad18ae7f-7cfd-4754-874a-31fca9a0af99 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-ad18ae7f-7cfd-4754-874a-31fca9a0af99');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-d9516c18-5e45-4417-a3af-be63412f5af0">
      <button class="colab-df-quickchart" onclick="quickchart('df-d9516c18-5e45-4417-a3af-be63412f5af0')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-d9516c18-5e45-4417-a3af-be63412f5af0 button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

  <div id="id_95394020-a3ba-4fd9-b7bf-d12ce5f4673c">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('df_churn_avgs')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_95394020-a3ba-4fd9-b7bf-d12ce5f4673c button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('df_churn_avgs');
      }
      })();
    </script>
  </div>

    </div>
  </div>




## 7. Export to BigQuery


```python
#send feature importance dataframe to a BigQuery table
df_feats.to_gbq('employeedata.feature_table',
                project_id ,
                chunksize=None,
                if_exists='replace')

#send at-risk count by department dataframe to a BigQuery table
df_depchurn.to_gbq('employeedata.at_risk_dept_count',
                   project_id,
                   chunksize = None,
                   if_exists = 'replace')


#send average values for at-risk employees dataframe to a BigQuery table

df_churn_avgs.to_gbq('employeedata.at_risk_avgs',
                     project_id,
                     chunksize = None,
                     if_exists = 'replace')
```

```python
#send our final predictions to BigQuery
df2_final.to_gbq('employeedata.churn_predictions',
                project_id,
                chunksize=None,
                if_exists ='replace')
```


## 8. Conclusion 

Recommendations for addressing employee churn:
* Employee Recognition Program: Acknowledging and rewarding employees could boost job satisfaciton and motivaiton, addressing a key factor in reducing turnover. 
* Professinal Development Initiatives: Investing in training and development helps employees grow within the company, increasing their commitment and job satisfaction. 
* Reward Long Term Employees: Offering retention incentives encourages long-term commitment, addressing the importance of time spent with the company. 


We've summarized the results of our analysis with a simple Looker Studio dashboard. Below you will find a link to the dashboard and an image preview of it. 

[Link to Dashboard](https://lookerstudio.google.com/reporting/0a4c54b6-b6dd-4562-97e8-1c4d69ea1f38){: .btn .btn--primary}



![image-center](/assets/images/2025-10-28-Employee-Churn-Prediction-with-Classification-Models/Churn_Report.png){: .align-center}




