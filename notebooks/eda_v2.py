
## import libraries
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
import sqlite3
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, PredefinedSplit
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from scipy.stats import zscore
from category_encoders import TargetEncoder
from sklearn.decomposition import PCA
from typing import Dict, List

#class for he db
database_name = 'db/house_prices.db'
# connect to the db
conn = sqlite3.connect(database_name)
train_df = pd.read_sql_query("SELECT * FROM train",conn)
test_df = pd.read_sql_query("SELECT * FROM test",conn)
# Close the connection
conn.close()


# Suppress all warnings
warnings.filterwarnings('ignore')
numerical_cols = train_df.select_dtypes(include=['number']).columns
numerical_dataset = train_df[numerical_cols]
numerical_dataset['SalePrice'] = train_df['SalePrice']
numerical_dataset.head()



cols_50perc = pd.DataFrame(train_df.isnull().sum().sort_values(ascending=False)).reset_index().rename(columns={'index': 'column_name', 0: 'missing_count'})
cols_50perc = cols_50perc[cols_50perc['missing_count'] > 0]
cols_50perc['missing_percentage'] = cols_50perc['missing_count'] / len(train_df)
cols_50perc = round(cols_50perc[cols_50perc['missing_percentage'] * 100 > 35] ,1)

#dropping columns 
drop_cols = cols_50perc['column_name'].values.tolist()
train_df = train_df.drop(columns=drop_cols, axis=1)
test_df = test_df.drop(columns=drop_cols, axis=1)
#check if any of hte columns are left
train_df.columns.isin(drop_cols)


# check for duplicates with in our dataset
duplicates_train = train_df.duplicated().sum()
duplicates_test = test_df.duplicated().sum()
print(f'There are {duplicates_train} duplicates in the train dataset')
print(f'There are {duplicates_test} duplicates in the test dataset')



skew_plot = numerical_cols.skew().sort_values(ascending=False)  ##.plot(kind='bar', figsize=(12, 6), color='#227B94')
zscore_cols = pd.DataFrame(numerical_cols).apply(zscore)


## Split data
X = train_df.drop(columns=['SalePrice','Id'])
y = train_df['SalePrice']

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

print(f"Training set size: {X_train.shape}")
print(f"Validation set size: {X_val.shape}")
print(f"Testing set size: {X_test.shape}")

#compare to test size
test_df = test_df.drop(columns=['Id'])
print(f"Prediction set size: {test_df.shape}")




## fix numerical data
numerical_cols = train_df.select_dtypes(include=['int64', 'float64']).columns
numerical_cols = [col for col in numerical_cols if col not in ['Id', 'SalePrice']]
# Define a functioon for skewness
def handle_skewness(df, numerical_cols):
    skew_plot = df[numerical_cols].skew().sort_values(ascending=False)
    skew_threshold = 2
    highly_skewed_cols = skew_plot[abs(skew_plot) > skew_threshold].index.values
    for col in highly_skewed_cols:
        df[col] = np.log1p(df[col])
    return df

# Assuming X_train, X_test, X_val are defined DataFrames
X_train_transformed = handle_skewness(X_train, numerical_cols)
X_test_transformed = handle_skewness(X_test, numerical_cols)
X_val_transformed = handle_skewness(X_val, numerical_cols)

#convert the test df
test_df_transformed = handle_skewness(test_df, numerical_cols)

print("X_train shape after skew transformationi",X_train_transformed.shape)
print("Precition set shpae after skew transformation",test_df_transformed.shape)



#handle missing data
numerical_cols = train_df.select_dtypes(include=['int64', 'float64']).columns
numerical_cols = [col for col in numerical_cols if col not in ['Id','SalePrice']]
categorical_cols = train_df.select_dtypes(include=['object']).columns

def handle_missing_data(df):
    for col in numerical_cols:
        df[col].fillna(df[col].mean(), inplace=True)
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)
    return df

#convert the values
X_train_imputed = handle_missing_data(X_train_transformed)
X_test_imputed = handle_missing_data(X_test_transformed)
X_val_imputed = handle_missing_data(X_val_transformed)

#convert for test set
test_df_imputed = handle_missing_data(test_df_transformed)

#printing shapes
print("X_train shape after skew imputation",X_train_imputed.shape)
print("X_val shape after skew imputation",X_val_imputed.shape)
print("Precition set shpae after skew imputation",test_df_imputed.shape)




## feature engineering
def handle_binary(df,col):
    """
    Encode binary columns as 0/1.
    Example: 'Street' (Grvl, Pave) → 0, 1.
    """
    le_ = LabelEncoder()
    le_.fit(df[col])

    df[col] = le_.transform(df[col])
    return df

binary_cols = ['Street', 'CentralAir', 'Utilities' ]

for col in binary_cols:
    X_train_fe = handle_binary(X_train_imputed, col)
    X_test_fe = handle_binary(X_test_imputed, col)
    X_val_fe = handle_binary(X_val_imputed, col)
    test_df_fe = handle_binary(test_df_imputed, col)

#printing shapes
print("X_train shape after skew imputation",X_train_fe.shape)
print("X_val shape after skew imputation",X_val_fe.shape)
print("Precition set shpae after skew imputation",test_df_fe.shape)



#Handling Low Cardinal Columns
class LowCardinalityEncoder:
    def __init__(self):
        self.encoders: Dict[str, OneHotEncoder] = {}
        self.feature_names: Dict[str, List[str]] = {}

    def fit(self, df: pd.DataFrame, low_card_cols: List[str]) -> None:
        """
        Fit OneHotEncoder on training data only.
        """
        for col in low_card_cols:
            self.encoders[col] = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            self.encoders[col].fit(df[[col]])
            self.feature_names[col] = self.encoders[col].get_feature_names_out([col])

    def transform(self, df: pd.DataFrame, low_card_cols: List[str]) -> pd.DataFrame:
        """
        Transform data using fitted encoders, ensuring consistent columns.
        """
        df = df.copy()
        for col in low_card_cols:
            encoded_features = pd.DataFrame(
                self.encoders[col].transform(df[[col]]),columns=self.feature_names[col],index=df.index)
            # Drop original column and add encoded features
            df = pd.concat([df.drop(col, axis=1), encoded_features], axis=1)
        return df

# Define columns
low_card_cols = [
    'HouseStyle', 'RoofMatl', 'Condition1', 'Condition2', 'Functional','SaleCondition', 
    'GarageType', 'SaleType', 'Heating', 'Foundation',
    'RoofStyle', 'BldgType', 'LotConfig', 'MSZoning', 'Electrical','LandContour', 'LotShape']
# Initialize encoder
encoder = LowCardinalityEncoder()
# Fit on training data only
encoder.fit(X_train_fe, low_card_cols)

# Transform all datasets
X_train_fo = encoder.transform(X_train_fe, low_card_cols)
X_val_fo = encoder.transform(X_val_fe, low_card_cols)
X_test_fo = encoder.transform(X_test_fe, low_card_cols)
test_df_fo = encoder.transform(test_df_fe, low_card_cols)

# Verify shapes
print(f"X_train shape: {X_train_fo.shape}")
print(f"X_val shape: {X_val_fo.shape}")
print(f"X_test shape: {X_test_fo.shape}")
print(f"Prediction set shape: {test_df_fo.shape}")

# Verify columns match
assert set(X_train_fo.columns) == set(X_val_fo.columns) == set(X_test_fo.columns) == set(test_df_fo.columns), "Column mismatch between datasets"

#High Cardinal Columns
def handle_high_cardinality(df_train, df_test, df_val, col, target_train):
    # Target encode with smoothing
    encoder_ = TargetEncoder(smoothing=10)
    encoder_.fit(df_train[col], target_train)

    # Transform all datasets using the same encoder
    df_train[col] = encoder_.transform(df_train[col])
    df_test[col] = encoder_.transform(df_test[col])
    df_val[col] = encoder_.transform(df_val[col])
    return df_train, df_test, df_val, encoder_

encoders = {}
high_card_cols = ['Neighborhood','Exterior2nd', 'Exterior1st']

for col in high_card_cols:
    X_train_fi, X_test_fi, X_val_fi, encoder_ = handle_high_cardinality(X_train_fo, X_test_fo, X_val_fo, col, y_train)
    encoders[col] = encoder_

# Transform the prediction set using the encoders saved
for col in high_card_cols:
    test_df_fo[col] = encoders[col].transform(test_df_fo[col])

# Verify shapes
print(f"X_train shape: {X_train_fi.shape}")
print(f"X_val shape: {X_val_fi.shape}")
print(f"X_test shape: {X_test_fi.shape}")
print(f"Prediction set shape: {test_df_fo.shape}")

##Ordinal Columsn
def handle_ordinal(df, col, mapping, default_value=2):
    # Map categories to numbers
    df[col] = df[col].map(mapping).fillna(default_value).astype(int)
    return df

## mapping for ordinal data columns
ordinal_cols = [
    'ExterQual', 'BsmtQual','BsmtCond','BsmtExposure','KitchenQual','HeatingQC','GarageQual','GarageCond','ExterCond',
    'LandSlope','PavedDrive','BsmtFinType1','BsmtFinType2', 'GarageFinish'
    ]
qual_mapping = {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0, 'missing': 2}
bsmt_exposure_mapping = {'Gd': 3, 'Av': 2, 'Mn': 1, 'No': 0, 'missing': 0}
garage_finish_mapping = {'Fin': 2, 'RFn': 1, 'Unf': 0, 'missing': 0}
paved_drive_mapping = {'Y': 2, 'P': 1, 'N': 0, 'missing': 0}

#mappings to column:
ordinal_mappings = {
    'ExterQual': qual_mapping,
    'BsmtQual': qual_mapping,
    'KitchenQual': qual_mapping,
    'HeatingQC': qual_mapping,
    'GarageQual': qual_mapping,
    'GarageCond': qual_mapping,
    'ExterCond': qual_mapping,
    'BsmtCond': qual_mapping,
    'BsmtExposure': bsmt_exposure_mapping,
    'LandSlope': {'Gtl': 0, 'Mod': 1, 'Sev': 2, 'missing': 0},
    'PavedDrive': paved_drive_mapping,
    'BsmtFinType1': {'GLQ': 5, 'ALQ': 4, 'BLQ': 3, 'Rec': 2, 'LwQ': 1, 'Unf': 0, 'missing': 0},
    'BsmtFinType2': {'GLQ': 5, 'ALQ': 4, 'BLQ': 3, 'Rec': 2, 'LwQ': 1, 'Unf': 0, 'missing': 0},
    'GarageFinish': garage_finish_mapping
}

# Ordinal columns
for col, mapping in ordinal_mappings.items():
    X_train_ord = handle_ordinal(X_train_fi, col, mapping)
    X_test_ord = handle_ordinal(X_test_fi, col, mapping)
    X_val_ord = handle_ordinal(X_val_fi, col, mapping)
    test_df_ord = handle_ordinal(test_df_fo, col, mapping)

# Verify shapes
print(f"X_train shape: {X_train_ord.shape}")
print(f"X_val shape: {X_test_ord.shape}")
print(f"X_test shape: {X_val_ord.shape}")
print(f"Prediction set shape: {test_df_ord.shape}")


### Modelling

#Finding best parameters using GridSearch CV
#But since we're using GridSearch cv it will automatically split the train data to a split train - val Thus we need to jon them

#combine X_train and X_val
X_train_val = np.vstack([X_train_ord, X_val_ord])
y_train_val = np.concatenate([y_train, y_val])

#a split index: -1 for training, 0 for validation
split_index = [-1] * len(X_train_ord) + [0] * len(X_val_ord)
ps = PredefinedSplit(test_fold=split_index)

xgb_model = XGBRegressor()
#define the parameters
parameters = {
    "learning_rate": (0.01, 0.03, 0.05, 0.10), #step size at which the model updates its predictions with each boosting round
    "subsample": [0.7, 0.8, 0.9, 1.0], #fraction of training dat used per tree
    "n_estimators": [100, 300, 500, 1000] , #numerb of boosting rounds or tree the model builds
    "max_depth": [ 3, 5, 7, 9], #depth of each tree
    "min_child_weight": [ 1, 3, 5, 7], #helps decide if a split is necessary
    "gamma":[ 0.0, 0.1, 0.2, 0.3], #specifies the minimum loss function
    "colsample_bytree":[ 0.3, 0.5, 0.7], #fraction of featrues randomlly sample to build each tree
    "reg_alpha": [0, 0.1, 0.5 ,1],   #lasso regularization
    "reg_lambda": [0.85, 1, 1.5, 2] #ridge regularization - L2
        }

##fiting parameters for early stopping
fit_params = {
    "eval_set": [(X_val, y_val)],
    "early_stopping_rounds": 10,
    "verbose": True
        }

#create a class
gcv = GridSearchCV(
    estimator = xgb_model,
    param_grid = parameters,
    scoring = 'neg_root_mean_squared_error',
    cv = ps,
    verbose = 0
        )

gcv.fit(X_train_val, y_train_val)

#save best model
best_model = gcv.best_estimator_
best_model.fit(X_train_val, y_train_val)
#evaluate on test set
test_rmse = -best_model.score(X_test_ord, y_test)

print("Best parameters:", gcv.best_params_)
print(f"Test RMSE: {test_rmse}")
print("Best cross-validation score (negative RMSE):", gcv.best_score_)


#Finalize Model
#get best parameters
best_params = gcv.best_params_
#create a new model with early stoppng and the best parmaeters from beforre
final_model = XGBRegressor(
    **best_params,
    early_stopping_rounds=10  #throwing an error before cna not be added to the gcv
    )
#training with validation montiroing
history = final_model.fit(
    X_train_ord, y_train, eval_set=[(X_train_ord, y_train), (X_val_ord, y_val)],verbose=False
    )
results = final_model.evals_result()


#actual vs predicted plot
y_test_pred = final_model.predict(X_test_ord)



import pickle
#save the model that has best parameters
with open('model/xgb_model.pkl', 'wb') as f:
    pickle.dump(final_model, f)



#prediction 
sumbmisions_df = best_model.predict(test_df_ord)
#class for he db
database_name = 'house_prices.db'
# connect to the db
conn = sqlite3.connect(database_name)
test_df_subm = pd.read_sql_query("SELECT * FROM test",conn)
# Close the connection
conn.close()


test_df_subm['SalePrice'] = pd.DataFrame(sumbmisions_df)
final_submission = test_df_subm[['Id', 'SalePrice']]

#final_submission.head()
final_submission.to_csv('predictions/final_submission.csv', index=False)
final_submission.head()