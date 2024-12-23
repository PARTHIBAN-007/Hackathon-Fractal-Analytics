import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score ,median_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

class DataLoader:
    def data_loading(self):
        train_df = pd.read_csv("Dataset/train.csv")
        test_df = pd.read_csv("Dataset/test.csv")
        return train_df, test_df

    def shape(self, train_df, test_df):
        print(f"Shape of the Train Data: {train_df.shape}")
        print(f"Shape of the Test Data: {test_df.shape}")

    def null_count(self, train_df, test_df):    
        print("No of Null Values in Train Data")
        print(train_df.isnull().sum())
        print("\n")
        print("No of Null Values in Test Data")
        print(test_df.isnull().sum())

    def duplicates(self, train_df, test_df):
        print(f"Duplicates in Train Data: {train_df.duplicated().sum()}")   
        print(f"Duplicates in Test Data: {test_df.duplicated().sum()}")    

    def data_cleaning(self, col, df):
        df[col] = df[col].str.replace('<h1>', '')
        df[col] = df[col].str.replace('</h1>', '')
        df[col] = df[col].str.replace('<h2>', '')
        df[col] = df[col].str.replace('</h2>', '')
        df[col] = df[col].str.findall(r"[\d.]+")
        df[col] = df[col].apply(lambda x: float(x[0]))

        print(f"Null values in {col} Level: {df[col].isnull().sum()}")
        return df[col]

    def data_split(self, x, y):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) 
        return x_train, x_test, y_train, y_test


# Load dataset
train_df = pd.read_csv(r"Dataset/train.csv")
test_df = pd.read_csv(r"Dataset/test.csv")

# Initialize DataLoader class
Dataloader = DataLoader()

# Perform Data Inspection
Dataloader.shape(train_df, test_df)
Dataloader.null_count(train_df, test_df)
Dataloader.duplicates(train_df, test_df)

# Clean specific columns
cols = ['ph', 'Hardness', 'Solids', 'Turbidity']
for col in cols:
    train_df[col] = Dataloader.data_cleaning(col, train_df)
    test_df[col] = Dataloader.data_cleaning(col, test_df)

# Prepare data for model
train_dff = train_df[['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 
                      'Organic_carbon', 'Trihalomethanes', 'Turbidity', 'Potability']]

train_dff.info()

# Split features and target
X = train_dff.drop(columns='Potability')
y = train_dff['Potability']

# Split into training and test sets
X_train, X_test, y_train, y_test = Dataloader.data_split(X, y)


# Evaluation metric function
def eval_metric(y_true, y_pred, model):
    print(model)
    print(f"Mean Absolute Error: {mean_absolute_error(y_true, y_pred)}")
    print(f"Mean Squared Error: {mean_squared_error(y_true, y_pred)}")
    print(f"Root Mean Squared Error: {root_mean_squared_error(y_true, y_pred)}")
    print(f"R2 Score: {r2_score(y_true, y_pred)}")
    mae = median_absolute_error(y_true, y_pred)

    # Calculate the custom score
    score = max(0, 100 * (1 - mae))

    print(f"Mean Absolute Error: {mae}")
    print(f"Custom Score: {score}")
    return score


# Model building function for Linear Regression, Decision Tree, Random Forest, XGBoost
def build_models():
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor(),
        "XGBoost": xgb.XGBRegressor(objective="reg:squarederror")
    }

    best_model = None
    best_score = -1  # Start with a low score to ensure the first model updates it

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = eval_metric(y_test, y_pred, model_name)

        # Keep track of the best model based on the custom score
        if score > best_score:
            best_score = score
            best_model = model

    return best_model


# Build and train the models, saving the best one
best_model = build_models()

# Save the best model as pickle
artifacts = {'model': best_model}
with open("./Model/best_model.pkl", 'wb') as file:
    pickle.dump(artifacts, file)

print("Best model pickle saved to model folder")

# Prepare the test data for prediction
test_dff = test_df[['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 
                    'Organic_carbon', 'Trihalomethanes', 'Turbidity']]

# Load the best model from pickle file
with open("./Model/best_model.pkl", 'rb') as file:
    artifacts = pickle.load(file)

best_model = artifacts['model']

# Make predictions using the best model
lr_predictions = best_model.predict(test_dff)

# Create DataFrame with predictions
predictions = pd.DataFrame({'Index': test_df['Index'], 'Potability': lr_predictions})

# Set Index for predictions
predictions = predictions.set_index('Index')

# Save predictions to CSV
predictions.to_csv("Submission.csv")

print("Predictions saved to Submission.csv")
