import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

reference_data = pd.read_csv('data/reference_data.csv')
new_data = pd.read_csv('data/new_data.csv')

df= pd.concat([reference_data, new_data], ignore_index=True)

X = df.drop('Outcome', axis=1)
y = df['Outcome']

numeric_features = X.columns
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

pipeline.fit(X, y)

with open('models/pipeline.pkl', 'wb') as f:
    pickle.dump(pipeline, f)
