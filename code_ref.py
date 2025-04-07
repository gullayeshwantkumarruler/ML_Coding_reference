from category_encoders import TargetEncoder
from sklearn.model_selection import train_test_split

# 1. Split the dataset
features = df.drop(columns=['tenure', 'attrition_flag'])
target_duration = df['tenure']
target_event = df['attrition_flag']

X_train, X_test, y_dur_train, y_dur_test, y_event_train, y_event_test = train_test_split(
    features, target_duration, target_event, test_size=0.2, random_state=42
)

# 2. Combine y_event_train with X_train for encoding
X_train['attrition_flag'] = y_event_train

# Identify categorical columns
cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

# 3. Reduce high-cardinality categories
def reduce_categories(df, col, top_n=10):
    top_categories = df[col].value_counts().nlargest(top_n).index
    df[col] = df[col].apply(lambda x: x if x in top_categories else 'Other')
    return df

for col in cat_cols:
    X_train = reduce_categories(X_train, col, top_n=10)
    X_test[col] = X_test[col].apply(lambda x: x if x in X_train[col].unique() else 'Other')

# 4. Fit encoder on training set
encoder = TargetEncoder(cols=cat_cols)
X_train_encoded = encoder.fit_transform(X_train[cat_cols], X_train['attrition_flag'])

# 5. Transform test set using fitted encoder
X_test_encoded = encoder.transform(X_test[cat_cols])

# 6. Merge back with numeric columns
X_train_final = pd.concat([X_train.drop(columns=cat_cols + ['attrition_flag']).reset_index(drop=True),
                           X_train_encoded.reset_index(drop=True)], axis=1)

X_test_final = pd.concat([X_test.drop(columns=cat_cols).reset_index(drop=True),
                          X_test_encoded.reset_index(drop=True)], axis=1)

# Now use X_train_final, y_dur_train, y_event_train for survival modeling
