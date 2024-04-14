import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load the datasets
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Function to convert 'Total Assets' values to numeric
def convert_assets_to_numeric(asset_value):
    if 'Crore' in asset_value:
        return float(asset_value.replace(' Crore+', '')) * 10**7
    elif 'Lac' in asset_value:
        return float(asset_value.replace(' Lac+', '')) * 10**5
    elif 'Thou' in asset_value:
        return float(asset_value.replace(' Thou+', '')) * 10**3
    else:
        return float(asset_value)

# Apply conversion function to 'Total Assets' column
train_data['Total Assets'] = train_data['Total Assets'].apply(convert_assets_to_numeric)
test_data['Total Assets'] = test_data['Total Assets'].apply(convert_assets_to_numeric)

# Display basic statistics of numerical features
print("Basic Statistics of Numerical Features:")
print(train_data.describe())

# Display the distribution of Education levels
plt.figure(figsize=(10, 6))
sns.countplot(x='Education', hue='Education', data=train_data, palette='Set2', legend=False)
plt.title('Distribution of Education Levels')
plt.xlabel('Education Level')
plt.ylabel('Count')
plt.show()

# Distribution of Criminal Cases by Party
plt.figure(figsize=(12, 6))
sns.countplot(x='Criminal Case', hue='Party', data=train_data, palette='tab10')
plt.title('Distribution of Criminal Cases by Party')
plt.xlabel('Number of Criminal Cases')
plt.ylabel('Count')
plt.legend(title='Party', bbox_to_anchor=(1, 1))
plt.show()

# Calculate median and standard deviation of criminal records
criminal_records_median = train_data['Criminal Case'].median()
criminal_records_std = train_data['Criminal Case'].std()

# Filter candidates with high criminal records
high_criminal_records = train_data[train_data['Criminal Case'] > (criminal_records_median + criminal_records_std)]

# Calculate percentage distribution of parties with candidates having high criminal records
criminal_records_by_party_high = high_criminal_records['Party'].value_counts(normalize=True) * 100

# Plot the pie chart
plt.figure(figsize=(10, 6))
plt.pie(criminal_records_by_party_high, labels=criminal_records_by_party_high.index, autopct='%1.1f%%', startangle=140)
plt.title('Percentage Distribution of Parties with Candidates Having High Criminal Records')
plt.show()

# Calculate median and standard deviation of total assets
total_assets_median = train_data['Total Assets'].median()
total_assets_std = train_data['Total Assets'].std()

# Filter candidates with high total assets
high_total_assets = train_data[train_data['Total Assets'] > (total_assets_median + total_assets_std)]

# Calculate percentage distribution of parties with candidates having high total assets
total_assets_by_party_high = high_total_assets['Party'].value_counts(normalize=True) * 100

# Plot the pie chart
plt.figure(figsize=(10, 6))
plt.pie(total_assets_by_party_high, labels=total_assets_by_party_high.index, autopct='%1.1f%%', startangle=140)
plt.title('Percentage Distribution of Parties with Candidates Having High Total Assets')
plt.show()

# Grouping data by state and education level
state_education_counts = train_data.groupby(['state', 'Education']).size().unstack(fill_value=0)

# Plotting the bar plot
plt.figure(figsize=(14, 8))
state_education_counts.plot(kind='bar', stacked=True)
plt.title('State vs. Education')
plt.xlabel('State')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Education Level')
plt.tight_layout()
plt.show()

# Grouping data by criminal cases and education level
criminal_education_counts = train_data.groupby('Education')['Criminal Case'].sum()

# Plotting the bar plot
plt.figure(figsize=(14, 8))
criminal_education_counts.plot(kind='bar')
plt.title('Number of Criminal Cases by Education Level')
plt.xlabel('Education Level')
plt.ylabel('Number of Criminal Cases')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# New feature engineering for training data
train_data['Prefix_Preference'] = train_data['Candidate'].apply(lambda x: 1 if 'Adv.' in x or 'Dr.' in x else 0)
train_data['Constituency_Preference'] = train_data['Constituency ∇'].apply(lambda x: -1 if x.startswith(('ST', 'SC')) else 0)

# New feature engineering for test data
test_data['Prefix_Preference'] = test_data['Candidate'].apply(lambda x: 1 if 'Adv.' in x or 'Dr.' in x else 0)
test_data['Constituency_Preference'] = test_data['Constituency ∇'].apply(lambda x: -1 if x.startswith(('ST', 'SC')) else 0)

# Define features and target
features = ['Constituency ∇', 'Party', 'Criminal Case', 'Total Assets', 'Liabilities', 'state', 'Prefix_Preference', 'Constituency_Preference']
target = 'Education'

# Convert categorical variables to numeric using LabelEncoder
le = LabelEncoder()
combined = pd.concat([train_data[features], test_data[features]])

for feature in features:
    le.fit(combined[feature])
    train_data[feature] = le.transform(train_data[feature])
    test_data[feature] = le.transform(test_data[feature])

# Remove non-numeric columns before calculating correlation
numeric_train_data = train_data.select_dtypes(include=['number'])

# Display correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(numeric_train_data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

X_train = train_data[features]
y_train = le.fit_transform(train_data[target])
X_test = test_data[features]

# Split the training data into training and validation sets
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Train the model
rf_classifier = RandomForestClassifier(n_estimators=1000, random_state=42) 
rf_classifier.fit(X_train_split, y_train_split)

# Make predictions on the test set
predictions = rf_classifier.predict(X_test)

# Convert the numeric predictions back to the original classes
predictions = le.inverse_transform(predictions)

# Write the predictions to a CSV file
submission_df = pd.DataFrame({'ID': test_data['ID'], 'Education': predictions})
submission_df.to_csv('my_submission_rf_improved_2.csv', index=False)
