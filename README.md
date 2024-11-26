# Install Dependencies

pip install numpy pandas seaborn matplotlib scikit-learn
# import dependencies

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC 
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score, auc
# EDA
# load data
df = pd.read_csv('../data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# view first few rows
df.head()
# check data shape
df.shape
# check data types
df.dtypes
# convert to floats and turn non-numeric values to NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors = 'coerce')
df['TotalCharges'].dtypes
# check duplicated values
df.duplicated().sum()
# check missing values
missing_info = df.isnull().sum()

# check unique values per column
unique_counts = df.nunique()

# check for non-informative columns (single unique value)
non_informative = [col for col in df.columns if df[col].nunique() <= 1]

# print results
print("Missing values per column:\n", missing_info[missing_info > 0])
print("\nUnique value counts per column:\n", unique_counts)
print("\nNon-informative columns:\n", non_informative)

# visualize missing values
missing_info.plot(kind ='bar', title = 'Missing Values per Column')
plt.show()
# Handling Missing Data
# drop NaN
df2 = df.dropna(axis = 0).copy()
# drop irrelevance data 
df2.drop('customerID', axis = 1, inplace = True)
# check isnull 
df2.isnull().sum()
df2.shape
df2.columns
# Visualize Data
# get value counts
churn_counts = df2['Churn'].value_counts()
total_customers = churn_counts.sum()

# create bar plot
plt.bar(churn_counts.index, churn_counts.values, color=['blue', 'orange'])
plt.xlabel('Churn')
plt.ylabel('Count')
plt.title('Count of Churn')
plt.xticks([0, 1], ['Not Churn', 'Churn'])
plt.show()

print(f"Churn: {churn_counts.iloc[1]} customers ({round((churn_counts.iloc[1] / total_customers) * 100, 2)}%)")
print(f"Not Churn: {churn_counts.iloc[0]} customers ({round((churn_counts.iloc[0] / total_customers) * 100, 2)}%)")
from here we can see it is an imbalanced data, "Not Churn" is almost 3 times more than "Churn"
churn_gender_counts = df2.groupby(['Churn', 'gender']).size().unstack()
churn_seniorcitizen_counts = df2.groupby(['Churn', 'SeniorCitizen']).size().unstack()
churn_partner_counts = df2.groupby(['Churn', 'Partner']).size().unstack()
churn_dependents_counts = df2.groupby(['Churn', 'Dependents']).size().unstack()

# Plotting the pie chart
plt.figure(figsize=(22, 6)) 

# establish the color
colors=['lightblue', 'lightcoral']

# Plie chart for Churn = 'Yes' (Gender)
plt.subplot(1, 4, 1)
plt.pie(churn_gender_counts.loc['Yes'], labels=churn_gender_counts.columns, autopct='%.1f%%', colors=colors)
plt.title('Gender')

# Pie chart for Churn = 'Yes' (Senior Citizen)
plt.subplot(1, 4, 2)
plt.pie(churn_seniorcitizen_counts.loc['Yes'], labels=['No','Yes'], autopct='%.1f%%',colors=colors)
plt.title('Senior Citizen Status')

# Pie chart for Churn = 'Yes' (Partner)
plt.subplot(1, 4, 3)
plt.pie(churn_partner_counts.loc['Yes'], labels=churn_partner_counts.columns, autopct='%1.1f%%',colors=colors)
plt.title('Partner')

# Pie chart for Churn = 'Yes' (Dependents)
plt.subplot(1, 4, 4)
plt.pie(churn_dependents_counts.loc['Yes'], labels=churn_dependents_counts.columns, autopct='%1.1f%%',colors=colors)
plt.title('Dependents')

# Display the plot
plt.tight_layout()  # Adjust layout for better spacing
plt.show()
- The data shows that nearly **50%** of each **gender** has switched their services.
- 75% of churned customers are not **Senior Citizens**, which requires further investigation.
- The pie charts for **Partner** and **Dependents** indicate that customers living alone tend to have a higher churn rate compared to others.
import matplotlib.pyplot as plt

# Grouping by Churn and various features
churn_contract_counts = df2.groupby(['Churn', 'Contract']).size().unstack()
churn_internet_service_counts = df2.groupby(['Churn', 'InternetService']).size().unstack()
churn_online_security_counts = df2.groupby(['Churn', 'OnlineSecurity']).size().unstack()
churn_online_backup_counts = df2.groupby(['Churn', 'OnlineBackup']).size().unstack()
churn_device_protection_counts = df2.groupby(['Churn', 'DeviceProtection']).size().unstack()
churn_tech_support_counts = df2.groupby(['Churn', 'TechSupport']).size().unstack()
churn_streaming_tv_counts = df2.groupby(['Churn', 'StreamingTV']).size().unstack()
churn_streaming_movies_counts = df2.groupby(['Churn', 'StreamingMovies']).size().unstack()

# Create the pie chart
plt.figure(figsize=(22, 8))

colors = ['lightblue', 'lightgreen', 'lightcoral']

plt.subplot(2, 4, 1)
plt.pie(
    churn_contract_counts.loc['Yes'],
    labels=churn_contract_counts.columns,
    autopct='%.1f%%',  # Display percentage
    colors=colors
)
plt.title('Contract Types')

plt.subplot(2, 4, 2)
plt.pie(
    churn_internet_service_counts.loc['Yes'],
    labels=churn_internet_service_counts.columns,
    autopct='%.1f%%',  # Display percentage
    colors=colors
)
plt.title('Internet Service')

plt.subplot(2, 4, 3)
plt.pie(
    churn_online_security_counts.loc['Yes'],
    labels=churn_online_security_counts.columns,
    autopct='%.1f%%',  # Display percentage
    colors=colors
)
plt.title('Online Security')

plt.subplot(2, 4, 4)
plt.pie(
    churn_online_backup_counts.loc['Yes'],
    labels=churn_online_backup_counts.columns,  
    autopct='%.1f%%',  # Display percentage
    colors=colors
)
plt.title('Online Backup')

plt.subplot(2, 4, 5)
plt.pie(
    churn_streaming_tv_counts.loc['Yes'],
    labels=churn_streaming_tv_counts.columns,
    autopct='%.1f%%',  # Display percentage
    colors=colors
)
plt.title('Streaming TV')

plt.subplot(2, 4, 6)
plt.pie(
    churn_device_protection_counts.loc['Yes'],
    labels=churn_device_protection_counts.columns,
    autopct='%.1f%%',  # Display percentage
    colors=colors
)
plt.title('Device Protection')

plt.subplot(2, 4, 7)
plt.pie(
    churn_tech_support_counts.loc['Yes'],
    labels=churn_tech_support_counts.columns,
    autopct='%.1f%%',  # Display percentage
    colors=colors
)
plt.title('Tech Support')

plt.subplot(2, 4, 8)
plt.pie(
    churn_streaming_movies_counts.loc['Yes'],
    labels=churn_streaming_movies_counts.columns,
    autopct='%.1f%%',  # Display percentage
    colors=colors
)
plt.title('Streaming Movies')

plt.tight_layout()  # Adjust layout for better spacing
plt.show()

- The pie chart above indicates that customers with a **month-to-month** contract have a significantly higher churn rate compared to other contract types.
- Additionally, customers with **fiber optic** service experience a churn rate that is 2.33 times higher than others.
- The other pie charts suggest that the primary reason for customer churn is not predominantly related to **no internet service**.
# See if customer tenure is different between customers who churned vs those who did not.
df2.boxplot(column='tenure', by='Churn')
plt.title('Box Plot of Tenure by Churn')
plt.suptitle('') # Remove the default title
plt.xlabel('Churn')
plt.ylabel('Tenure')
plt.grid(True)
plt.show()
Key Points from the Box Plot:
1. Churn = No (Customers who did not churn):
- Median tenure (green line): Customers who stayed tend to have a median tenure of around 40 months.
- Range: The tenure varies widely, with most customers having between approximately 15 and 60 months.
- Outliers: There are no significant outliers for this group.
- This suggests that longer-tenured customers are more likely to remain with the company.

2. Churn = Yes (Customers who churned):
- Median tenure (green line): Customers who churned have a lower median tenure of around 10 months.
- Range: The tenure for this group mostly falls between 0 and 30 months, showing that churned customers had shorter relationships with the company.
- Outliers: There are several outliers for customers who churned, representing a few individuals who had been with the company for a longer time but still churned (over 60 months).

3. Insights:
- Shorter tenure is strongly associated with customer churn, as most customers who left the company had shorter relationships (around 10 months).
- Longer tenure customers are more likely to stay, as seen with the "No Churn" group having a higher median and wider tenure distribution.
  
4. This visualization highlights that tenure is a significant predictor of churn, with customers who have been with the company longer being less likely to leave.
    # Compare monthly charges across different internet service types (DSL, Fiber optic, No internet).
    df2.boxplot(column='MonthlyCharges', by='InternetService')
    plt.title('Box Plot of Monthly Charges by Internet Service')
    plt.suptitle('') # Remove the default title
    plt.xlabel('Internet Service')
    plt.ylabel('Monthly Charges')
    plt.grid(True)
    plt.show()
Key Points from the Box Plot:
1. Internet Service = DSL:
- Represents customers who have DSL (Digital Subscriber Line) internet service.
- Median monthly charge: Around 60 USD.
- Range: Charges are spread out between approximately 20 USD and 110 USD, with some variability.
- This group may have different package options (like bundled services), leading to varying monthly charges.

2. Internet Service = Fiber optic:
- Represents customers who have Fiber optic internet service.
- Median monthly charge: Around 90 USD.
- Range: Charges are mostly between 70 USD and 110 USD, suggesting higher and more consistent monthly charges for this group.
- Fiber optic customers typically experience faster internet speeds, which may justify the higher and more uniform pricing.

3. Internet Service = No:
- This group likely represents customers without internet access, possibly due to opting for a lower-cost service or being eligible for promotional discounts.
- Median monthly charge: Very low, around 20 USD.
- Range: Monthly charges for this group are tightly clustered around 20 USD, with very little variation.
- Outliers: A few customers have slightly higher monthly charges, but most are concentrated around the same low price point.

4. Insights:
- Customers with Internet Service = **DSL** have a moderate range of monthly charges, which may reflect the variability in other services they subscribe to.
- Customers with Internet Service = **Fiber optic** pay higher monthly charges, with most paying around 90 USD to 100 USD.
- Customers with Internet Service = **No** have consistently low charges, likely reflecting a lack of internet access or minimal service offerings.
- The box plot clearly shows the relationship between internet service types and monthly charges, indicating that standard internet users (DSL and Fiber optic) pay significantly more than those without internet service.
# Examine how total charges vary by the type of contract (Month-to-month, One-year, Two-year).
df2.boxplot(column='TotalCharges', by='Contract')
plt.title('Box Plot of Total Charges by Contract')
plt.suptitle('') # Remove the default title
plt.xlabel('Contract')
plt.ylabel('Total Charges')
plt.grid(True)
plt.show()
Key Points:
1. Month-to-Month Contract:
- Median Total Charges: Relatively low, around 1,000 USD.
- Range: Most values fall between 0 USD and 2,000 USD.
- Outliers: There are many outliers above 6,000 USD, representing customers on month-to-month contracts who have accumulated higher total charges, potentially from long-term usage or higher monthly costs.
- This suggests that while many customers using this plan accumulate lower total charges, a few have stayed long enough to accumulate high totals, leading to the outliers.

2. One-Year Contract:
- Median Total Charges: Around 3,500 USD.
- Range: The distribution is more spread out, from approximately 0 USD to 6,000 USD.
- There are fewer outliers compared to the month-to-month group, indicating more consistency in total charges among customers on one-year contracts.
- Customers on this contract tend to accumulate higher total charges than those on month-to-month contracts, likely because they commit to longer periods.

3. Two-Year Contract:
- Median Total Charges: Slightly higher than the one-year contract, around 4,000 USD.
- Range: The total charges vary widely from 0 USD to almost 8,000 USD.
- No significant outliers, indicating a more stable distribution of total charges.
- Customers on two-year contracts accumulate the highest total charges, likely due to their long-term commitment to the service.

4. Insights:
- Month-to-month contracts show more variability, with many customers accumulating low total charges, but a few outliers indicate some customers have stayed long enough to accumulate large totals.
- One-year and two-year contracts show more stable distributions of total charges, with the two-year contract customers generally having higher total charges, as expected from longer commitments.

5. This box plot highlights that longer contracts (one or two years) generally lead to higher total charges compared to month-to-month plans.
# create scatter plot
sns.scatterplot(x='tenure', y='MonthlyCharges', hue='Churn', data=df2)
**Purpose:** To see if there is any relationship between how long a customer has stayed (tenure) and their monthly charges.

**Tenure** ranges from 0 to 72 months, and **MonthlyCharges** range from approximately 20 USD to 120 USD.

There **doesn't appear to be a clear linear or strong correlation** between tenure and MonthlyCharges, as the data points are widely scattered across the plot.

Customers with varying tenures (both short and long) seem to have a **broad range of MonthlyCharges**, suggesting that the duration of a customer's subscription does not necessarily increase or decrease their monthly charges.

**No Clear Trend:** Customers are paying anywhere from low to high monthly charges regardless of how long they have been with the company.
# create scatter plot
sns.scatterplot(x='tenure', y='TotalCharges', hue='Churn', data=df2)
**Purpose:** To explore if customers who churn tend to have higher total charges compared to those who do not.

**Upward Trend:** Customers who churn are often found on the higher side of TotalCharges. This suggests that those who leave the service may have initially spent more, indicating potential dissatisfaction with the value received relative to their spending.

**Spread of Charges:** There is notable variation in TotalCharges among customers at different tenure levels. This spread implies that customers with the same tenure can have significantly different charges, likely due to factors such as monthly charges, additional services, or varying discounts.

**High TotalCharges Among Churned Customers:** The presence of churned customers with high TotalCharges indicates that these individuals may have used more services before deciding to leave, pointing to a potential disconnect between perceived value and cost.

**Conclusion:** While there is a trend showing that customers who churn often have higher TotalCharges, the variations at each tenure level suggest that different factors contribute to this outcome. Understanding these dynamics is crucial for developing effective retention strategies.

# Understand the relationship between churn rates and contract types.
crosstab1 = pd.crosstab(df2['Churn'], df2['Contract'])
crosstab1
# Visualize how contract types relate to churn rates.
crosstab1.plot(kind='bar', stacked=True, color=['lightblue', 'lightgreen', 'lightcoral'])
plt.xlabel('Churn')
plt.ylabel('Count')
plt.title('Crosstab of Churn vs Contract')
plt.xticks(rotation=0)  # Optional: rotate x-axis labels if needed
plt.legend(title='Contract Type')
plt.show()
The stacked bar chart shows that most customers who churn do not sign long contracts. This suggests that shorter contracts are linked to higher churn rates
Summary:
- Pie Charts: Effective for showing the composition of categorical variables, such as the proportion of customers who churn versus those who do not. They can help visualize the overall distribution of churn rates within the customer base, but they may be less effective with too many categories.

- Scatter Plots: Best for exploring relationships between two numerical variables, like tenure and churn. They allow you to see patterns, trends, and potential correlations, such as whether longer-tenured customers are less likely to churn. This visualization can help identify clusters or outliers in the data.

- Boxplots: Best for comparing numerical variables (like tenure, MonthlyCharges, TotalCharges) across categories (like Churn, InternetService, Contract).

- Crosstabs: Useful for understanding relationships between categorical variables (e.g., Churn, Contract, PaymentMethod).

- Stacked Bar Charts: Ideal for visualizing the proportion of churn across different categories, such as Contract, InternetService, and PaymentMethod.
sns.pairplot(df2, hue='Churn')
**Purpose:**  
- A **pairplot** allows you to visualize pairwise relationships in a dataset, helping you identify potential **correlations** and **patterns** among different variables.

**Insights:**  
- **Correlations:** You can identify potential **linear or non-linear correlations** between variables. For example, a **positive correlation** would appear as an upward trend in the scatter plots.  
- **Clusters:** You may observe distinct **clusters** of data points, suggesting different groups within the dataset.  
- **Outliers:** Unusual points that stand out from the general trend can also be identified easily.  

**Conclusion:**  
- **Pairplots** provide a comprehensive view of your data, making them invaluable for **exploratory data analysis**. They help to uncover relationships and insights that might warrant further investigation, guiding your analytical approach.
# Encode Variables
for col in df2.select_dtypes(include=['object']).columns:
    print(f"{col}: {df2[col].unique()}")
for col in df2.select_dtypes(include=['object']).columns:
    le=LabelEncoder()
    le.fit(df2[col].unique())
    df2[col]=le.fit_transform(df2[col])
    print(f"{col}: {df2[col].unique()}")
# Save all encoders to a single file
joblib.dump(le, 'label_encoder.pkl')
df2.columns
df2.info()
df2.shape
df2.head()
correlation_matrix = df2.corr()
plt.figure(figsize=(20, 16))  
sns.heatmap(correlation_matrix, annot=True, fmt=".1f", cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.show()
In a correlation matrix, the strength of the correlation between two variables is typically assessed using correlation coefficients that range from -1 to 1:

- **0.0 to 0.3 (or 0.0 to -0.3)**: Weak correlation
- **0.3 to 0.6 (or -0.3 to -0.6)**: Moderate correlation
- **0.6 to 0.9 (or -0.6 to -0.9)**: Strong correlation
- **0.9 to 1.0 (or -0.9 to -1.0)**: Very strong correlation

**For exploratory data analysis**: Moderate correlations can provide valuable insights, especially when looking for trends or patterns.

# Data Preprocessing
y = df2['Churn']
x = df2.drop('Churn', axis=1)
x_train, x_test, y_train, y_test = train_test_split (x, y, test_size = 0.2, random_state=42)
scaler = StandardScaler()

# Fit and transform the training data
x_train_scaled = scaler.fit_transform(x_train)

# Transform the test data
x_test_scaled = scaler.transform(x_test)

# Convert the scaled data back to DataFrame for easier handling
x_train_scaled_df = pd.DataFrame(x_train_scaled, columns=x_train.columns)
x_test_scaled_df = pd.DataFrame(x_test_scaled, columns=x_test.columns)
# Save the scaler
joblib.dump(scaler, 'scaler.pkl')
# Data Reduction
# Apply PCA on the training set
pca = PCA(n_components=None)  
x_train_pca = pca.fit_transform(x_train_scaled_df)

# Transform the test set using the same PCA model
x_test_pca = pca.transform(x_test_scaled_df)

# Check explained variance to see how much variance is captured by PCA
explained_variance = pca.explained_variance_ratio_
print(f"Explained Variance (first few components): {explained_variance[:5]}")  
# Calculate cumulative explained variance
cumulative_explained_variance = explained_variance.cumsum()

# Plot the explained variance ratio
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='--')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Explained Variance')
plt.grid(True)
plt.axhline(y=0.95, color='r', linestyle='--')  # Line to indicate the threshold
plt.show()

# Set a threshold for the variance you want to retain (e.g., 95%)
threshold = 0.95

# To select n_components based on the threshold 
n_components = np.argmax(cumulative_explained_variance >= threshold) + 1

# Print the optimal number of components to retain the thresholded variance
print(f"Optimal number of components to retain {int(threshold * 100)}% variance: {n_components}")
# Apply PCA with the optimal number of components (16)
pca_optimal = PCA(n_components=16)  
x_train_pca_optimal = pca_optimal.fit_transform(x_train_scaled_df)

# Transform the test set using the same PCA model
x_test_pca_optimal = pca_optimal.transform(x_test_scaled_df)
# Save the trained PCA model
joblib.dump(pca_optimal, 'pca.pkl')
# Model Training
# Define models
models = {
    "KNN": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(random_state=42),
    "SVC": SVC(probability=True, random_state=42),
    "GaussianNB": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SGD": SGDClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(algorithm='SAMME', random_state=42)
}

# Train models and store results
results = {}

for name, model in models.items():
    try:
        model.fit(x_train_pca_optimal, y_train)
        y_pred = model.predict(x_test_pca_optimal)
        train_accuracy = model.score(x_train_pca_optimal, y_train)
        test_accuracy = accuracy_score(y_test, y_pred)
    
        # Store relevant metrics
        results[name] = {
            'Train Accuracy': train_accuracy,
            'Test Accuracy': test_accuracy,
            'Predictions': y_pred
        }
    except Exception as e:
        print(f"An error occurred with {name}: {e}")
# Model Evaluation
# Print the overall results without predictions
for model_name, metrics in results.items():
    print(f"Model: {model_name}")
    print(f"Train Accuracy: {metrics['Train Accuracy']:.4f}")
    print(f"Test Accuracy: {metrics['Test Accuracy']:.4f}")

    # Access predictions and print confusion matrix and classification report
    y_pred = metrics['Predictions']
    print(f"{model_name} Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    print(f"{model_name} Classification Report:\n{classification_report(y_test, y_pred)}\n")
# Find the model with the highest test accuracy
best_model_name = max(results, key=lambda x: results[x]['Test Accuracy'])
best_model_accuracy = results[best_model_name]['Test Accuracy']

print(f"The model with the highest test accuracy is: {best_model_name} with a Test Accuracy of {best_model_accuracy:.4f}")

# Create classifiers
clf1 = LogisticRegression()
clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
clf3 = GaussianNB()

# Create a Voting Classifier
clf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')

# Fit the model
clf.fit(x_train_pca_optimal, y_train)

# Make predictions
y_pred = clf.predict(x_test_pca_optimal)

# Calculate accuracies
train_accuracy_stacking = accuracy_score(y_train, clf.predict(x_train_pca_optimal))
test_accuracy_stacking = accuracy_score(y_test, y_pred)

# Print results
print(f"Model: Stacking")
print(f"Train Accuracy: {train_accuracy_stacking:.4f}")
print(f"Test Accuracy: {test_accuracy_stacking:.4f}")

# Confusion Matrix and Classification Report
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
# Hyper-parameter Tuning
param = {'C': [0.01,0.1,1,10,100],
        'gamma':[0.001,0.01,0.1,1,10,100]}

SVC_grid=GridSearchCV(SVC(kernel='rbf', probability=True),
                     param,
                     refit=True,
                     verbose=3)

SVC_grid.fit(x_train_pca_optimal,y_train)
# Retrieve the best parameters
best_params = SVC_grid.best_params_
print(f"Best parameters found: {best_params}")
### Best parameters found: {'C': 10, 'gamma': 0.01}
# Now perform cross-validation with the best parameters
clf = SVC(kernel='rbf', C=best_params['C'], gamma=best_params['gamma'])

# Perform cross-validation on the features
best_score = cross_val_score(clf, x_train_pca_optimal, y_train, cv=5, verbose=3)

# Output the cross-validation scores
print(f"Cross-validation scores: {best_score}")
print(f"Mean cross-validation score: {best_score.mean():.2f}")
# Retrieve the best model
hypertuning_model = SVC_grid.best_estimator_

# Evaluate the best model on the test set
y_pred = hypertuning_model.predict(x_test_pca_optimal)

# Evaluate the best model on the test set
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {test_accuracy:.4f}")
# Save the model and metadata
model_filename = 'hypertuning_model.pkl'
metadata_filename = 'model_metadata.pkl'

# Create a dictionary for metadata
metadata = {
    'best_params': best_params,
    'mean_cv_score': best_score.mean(),
    'test_accuracy': test_accuracy
}
# Dump the model 
joblib.dump(hypertuning_model, model_filename)
# Save metadata to a file
joblib.dump(metadata, metadata_filename)
# Use the hypertuned model to predict probabilities
y_probs = hypertuning_model.predict_proba(x_test_pca_optimal)[:, 1]  # Get probabilities for the positive class

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

# Calculate AUC
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', marker='o', label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()
An ROC curve area (AUC) of **0.79** is generally considered to be **good**. Here’s a breakdown of what that means:

### Understanding AUC:
1. **Definition of AUC**:
   - The Area Under the ROC Curve (AUC) measures a model’s ability to differentiate between positive and negative classes, with a range from **0 to 1**.

2. **Interpretation**:
   - **AUC = 0.5**: Indicates that the model performs no better than random chance.
   - **AUC < 0.7**: Generally signifies poor model performance.
   - **AUC between 0.7 and 0.8**: Reflects acceptable to good performance.
   - **AUC between 0.8 and 0.9**: Indicates very good performance.
   - **AUC > 0.9**: Represents excellent performance.

### Specifics for AUC = 0.79 in Customer Churn Prediction:
- **Acceptable Discriminative Ability**: An AUC of 0.79 suggests that the model has a reasonable ability to distinguish between customers who are likely to churn and those who are likely to stay. Specifically, this means that if you randomly select one customer who churned and one who didn’t, the model will correctly rank the churn-risk customer higher 79% of the time.

- **Practical Considerations for Churn**: While an AUC of 0.79 is generally acceptable, it may not be sufficient for high-stakes scenarios. In the context of customer churn prediction, this score indicates that the model can be useful, but there are some considerations to keep in mind regarding false positives and false negatives:
  - **False Positives**: These may lead to unnecessary retention efforts, potentially wasting resources on customers who are not at risk of leaving.
  - **False Negatives**: Missing customers who are likely to churn can result in lost revenue and a negative impact on customer satisfaction.

Given the AUC of 0.79, the model demonstrates potential but also indicates that there may be room for improvement. Exploring additional feature engineering, hyperparameter tuning, or alternative modeling approaches could enhance its performance.

### Conclusion:
An AUC of 0.79 is a solid indication that the model is performing reasonably well in distinguishing between customers at risk of churn and those who are not. While it reflects an acceptable predictive capability, ongoing efforts to refine the model could lead to improved performance and more effective customer retention strategies.
# Confusion Matrix
# Get predictions from the best model
best_model_name = "SVC"  
best_model = hypertuning_model  # Use the hypertuned model

best_model_predictions = best_model.predict(x_test_pca_optimal)

# Generate and print the confusion matrix
confusion_mat = confusion_matrix(y_test, best_model_predictions)
print(f"Confusion Matrix for {best_model_name}:\n{confusion_mat}")

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not Churned', 'Churned'], 
            yticklabels=['Not Churned', 'Churned'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix for {best_model_name}')
plt.show()
