import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import logistic
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix




# Load data
df = pd.read_csv("train.csv")

# Inspect the data
# print(df.info())
# print(df.describe())

# Prints histogram remove hash from both print and show plot
# print(df['Sales'].hist())
#plt.show()

# Convert order date and ship date to datetime for pandas (format is d-m-y so day first)
df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
df['Ship Date'] = pd.to_datetime(df['Ship Date'], dayfirst=True)

# Create month and year columns
df['Order Month'] = df['Order Date'].dt.month
df['Order Year'] = df['Order Date'].dt.year
# Create month-year for charts (This lets you plot sales or profit over time with clean labels like "2022-07")
df['Month_Year'] = df['Order Date'].dt.to_period('M').astype(str)

# Subtract Order Date from Ship Date to get the time delta. Then extract the number of days
df['Days to Ship'] = (df['Ship Date'] - df['Order Date']).dt.days

# Show new columns created
# print(df[['Order Date', 'Ship Date', 'Days to Ship', 'Month_Year']].head())

# Categorize sales by category
category_sales = df.groupby('Category')['Sales'].sum().reset_index()
# print(category_sales)
# Plot for categorized sales remove triple quotes to run
"""
sns.barplot(data=category_sales, x='Sales', y='Category')
plt.title('Total Sales by Category')
plt.ylabel('Total Sales')
plt.xlabel('Category')
plt.tight_layout()
plt.savefig('total_sales_by_category')
"""

sub_category_sales = df.groupby('Sub-Category')['Sales'].sum().reset_index()
# print(sub_category_sales)

# Count orders per customer
customer_freq = df['Customer Name'].value_counts().reset_index()
customer_freq.columns = ['Customer Name', 'Order Count']

# Merge into original DF
df = df.merge(customer_freq, on='Customer Name', how='left')

# creates boolean and highlights frequent buyers over 9 orders
df['Frequent Buyer'] = df['Order Count'] > 9
# print(df[['Customer Name', 'Order Count', 'Frequent Buyer']])

# Prediction Target
# Predict if an order will exceed a revenue threshold
# Define a threshold for high sales - $250
df['High Value'] = df['Sales'] > 250
# Check Distribution False 0.78 True 0.22
# print(df['High Value'].value_counts(normalize=True))

# Select Features and Target
features = ['Order Month', 'Days to Ship', 'Order Count', 'Category', ]
# print(df.columns)

# Ecnode so model can read text
"""
- df[features]: selects just the features you want (like 'Order Month', 'Days to Ship', 'Order Count', 'Category')
- pd.get_dummies(...): creates binary columns for each category
- drop_first=True: drops the first category to prevent duplicate info (avoids multicollinearity)
"""
df_encoded = pd.get_dummies(df[features], drop_first=True)

# Feature Matrix
x = df_encoded
y = df['High Value']

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Train Test Split (80 percent for train 20 for testing performance
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

model = LogisticRegression(class_weight='balanced')
model.fit(x_train, y_train)


y_pred = model.predict(x_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

