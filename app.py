import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Cấu hình trang
st.set_page_config(page_title="abc_manufacturing_data.csv", layout="wide")
st.title("📊 ABC Manufacturing Data Analysis Dashboard")

# ========== LOAD DATA ==========
@st.cache_data
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/minthangcris/data-analysis/refs/heads/main/abc_manufacturing_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date_Ordinal'] = df['Date'].map(pd.Timestamp.toordinal)
    return df

df_raw = load_data()

st.header("🔧 Data Preprocessing")

# 1. Show raw data
st.subheader("Raw Data")
st.dataframe(df_raw.head())

st.code("""
### @st.cache_data
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/minthangcris/data-analysis/refs/heads/main/abc_manufacturing_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date_Ordinal'] = df['Date'].map(pd.Timestamp.toordinal)
    return df

df_raw = load_data()

st.header("🔧 Data Preprocessing")

# 1. Show raw data
st.subheader("Raw Data")
st.dataframe(df_raw.head())
""", language="python")

# 2. Check & remove nulls
st.subheader("Step 1: Handling Null Values")
null_counts = df_raw.isnull().sum()
st.write("Missing values per column:")
st.write(null_counts)
st.code("""
### clean null value
if null_counts.sum() > 0:
    df_clean = df_raw.dropna()
    st.success(f"Removed {len(df_raw) - len(df_clean)} rows with null values.")
else:
    st.info("No null values found.")
    df_clean = df_raw.copy()
""", language="python")
if null_counts.sum() > 0:
    df_clean = df_raw.dropna()
    st.success(f"Removed {len(df_raw) - len(df_clean)} rows with null values.")
else:
    st.info("No null values found.")
    df_clean = df_raw.copy()

# 3. Remove duplicates
st.subheader("Step 2: Removing Duplicates")
st.code("""
### Removing Duplicates
before = len(df_clean)
df_clean = df_clean.drop_duplicates()
after = len(df_clean)
st.success(f"Removed {before - after} duplicate records.")
""", language="python")
before = len(df_clean)
df_clean = df_clean.drop_duplicates()
after = len(df_clean)
st.success(f"Removed {before - after} duplicate records.")

# 4. Data normalization/standardization
st.subheader("Step 3: Data Normalization (Standardization)")

numeric_cols = ['Sales_Quantity', 'Inventory_Level', 'Machine_Uptime_Hours',
                'Machine_Downtime_Hours', 'Quality_Issue_Count', 'Delivery_Time_Days']

scaler = StandardScaler()
df_clean[numeric_cols] = scaler.fit_transform(df_clean[numeric_cols])

st.write("Standardized Numeric Columns:")
st.dataframe(df_clean[numeric_cols].head())

st.code("""
numeric_cols = ['Sales_Quantity', 'Inventory_Level', 'Machine_Uptime_Hours',
                'Machine_Downtime_Hours', 'Quality_Issue_Count', 'Delivery_Time_Days']

scaler = StandardScaler()
df_clean[numeric_cols] = scaler.fit_transform(df_clean[numeric_cols])
""", language="python")

# Show data after preprocessing
st.subheader("✅ Preprocessed Data Sample")
st.dataframe(df_clean.head())

st.code("""
st.subheader("✅ Preprocessed Data Sample")
st.dataframe(df_clean.head())
""", language="python")

# ================== DATA ANALYSIS SECTION ==================
st.header("🔎 Data Analysis")

# 1. Distribution of Sales Quantity
st.subheader("📊 Distribution of Sales Quantity")
fig_a1, ax_a1 = plt.subplots(figsize=(8, 4))
sns.histplot(df_clean['Sales_Quantity'], kde=True, ax=ax_a1, bins=20)
ax_a1.set_title("Distribution of Sales Quantity")
st.pyplot(fig_a1)

st.code("""
st.subheader("📊 Distribution of Sales Quantity")
fig_a1, ax_a1 = plt.subplots(figsize=(8, 4))
sns.histplot(df_clean['Sales_Quantity'], kde=True, ax=ax_a1, bins=20)
ax_a1.set_title("Distribution of Sales Quantity")
st.pyplot(fig_a1)
""", language="python")

# 2. Correlation Heatmap
st.subheader("🔗 Correlation Between Numeric Variables")
fig_a2, ax_a2 = plt.subplots(figsize=(10, 6))
corr = df_clean[numeric_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax_a2)
ax_a2.set_title("Correlation Heatmap of Key Variables")
st.pyplot(fig_a2)

st.code("""
st.subheader("🔗 Correlation Between Numeric Variables")
fig_a2, ax_a2 = plt.subplots(figsize=(10, 6))
corr = df_clean[numeric_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax_a2)
ax_a2.set_title("Correlation Heatmap of Key Variables")
st.pyplot(fig_a2)

""", language="python")

# 3. Average Downtime vs Quality Issues
st.subheader("🛠️ Average Downtime vs Quality Issue Count per Machine")
downtime_quality = df_clean.groupby('Machine_ID')[['Machine_Downtime_Hours', 'Quality_Issue_Count']].mean().reset_index()
fig_a3, ax_a3 = plt.subplots(figsize=(8, 5))
sns.barplot(data=downtime_quality, x='Machine_ID', y='Quality_Issue_Count', color='orange', label='Quality Issues', ax=ax_a3)
sns.lineplot(data=downtime_quality, x='Machine_ID', y='Machine_Downtime_Hours', color='blue', marker='o', label='Downtime (hrs)', ax=ax_a3)
ax_a3.set_title("Avg. Downtime and Quality Issues per Machine")
ax_a3.legend()
st.pyplot(fig_a3)

st.code("""
st.subheader("🛠️ Average Downtime vs Quality Issue Count per Machine")
downtime_quality = df_clean.groupby('Machine_ID')[['Machine_Downtime_Hours', 'Quality_Issue_Count']].mean().reset_index()
fig_a3, ax_a3 = plt.subplots(figsize=(8, 5))
sns.barplot(data=downtime_quality, x='Machine_ID', y='Quality_Issue_Count', color='orange', label='Quality Issues', ax=ax_a3)
sns.lineplot(data=downtime_quality, x='Machine_ID', y='Machine_Downtime_Hours', color='blue', marker='o', label='Downtime (hrs)', ax=ax_a3)
ax_a3.set_title("Avg. Downtime and Quality Issues per Machine")
ax_a3.legend()
st.pyplot(fig_a3)
""", language="python")

# 4. Inventory Level by Supplier
st.subheader("🏷️ Inventory Level by Supplier")
fig_a4, ax_a4 = plt.subplots(figsize=(10, 5))
sns.boxplot(data=df_clean, x='Supplier_ID', y='Inventory_Level', ax=ax_a4)
ax_a4.set_title("Inventory Level Distribution by Supplier")
st.pyplot(fig_a4)

st.code("""
st.subheader("🏷️ Inventory Level by Supplier")
fig_a4, ax_a4 = plt.subplots(figsize=(10, 5))
sns.boxplot(data=df_clean, x='Supplier_ID', y='Inventory_Level', ax=ax_a4)
ax_a4.set_title("Inventory Level Distribution by Supplier")
st.pyplot(fig_a4)
""", language="python")

# 5. Quality Check Pass Rate
st.subheader("✅ Quality Check Pass Rate")
quality_summary = df_clean.groupby('Product_ID')['Quality_Check_Pass'].mean().reset_index()
quality_summary['Pass_Rate (%)'] = quality_summary['Quality_Check_Pass'] * 100
fig_a5, ax_a5 = plt.subplots(figsize=(8, 5))
sns.barplot(x='Product_ID', y='Pass_Rate (%)', data=quality_summary, ax=ax_a5)
ax_a5.set_ylim(0, 110)
ax_a5.set_title("Quality Check Pass Rate by Product")
st.pyplot(fig_a5)

st.code("""
st.subheader("✅ Quality Check Pass Rate")
quality_summary = df_clean.groupby('Product_ID')['Quality_Check_Pass'].mean().reset_index()
quality_summary['Pass_Rate (%)'] = quality_summary['Quality_Check_Pass'] * 100
fig_a5, ax_a5 = plt.subplots(figsize=(8, 5))
sns.barplot(x='Product_ID', y='Pass_Rate (%)', data=quality_summary, ax=ax_a5)
ax_a5.set_ylim(0, 110)
ax_a5.set_title("Quality Check Pass Rate by Product")
st.pyplot(fig_a5)
""", language="python")

# ================== SIDEBAR FILTER ==================
st.sidebar.header("Filter Options")

product_filter = st.sidebar.multiselect(
    "Select Product ID",
    options=df_clean['Product_ID'].unique(),
    default=df_clean['Product_ID'].unique()
)

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=[df_clean['Date'].min(), df_clean['Date'].max()]
)

if len(date_range) == 2:
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])

    if start_date > end_date:
        st.error("Start date cannot be after end date. Please adjust.")
        st.stop()

    filtered_df = df_clean[
        (df_clean['Product_ID'].isin(product_filter)) &
        (df_clean['Date'] >= start_date) &
        (df_clean['Date'] <= end_date)
    ]
else:
    st.warning("Please select a start and end date for filtering.")
    filtered_df = df_clean.copy()

# ================== VISUALIZATION ==================
st.header("📈 Data Visualizations")

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# 1. Sales Quantity Trend
st.subheader("1️⃣ Sales Quantity Trend by Product")
fig1, ax1 = plt.subplots(figsize=(10,5))
sns.lineplot(data=filtered_df, x='Date', y='Sales_Quantity', hue='Product_ID', marker='o', ax=ax1)
ax1.set_title('Sales Quantity Trend by Product')
ax1.tick_params(axis='x', rotation=45)
st.pyplot(fig1)

st.code("""
fig1, ax1 = plt.subplots(figsize=(10,5))
sns.lineplot(data=filtered_df, x='Date', y='Sales_Quantity', hue='Product_ID', marker='o', ax=ax1)
ax1.set_title('Sales Quantity Trend by Product')
ax1.tick_params(axis='x', rotation=45)
st.pyplot(fig1)
""", language="python")

# 2. Inventory Level Trend
st.subheader("2️⃣ Inventory Level Trend by Product")
fig2, ax2 = plt.subplots(figsize=(10,5))
sns.lineplot(data=filtered_df, x='Date', y='Inventory_Level', hue='Product_ID', marker='o', ax=ax2)
ax2.set_title('Inventory Level Trend by Product')
ax2.tick_params(axis='x', rotation=45)
st.pyplot(fig2)

st.code("""
fig2, ax2 = plt.subplots(figsize=(10,5))
sns.lineplot(data=filtered_df, x='Date', y='Inventory_Level', hue='Product_ID', marker='o', ax=ax2)
ax2.set_title('Inventory Level Trend by Product')
ax2.tick_params(axis='x', rotation=45)
st.pyplot(fig2)

""", language="python")

# 3. Machine Downtime by Machine
st.subheader("3️⃣ Machine Downtime Hours by Machine")
fig3, ax3 = plt.subplots(figsize=(10,5))
sns.boxplot(x='Machine_ID', y='Machine_Downtime_Hours', data=filtered_df, ax=ax3)
ax3.set_title('Machine Downtime by Machine ID')
st.pyplot(fig3)

st.code("""
fig3, ax3 = plt.subplots(figsize=(10,5))
sns.boxplot(x='Machine_ID', y='Machine_Downtime_Hours', data=filtered_df, ax=ax3)
ax3.set_title('Machine Downtime by Machine ID')
st.pyplot(fig3)
""", language="python")

# 4. Quality Failure Rate
st.subheader("4️⃣ Quality Failure Rate by Product")
quality_fail = filtered_df[filtered_df['Quality_Check_Pass'] == False].groupby('Product_ID').size() / filtered_df.groupby('Product_ID').size()
quality_fail = quality_fail.reset_index(name='Failure_Rate')
fig4, ax4 = plt.subplots(figsize=(8, 5))
sns.barplot(x='Product_ID', y='Failure_Rate', data=quality_fail, ax=ax4)
ax4.set_title('Quality Failure Rate by Product')
st.pyplot(fig4)

st.code("""
quality_fail = filtered_df[filtered_df['Quality_Check_Pass'] == False].groupby('Product_ID').size() / filtered_df.groupby('Product_ID').size()
quality_fail = quality_fail.reset_index(name='Failure_Rate')
fig4, ax4 = plt.subplots(figsize=(8, 5))
sns.barplot(x='Product_ID', y='Failure_Rate', data=quality_fail, ax=ax4)
ax4.set_title('Quality Failure Rate by Product')
st.pyplot(fig4)
""", language="python")

# 5. Sales Quantity vs Quality Issues
st.subheader("5️⃣ Sales Quantity vs Quality Issue Count")
fig5, ax5 = plt.subplots(figsize=(8, 5))
sns.scatterplot(x='Sales_Quantity', y='Quality_Issue_Count', hue='Product_ID', data=filtered_df, ax=ax5)
ax5.set_title('Sales Quantity vs Quality Issue Count')
st.pyplot(fig5)

st.code("""
fig5, ax5 = plt.subplots(figsize=(8, 5))
sns.scatterplot(x='Sales_Quantity', y='Quality_Issue_Count', hue='Product_ID', data=filtered_df, ax=ax5)
ax5.set_title('Sales Quantity vs Quality Issue Count')
st.pyplot(fig5)
""", language="python")

# ================== AI MODEL - Forecasting ==================
st.header("🤖 AI Model: Sales Quantity Forecasting")

selected_product = st.selectbox("Select Product for Forecasting", df_clean['Product_ID'].unique())

model_data = df_clean[df_clean['Product_ID'] == selected_product]
X = model_data[['Date_Ordinal']]
y = model_data['Sales_Quantity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.code("""
selected_product = st.selectbox("Select Product for Forecasting", df_clean['Product_ID'].unique())

model_data = df_clean[df_clean['Product_ID'] == selected_product]
X = model_data[['Date_Ordinal']]
y = model_data['Sales_Quantity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
""", language="python")

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

st.write(f"**Mô hình:** Linear Regression")
st.write(f"**Sản phẩm đã chọn:** {selected_product}")
st.write(f"**RMSE:** {rmse:.2f}")

st.code("""
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

st.write(f"**Mô hình:** Linear Regression")
st.write(f"**Sản phẩm đã chọn:** {selected_product}")
st.write(f"**RMSE:** {rmse:.2f}")
""", language="python")

fig6, ax6 = plt.subplots(figsize=(10,5))
ax6.scatter(X_test, y_test, color='blue', label='Actual')
ax6.scatter(X_test, y_pred, color='red', label='Predicted')
ax6.set_title(f'Actual vs Predicted Sales Quantity for {selected_product}')
ax6.set_xlabel('Date (Ordinal)')
ax6.set_ylabel('Sales Quantity')
ax6.legend()
st.pyplot(fig6)

st.code("""
fig6, ax6 = plt.subplots(figsize=(10,5))
ax6.scatter(X_test, y_test, color='blue', label='Actual')
ax6.scatter(X_test, y_pred, color='red', label='Predicted')
ax6.set_title(f'Actual vs Predicted Sales Quantity for {selected_product}')
ax6.set_xlabel('Date (Ordinal)')
ax6.set_ylabel('Sales Quantity')
ax6.legend()
st.pyplot(fig6)
""", language="python")

st.success("✅ Dashboard generated successfully!")
