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
import plotly.express as px

# Cấu hình trang
st.set_page_config(page_title="ABC Manufacturing Data Analysis", layout="wide")
st.title("📊 ABC Manufacturing Data Analysis Dashboard")

# ========== LOAD DATA ==========
@st.cache_data
def load_data():
    # Sử dụng dữ liệu từ URL hoặc tệp cục bộ
    df = pd.read_csv("abc_manufacturing_data.csv")  # Thay bằng URL nếu cần
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date_Ordinal'] = df['Date'].map(pd.Timestamp.toordinal)
    return df

df_raw = load_data()

# ========== DATA PREPROCESSING ==========
st.header("🔧 Data Preprocessing")

# 1. Show raw data
st.subheader("Raw Data")
st.write("Số lượng bản ghi:", len(df_raw))
st.dataframe(df_raw.head())
st.code("""
@st.cache_data
def load_data():
    df = pd.read_csv("abc_manufacturing_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date_Ordinal'] = df['Date'].map(pd.Timestamp.toordinal)
    return df
df_raw = load_data()
""", language="python")

# 2. Check & remove nulls
st.subheader("Step 1: Handling Null Values")
null_counts = df_raw.isnull().sum()
st.write("Missing values per column:")
st.write(null_counts)
if null_counts.sum() > 0:
    df_clean = df_raw.dropna()
    st.success(f"Removed {len(df_raw) - len(df_clean)} rows with null values.")
else:
    st.info("No null values found.")
    df_clean = df_raw.copy()
st.code("""
null_counts = df_raw.isnull().sum()
if null_counts.sum() > 0:
    df_clean = df_raw.dropna()
    st.success(f"Removed {len(df_raw) - len(df_clean)} rows with null values.")
else:
    st.info("No null values found.")
    df_clean = df_raw.copy()
""", language="python")

# 3. Remove duplicates
st.subheader("Step 2: Removing Duplicates")
before = len(df_clean)
df_clean = df_clean.drop_duplicates()
after = len(df_clean)
st.success(f"Removed {before - after} duplicate records.")
st.code("""
before = len(df_clean)
df_clean = df_clean.drop_duplicates()
after = len(df_clean)
st.success(f"Removed {before - after} duplicate records.")
""", language="python")

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

# Show preprocessed data
st.subheader("✅ Preprocessed Data Sample")
st.dataframe(df_clean.head())
st.code("""
st.subheader("✅ Preprocessed Data Sample")
st.dataframe(df_clean.head())
""", language="python")

# ========== SIDEBAR FILTER ==========
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

st.code("""
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
""", language="python")

# ========== VISUALIZATION ==========
st.header("📈 Data Visualization")

# 1. Sales Quantity Trend
st.subheader("1️⃣ Sales Quantity Trend by Product")
fig1 = px.line(filtered_df, x='Date', y='Sales_Quantity', color='Product_ID',
               title='Sales Quantity Trend by Product', markers=True)
st.plotly_chart(fig1)
st.code("""
fig1 = px.line(filtered_df, x='Date', y='Sales_Quantity', color='Product_ID',
               title='Sales Quantity Trend by Product', markers=True)
st.plot Surprisingly, the Plotly version is more interactive than Matplotlib!
""", language="python")

# 2. Inventory Level Trend
st.subheader("2️⃣ Inventory Level Trend by Product")
fig2 = px.line(filtered_df, x='Date', y='Inventory_Level', color='Product_ID',
               title='Inventory Level Trend by Product', markers=True)
st.plotly_chart(fig2)
st.code("""
fig2 = px.line(filtered_df, x='Date', y='Inventory_Level', color='Product_ID',
               title='Inventory Level Trend by Product', markers=True)
st.plotly_chart(fig2)
""", language="python")

# 3. Machine Downtime by Machine
st.subheader("3️⃣ Machine Downtime Hours by Machine")
fig3 = px.box(filtered_df, x='Machine_ID', y='Machine_Downtime_Hours',
              title='Machine Downtime by Machine ID', color='Machine_ID')
st.plotly_chart(fig3)
st.code("""
fig3 = px.box(filtered_df, x='Machine_ID', y='Machine_Downtime_Hours',
              title='Machine Downtime by Machine ID', color='Machine_ID')
st.plotly_chart(fig3)
""", language="python")

# 4. Quality Failure Rate
st.subheader("4️⃣ Quality Failure Rate by Product")
quality_fail = filtered_df[filtered_df['Quality_Check_Pass'] == False].groupby('Product_ID').size() / filtered_df.groupby('Product_ID').size()
quality_fail = quality_fail.reset_index(name='Failure_Rate')
fig4 = px.bar(quality_fail, x='Product_ID', y='Failure_Rate', color='Product_ID',
              title='Quality Failure Rate by Product')
st.plotly_chart(fig4)
st.code("""
quality_fail = filtered_df[filtered_df['Quality_Check_Pass'] == False].groupby('Product_ID').size() / filtered_df.groupby('Product_ID').size()
quality_fail = quality_fail.reset_index(name='Failure_Rate')
fig4 = px.bar(quality_fail, x='Product_ID', y='Failure_Rate', color='Product_ID',
              title='Quality Failure Rate by Product')
st.plotly_chart(fig4)
""", language="python")

# 5. Sales Quantity vs Quality Issues
st.subheader("5️⃣ Sales Quantity vs Quality Issue Count")
fig5 = px.scatter(filtered_df, x='Sales_Quantity', y='Quality_Issue_Count', color='Product_ID',
                  title='Sales Quantity vs Quality Issue Count')
st.plotly_chart(fig5)
st.code("""
fig5 = px.scatter(filtered_df, x='Sales_Quantity', y='Quality_Issue_Count', color='Product_ID',
                  title='Sales Quantity vs Quality Issue Count')
st.plotly_chart(fig5)
""", language="python")

# 6. Additional Visualizations from Previous Code
st.subheader("6️⃣ Additional Insights")
# Sales by Product
sales_by_product = filtered_df.groupby('Product_ID')['Sales_Quantity'].sum().reset_index()
fig6 = px.bar(sales_by_product, x='Product_ID', y='Sales_Quantity',
              title="Total Sales by Product", color='Product_ID',
              color_discrete_sequence=px.colors.qualitative.Set2)
st.plotly_chart(fig6)

# Quality Issue Ratio
filtered_df['Quality_Issue_Ratio'] = filtered_df['Quality_Issue_Count'] / filtered_df['Sales_Quantity']
quality_by_product = filtered_df.groupby('Product_ID')['Quality_Issue_Ratio'].mean().reset_index()
fig7 = px.bar(quality_by_product, x='Product_ID', y='Quality_Issue_Ratio',
              title="Average Quality Issue Ratio by Product", color='Product_ID',
              color_discrete_sequence=px.colors.qualitative.Set1)
st.plotly_chart(fig7)

# Delivery Time by Supplier
delivery_by_supplier = filtered_df.groupby('Supplier_ID')['Delivery_Time_Days'].mean().reset_index()
fig8 = px.bar(delivery_by_supplier, x='Supplier_ID', y='Delivery_Time_Days',
              title="Average Delivery Time by Supplier", color='Supplier_ID',
              color_discrete_sequence=px.colors.qualitative.Pastel)
st.plotly_chart(fig8)

# ========== AI MODEL - Forecasting ==========
st.header("🤖 AI Model: Sales Quantity Forecasting")
selected_product = st.selectbox("Select Product for Forecasting", df_clean['Product_ID'].unique())
model_data = df_clean[df_clean['Product_ID'] == selected_product]
X = model_data[['Date_Ordinal']]
y = model_data['Sales_Quantity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

st.write(f"**Model:** Linear Regression")
st.write(f"**Selected Product:** {selected_product}")
st.write(f"**RMSE:** {rmse:.2f}")
fig9 = px.scatter(x=X_test['Date_Ordinal'], y=y_test, title=f'Actual vs Predicted Sales Quantity for {selected_product}',
                  labels={'x': 'Date (Ordinal)', 'y': 'Sales Quantity'})
fig9.add_scatter(x=X_test['Date_Ordinal'], y=y_pred, mode='markers', name='Predicted', marker=dict(color='red'))
st.plotly_chart(fig9)

st.code("""
selected_product = st.selectbox("Select Product for Forecasting", df_clean['Product_ID'].unique())
model_data = df_clean[df_clean['Product_ID'] == selected_product]
X = model_data[['Date_Ordinal']]
y = model_data['Sales_Quantity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
fig9 = px.scatter(x=X_test['Date_Ordinal'], y=y_test, title=f'Actual vs Predicted Sales Quantity for {selected_product}',
                  labels={'x': 'Date (Ordinal)', 'y': 'Sales Quantity'})
fig9.add_scatter(x=X_test['Date_Ordinal'], y=y_pred, mode='markers', name='Predicted', marker=dict(color='red'))
st.plotly_chart(fig9)
""", language="python")

st.success("✅ Dashboard generated successfully!")
