import streamlit as st
  import pandas as pd
  import matplotlib.pyplot as plt
  import seaborn as sns
  from datetime import datetime

  # Thiết lập tiêu đề và cấu hình trang
  st.title("ABC Manufacturing Data Analysis Dashboard")
  st.markdown("Interactive dashboard for analyzing manufacturing data including sales, inventory, machine performance, and quality metrics.")

  # Tải dữ liệu với xử lý lỗi
  @st.cache_data
  def load_data():
      try:
          df = pd.read_csv("abc_manufacturing_data.csv")
          df['Date'] = pd.to_datetime(df['Date'])
          if df.empty:
              st.error("Data file is empty. Please check abc_manufacturing_data.csv.")
          return df
      except FileNotFoundError:
          st.error("Error: abc_manufacturing_data.csv not found. Please upload the file to the repository.")
          return pd.DataFrame()
      except Exception as e:
          st.error(f"Error loading data: {str(e)}")
          return pd.DataFrame()

  df = load_data()

  # Nếu dữ liệu rỗng, dừng ứng dụng
  if df.empty:
      st.stop()

  # Sidebar: Bộ lọc dữ liệu
  st.sidebar.header("Filter Options")
  product_filter = st.sidebar.multiselect("Select Product ID", options=df['Product_ID'].unique(), default=df['Product_ID'].unique())
  date_range = st.sidebar.date_input("Select Date Range", [df['Date'].min(), df['Date'].max()])
  date_range = [pd.to_datetime(d) for d in date_range]

  # Lọc dữ liệu theo bộ lọc
  filtered_df = df[(df['Product_ID'].isin(product_filter)) & 
                   (df['Date'] >= date_range[0]) & 
                   (df['Date'] <= date_range[1])]

  # Hiển thị bảng dữ liệu
  st.subheader("Filtered Data")
  st.dataframe(filtered_df)

  # Thiết lập phong cách biểu đồ
  plt.style.use('seaborn')
  sns.set_palette("husl")

  # Biểu đồ 1: Xu hướng Sales_Quantity
  st.subheader("Sales Quantity Trend by Product")
  fig1, ax1 = plt.subplots(figsize=(8, 4))  # Giảm kích thước để tối ưu tài nguyên
  for product in filtered_df['Product_ID'].unique():
      product_data = filtered_df[filtered_df['Product_ID'] == product]
      ax1.plot(product_data['Date'], product_data['Sales_Quantity'], marker='o', label=product)
  ax1.set_title('Sales Quantity Trend by Product')
  ax1.set_xlabel('Date')
  ax1.set_ylabel('Sales Quantity')
  ax1.legend()
  ax1.tick_params(axis='x', rotation=45)
  plt.tight_layout()
  st.pyplot(fig1)

  # Biểu đồ 2: Xu hướng Inventory_Level
  st.subheader("Inventory Level Trend by Product")
  fig2, ax2 = plt.subplots(figsize=(8, 4))
  for product in filtered_df['Product_ID'].unique():
      product_data = filtered_df[filtered_df['Product_ID'] == product]
      ax2.plot(product_data['Date'], product_data['Inventory_Level'], marker='o', label=product)
  ax2.set_title('Inventory Level Trend by Product')
  ax2.set_xlabel('Date')
  ax2.set_ylabel('Inventory Level')
  ax2.legend()
  ax2.tick_params(axis='x', rotation=45)
  plt.tight_layout()
  st.pyplot(fig2)

  # Biểu đồ 3: Phân bố Machine_Downtime_Hours
  st.subheader("Machine Downtime Hours by Machine ID")
  fig3, ax3 = plt.subplots(figsize=(8, 4))
  sns.boxplot(x='Machine_ID', y='Machine_Downtime_Hours', data=filtered_df, ax=ax3)
  ax3.set_title('Machine Downtime Hours by Machine ID')
  ax3.set_xlabel('Machine ID')
  ax3.set_ylabel('Downtime Hours')
  plt.tight_layout()
  st.pyplot(fig3)

  # Biểu đồ 4: Tỷ lệ không đạt chất lượng
  st.subheader("Quality Failure Rate by Product")
  quality_fail = filtered_df[filtered_df['Quality_Check_Pass'] == False].groupby('Product_ID').size() / filtered_df.groupby('Product_ID').size()
  quality_fail = quality_fail.reset_index(name='Failure_Rate')
  fig4, ax4 = plt.subplots(figsize=(8, 4))
  sns.barplot(x='Product_ID', y='Failure_Rate', data=quality_fail, ax=ax4)
  ax4.set_title('Quality Failure Rate by Product')
  ax4.set_xlabel('Product ID')
  ax4.set_ylabel('Failure Rate')
  plt.tight_layout()
  st.pyplot(fig4)

  # Biểu đồ 5: Sales_Quantity vs Quality_Issue_Count
  st.subheader("Sales Quantity vs Quality Issue Count")
  fig5, ax5 = plt.subplots(figsize=(8, 4))
  sns.scatterplot(x='Sales_Quantity', y='Quality_Issue_Count', hue='Product_ID', size='Product_ID', data=filtered_df, ax=ax5)
  ax5.set_title('Sales Quantity vs Quality Issue Count')
  ax5.set_xlabel('Sales Quantity')
  ax5.set_ylabel('Quality Issue Count')
  plt.tight_layout()
  st.pyplot(fig5)
