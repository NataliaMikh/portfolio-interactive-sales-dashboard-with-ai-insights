import streamlit as st
import pandas as pd
import plotly.express as px
import openai
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import ssl

# Disabling SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

# Setting the page configuration
st.set_page_config(layout="wide")

# Custom CSS for the application
st.markdown("""
    <style>
    .fixed-title {
        position: fixed;
        top: 45px;
        left: 0;
        width: 100%;
        background-color: #346E9F;
        padding: 15px 0px 3px 0px;
        font-size: 24px;
        color: #FFFFFF !important;
        z-index: 9999;
        text-align: center;
    }
    .content {
        margin-top: 70px;
    }
    </style>
    """, unsafe_allow_html=True)

# Fixed title for the dashboard
st.markdown('<div class="fixed-title">Sales Analytics Dashboard</div>', unsafe_allow_html=True)

# Function to load default data
def load_default_data():
    paths = {
        'BTN': 'https://raw.githubusercontent.com/NGravereaux/interactive-sales-dashboard-with-ai-insights/main/default_Site-BTN_Sales_History.CSV',
        'DOR': 'https://raw.githubusercontent.com/NGravereaux/interactive-sales-dashboard-with-ai-insights/main/default_Site-DOR_Sales_History.CSV',
        'TOR': 'https://raw.githubusercontent.com/NGravereaux/interactive-sales-dashboard-with-ai-insights/main/default_Site-TOR_Sales_History.CSV'
    }
    dataframes = {site: pd.read_csv(path, encoding='ISO-8859-1', low_memory=False) for site, path in paths.items()}
    for site, df in dataframes.items():
        df['Site'] = site
    return pd.concat(dataframes.values(), ignore_index=True)

# Main function defining the application
def load_default_data():
    # Paths to default data
    default_btn_path = 'default_Site-BTN_Sales_History.CSV'
    default_dor_path = 'default_Site-DOR_Sales_History.CSV'
    default_tor_path = 'default_Site-TOR_Sales_History.CSV'

    # Load data
    df_btn = pd.read_csv(default_btn_path, encoding='ISO-8859-1', low_memory=False)
    df_dor = pd.read_csv(default_dor_path, encoding='ISO-8859-1', low_memory=False)
    df_tor = pd.read_csv(default_tor_path, encoding='ISO-8859-1', low_memory=False)

    df_btn['Site'] = 'BTN'
    df_dor['Site'] = 'DOR'
    df_tor['Site'] = 'TOR'

    df = pd.concat([df_btn, df_dor, df_tor], ignore_index=True)
    return df

def main():
    if 'data_uploaded' not in st.session_state or not st.session_state.data_uploaded:
        df = load_default_data()
        st.session_state.df = df
        st.session_state.data_uploaded = True  # Mark as data uploaded

    tabs = st.tabs(["1.Load Your Data", "2.Dashboard", "3. Generate AI Insights"])

    with tabs[0]:
        st.markdown("<h3 style='font-size:20px;color: #164871'>Upload Updated Sales Data </h3>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            uploaded_btn = st.file_uploader("Upload Site-BTN Sales History CSV", type="csv")
        with col2:
            uploaded_dor = st.file_uploader("Upload Site-DOR Sales History CSV", type="csv")
        with col3:
            uploaded_tor = st.file_uploader("Upload Site-TOR Sales History CSV", type="csv")

        if st.button("Submit"):
            if uploaded_btn and uploaded_dor and uploaded_tor:
                df_btn = pd.read_csv(uploaded_btn, encoding='ISO-8859-1', low_memory=False)
                df_dor = pd.read_csv(uploaded_dor, encoding='ISO-8859-1', low_memory=False)
                df_tor = pd.read_csv(uploaded_tor, encoding='ISO-8859-1', low_memory=False)

                df_btn['Site'] = 'BTN'
                df_dor['Site'] = 'DOR'
                df_tor['Site'] = 'TOR'

                df = pd.concat([df_btn, df_dor, df_tor], ignore_index=True)
                st.session_state.df = df
                st.session_state.data_uploaded = True
                st.success("Data uploaded and combined successfully!")
            else:
                st.error("Please upload all three CSV files.")


    # Tab 2: Dashboard
    if 'data_uploaded' in st.session_state and st.session_state.data_uploaded:
        df = st.session_state.df

        with tabs[1]:
            # Data Cleaning Functions
            def convert_date_to_datetime(df, date_col):
                df[date_col] = pd.to_datetime(
                    df[date_col], format='%y%m%d', errors='coerce')
                return df

            df = convert_date_to_datetime(df, 'Shipped Date')

            def add_unit_price_usd(df):
                df['Unit Price USD'] = pd.NA
                df.loc[df['Based Currency'] == 'USD',
                       'Unit Price USD'] = df['Unit Price']
                df.loc[df['Based Currency'] == 'CAD',
                       'Unit Price USD'] = df['Unit Price'] / 1.36
                return df

            df = add_unit_price_usd(df)

            def calculate_sales_total_amount_usd(row):
                if pd.notnull(row['Unit Price USD']) and pd.notnull(row['Quantity']):
                    return row['Unit Price USD'] * row['Quantity']
                return None

            df['Sales USD'] = df.apply(
                calculate_sales_total_amount_usd, axis=1)

            def add_unit_cost_usd(df):
                df['Unit Cost USD'] = pd.NA
                df.loc[df['Based Currency'] == 'USD',
                       'Unit Cost USD'] = df['Unit Cost']
                df.loc[df['Based Currency'] == 'CAD',
                       'Unit Cost USD'] = df['Unit Cost'] / 1.36
                return df

            df = add_unit_cost_usd(df)

            def calculate_total_cost_usd(row):
                if pd.notnull(row['Unit Cost USD']) and pd.notnull(row['Quantity']):
                    return row['Unit Cost USD'] * row['Quantity']
                return None

            df['Total Cost USD'] = df.apply(
                calculate_total_cost_usd, axis=1)

            def add_profit_usd_column(df):
                df['Profit USD'] = df.apply(
                    lambda row: row['Sales USD'] -
                    row['Total Cost USD']
                    if pd.notnull(row['Sales USD']) and pd.notnull(row['Total Cost USD'])
                    else None,
                    axis=1
                )
                return df

            df = add_profit_usd_column(df)

            def calculate_margin(df):
                df['Margin %'] = df['Profit USD'] / \
                    df['Sales USD']
                df['Margin %'] = df['Margin %'].fillna(
                    0).replace([float('inf'), -float('inf')], 0)
                return df

            df = calculate_margin(df)

            #  create the 'Customer Name' column based on 'Ship to Name'
            def create_customer_name_column(df):
                df['Customer Name'] = df['Ship to Name']
                return df
            df = create_customer_name_column(df)

            #  create the 'Sales Rep Name' column based on 'Name.1'
            def create_customer_name_column(df):
                df['Sales Rep Name'] = df['Name.1']
                return df
            df = create_customer_name_column(df)

            def delete_rows_with_value(df, column_name, value_to_delete):
                df = df[df[column_name] != value_to_delete]
                return df

            df = delete_rows_with_value(df, 'Family', 'D9')

            # st.success("Data cleaned successfully!")

            # Step 3: Build visualizations

            # Convert the 'Shipped Date' to Year
            df['Year'] = pd.to_datetime(
                df['Shipped Date']).dt.strftime('%Y')

            # Filters inside the Dashboard tab
            with st.container():
                

                # Creating four columns in one row
                col1, col2, col3, col4 = st.columns(4)

                # Adding filters to each column
                with col1:
                    year_filter = st.multiselect(
                        "Select Year", options=df['Year'].unique())

                with col2:
                    site_filter = st.multiselect(
                        "Select Site", options=df['Site'].unique())

                with col3:
                    market_segment_filter = st.multiselect(
                        "Select Market Segment", options=df['Market Segment'].unique())

                with col4:
                    sales_rep_filter = st.multiselect(
                        "Select Sales Rep", options=df['Sales Rep Name'].unique())

            def load_custom_css():
                st.markdown("""
                    <style>
                    hr {
                        margin-top: 0px; /* Reduces the space above the horizontal line */
                        margin-bottom: 0px; /* Reduces the space below the horizontal line */
                    }
                    </style>
                    """, unsafe_allow_html=True)
            load_custom_css()
            # Insert a horizontal line
            st.markdown("---")
           
            # Apply filters to the dataframe
            filtered_df = df.copy()

            if year_filter:
                filtered_df = filtered_df[filtered_df['Year'].isin(
                    year_filter)]

            if site_filter:
                filtered_df = filtered_df[filtered_df['Site'].isin(
                    site_filter)]

            if market_segment_filter:
                filtered_df = filtered_df[filtered_df['Market Segment'].isin(
                    market_segment_filter)]

            if sales_rep_filter:
                filtered_df = filtered_df[filtered_df['Sales Rep Name'].isin(
                    sales_rep_filter)]

            # Create a container to hold all rows and columns
            with st.container():
                # ROW 1 with 3 Columns
                row1_col1, row1_col2, row1_col3 = st.columns(3)

                # Viz 1: Sales & Profit Overview
                with row1_col1:
                    def sales_profit_overview_table(df):
                        total_profit = df['Profit USD'].sum()
                        total_sales = df['Sales USD'].sum()
                        total_margin = (
                            total_profit / total_sales) if total_sales != 0 else 0
                        formatted_margin = f"{total_margin:.2%}"
                        summary_df = pd.DataFrame({
                            'Metric': ['Total Sales USD(M)', 'Total Profit USD(M)', 'Margin %'],
                            'Value': [
                                round(total_sales / 1_000_000, 2),
                                round(total_profit / 1_000_000, 2),
                                formatted_margin
                            ]
                        })
                        # Drop the index
                        return summary_df.reset_index(drop=True)


                    st.markdown(
                    """
                    <h3 style='font-size:20px; color: #164871; border-bottom: 1px solid #164871; padding-bottom: 3px;'>
                        Sales & Profit Overview
                    </h3>
                    """, unsafe_allow_html=True)
                    st.dataframe(sales_profit_overview_table(filtered_df))

                # Viz 2: Site Sales & Profit
                with row1_col2:
                    def site_sales_profit_summary(df):
                        grouped_df = df.groupby('Site').agg({
                            'Sales USD': 'sum',
                            'Profit USD': 'sum'
                        }).reset_index()
                        grouped_df['Margin %'] = grouped_df.apply(
                            lambda row: row['Profit USD'] /
                            row['Sales USD'] if row['Sales USD'] != 0 else 0,
                            axis=1
                        )
                        grouped_df['Margin %'] = grouped_df['Margin %'].apply(lambda x: f"{
                                                                              x:.2%}")
                        grouped_df['Sales USD(M)'] = grouped_df['Sales USD'] / \
                            1_000_000
                        grouped_df['Profit USD(M)'] = grouped_df['Profit USD'] / \
                            1_000_000
                        summary_df = grouped_df[[
                            'Site', 'Sales USD(M)', 'Profit USD(M)', 'Margin %']]
                        summary_df['Sales USD(M)'] = summary_df['Sales USD(M)'].round(
                            2)
                        summary_df['Profit USD(M)'] = summary_df['Profit USD(M)'].round(
                            2)
                        summary_df = summary_df.sort_values(
                            by='Sales USD(M)', ascending=False)
                        return summary_df
                    
                    st.markdown(
                    """
                    <h3 style='font-size:20px; color: #164871; border-bottom: 1px solid #164871; padding-bottom: 3px;'>
                        Sales Per Site
                    </h3>
                    """, unsafe_allow_html=True)
                    st.dataframe(site_sales_profit_summary(filtered_df))

                # Viz 3: Sales Distribution per Site (Donut Chart)
                with row1_col3:
                    st.markdown(
                    """
                    <h3 style='font-size:20px; color: #164871; border-bottom: 1px solid #164871; padding-bottom: 3px;'>
                        Sales Distribution per Site
                    </h3>
                    """, unsafe_allow_html=True)
                    # Filter out only numeric columns before applying the groupby and sum operation
                    filtered_df_numeric = filtered_df[[
                        'Site', 'Sales USD']].copy()
                    filtered_df_numeric = filtered_df_numeric.groupby(
                        'Site').sum().reset_index()

                    # Create Donut Chart with adjusted size and legend position
                    fig = px.pie(
                        filtered_df_numeric,
                        values='Sales USD',
                        names='Site',
                        hole=0.5,  # This makes it a donut chart
                        color_discrete_sequence=['#AECDE7', '#7BA8CD', '#5480A5']
                    )
                    fig.update_layout(
                        width=210,  # Reduce the width of the chart
                        height=210,  # Reduce the height of the chart
                        # Adjust margins to fit better
                        margin=dict(l=0, r=0, t=0, b=0),
                        legend=dict(
                            orientation="v",  # Vertical legend
                            y=0.5,  # Center vertically
                            x=1.1,  # Position to the right of the chart
                            xanchor='left',  # Align legend to the left
                            yanchor='middle'  # Align legend to the middle
                        )
                    )
                    st.plotly_chart(fig, use_container_width=False)

                # ROW 2 with 2 Columns

                row2_col1, row2_col2, row2_col3 = st.columns(3)

                with row2_col1:
                    def market_segment_sales_profit_summary(df):
                        grouped_df = df.groupby('Market Segment').agg({
                            'Sales USD': 'sum',
                            'Profit USD': 'sum'
                        }).reset_index()
                        grouped_df['Margin %'] = grouped_df.apply(
                            lambda row: row['Profit USD'] /
                            row['Sales USD'] if row['Sales USD'] != 0 else 0,
                            axis=1
                        )
                        grouped_df['Margin %'] = grouped_df['Margin %'].apply(lambda x: f"{
                                                                              x:.2%}")
                        grouped_df['Sales USD(M)'] = grouped_df['Sales USD'] / \
                            1_000_000
                        grouped_df['Profit USD(M)'] = grouped_df['Profit USD'] / \
                            1_000_000
                        summary_df = grouped_df[[
                            'Market Segment', 'Sales USD(M)', 'Profit USD(M)', 'Margin %']]
                        summary_df['Sales USD(M)'] = summary_df['Sales USD(M)'].round(
                            2)
                        summary_df['Profit USD(M)'] = summary_df['Profit USD(M)'].round(
                            2)
                        summary_df = summary_df.sort_values(
                            by='Sales USD(M)', ascending=False).reset_index(drop=True)
                        return summary_df.head(5)

                    
                    st.markdown(
                    """
                    <h3 style='font-size:20px; color: #164871; border-bottom: 1px solid #164871; padding-bottom: 3px;'>
                        Sales Per Market Segment (Top 5)
                    </h3>
                    """, unsafe_allow_html=True)
                    st.dataframe(
                        market_segment_sales_profit_summary(filtered_df))

                with row2_col2:
                    def sales_rep_sales_profit_summary(df):
                        grouped_df = df.groupby('Sales Rep Name').agg({
                            'Sales USD': 'sum',
                            'Profit USD': 'sum'
                        }).reset_index()
                        grouped_df['Margin %'] = grouped_df.apply(
                            lambda row: row['Profit USD'] /
                            row['Sales USD'] if row['Sales USD'] != 0 else 0,
                            axis=1
                        )
                        grouped_df['Margin %'] = grouped_df['Margin %'].apply(lambda x: f"{
                                                                              x:.2%}")
                        grouped_df['Sales USD(M)'] = grouped_df['Sales USD'] / \
                            1_000_000
                        grouped_df['Profit USD(M)'] = grouped_df['Profit USD'] / \
                            1_000_000
                        summary_df = grouped_df[[
                            'Sales Rep Name', 'Sales USD(M)', 'Profit USD(M)', 'Margin %']]
                        summary_df['Sales USD(M)'] = summary_df['Sales USD(M)'].round(
                            2)
                        summary_df['Profit USD(M)'] = summary_df['Profit USD(M)'].round(
                            2)
                        summary_df = summary_df.sort_values(
                            by='Sales USD(M)', ascending=False).reset_index(drop=True)
                        return summary_df.head(5)

                
                    st.markdown(
                    """
                    <h3 style='font-size:20px; color: #164871; border-bottom: 1px solid #164871; padding-bottom: 3px;'>
                        Per Sales Rep (Top 5)
                    </h3>
                    """, unsafe_allow_html=True)
                    st.dataframe(sales_rep_sales_profit_summary(filtered_df))

                with row2_col3:
                    def customer_sales_profit_summary(df):
                        grouped_df = df.groupby('Customer Name').agg({
                            'Sales USD': 'sum',
                            'Profit USD': 'sum'
                        }).reset_index()
                        grouped_df['Margin %'] = grouped_df.apply(
                            lambda row: row['Profit USD'] /
                            row['Sales USD'] if row['Sales USD'] != 0 else 0,
                            axis=1
                        )
                        grouped_df['Margin %'] = grouped_df['Margin %'].apply(lambda x: f"{
                                                                              x:.2%}")
                        grouped_df['Sales USD(M)'] = grouped_df['Sales USD'] / \
                            1_000_000
                        grouped_df['Profit USD(M)'] = grouped_df['Profit USD'] / \
                            1_000_000
                        summary_df = grouped_df[[
                            'Customer Name', 'Sales USD(M)', 'Profit USD(M)', 'Margin %']]
                        summary_df['Sales USD(M)'] = summary_df['Sales USD(M)'].round(
                            2)
                        summary_df['Profit USD(M)'] = summary_df['Profit USD(M)'].round(
                            2)
                        summary_df = summary_df.sort_values(
                            by='Sales USD(M)', ascending=False).reset_index(drop=True)
                        # Display only the first 5 rows
                        return summary_df.head(5)

                    st.markdown(
                    """
                    <h3 style='font-size:20px; color: #164871; border-bottom: 1px solid #164871; padding-bottom: 3px;'>
                        Per Customer Name (Top 5)
                    </h3>
                    """, unsafe_allow_html=True)
                    st.dataframe(customer_sales_profit_summary(filtered_df))

                # ROW 3 with 2 Columns
                row3_col1, row3_col2, row3_col3 = st.columns(3)

                with row3_col1:
                    # Monthly Sales
                    def monthly_sales_viz(df):
                        # Convert 'Shipped Date' to datetime and extract the month names
                        df['Month'] = pd.to_datetime(
                            df['Shipped Date']).dt.strftime('%b')

                        # Group by Month and calculate aggregations
                        grouped_df = df.groupby('Month', as_index=False).agg({
                            'Sales USD': 'sum'
                        })

                        # Ensure the month order is correct
                        month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May',
                                       'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                        grouped_df['Month'] = pd.Categorical(
                            grouped_df['Month'], categories=month_order, ordered=True)
                        grouped_df = grouped_df.sort_values('Month')

                        # Plotting using Seaborn
                        plt.figure(figsize=(10, 6))
                        ax = sns.barplot(
                            data=grouped_df, x='Month', y='Sales USD', color="#8DA6BB")

                        # Annotate the bars with values above the bars using the text() function
                        for i, value in enumerate(grouped_df['Sales USD']):
                            plt.text(i, value + 0.02 * max(grouped_df['Sales USD']), f"{
                                     value/1_000_000:.1f}M", ha='center', fontsize=12)

                        # Hide the y-axis labels and ticks
                        ax.yaxis.set_visible(False)
                        # Make the month labels bigger
                        ax.set_xticklabels(
                            ax.get_xticklabels(), fontsize=14)

                        # Remove the x-axis label "Month"
                        ax.set_xlabel('')

                        # Remove the border
                        sns.despine(left=True, bottom=True)
                        st.pyplot(plt)

                   
                    st.markdown(
                    """
                    <h3 style='font-size:20px; color: #164871; border-bottom: 1px solid #164871; padding-bottom: 3px;'>
                        Sales per Month
                    </h3>
                    """, unsafe_allow_html=True)
                    monthly_sales_viz(filtered_df)

                with row3_col2:
                    # Monthly Profit
                    def monthly_profit_viz(df):
                        # Convert 'Shipped Date' to datetime and extract the month names
                        df['Month'] = pd.to_datetime(
                            df['Shipped Date']).dt.strftime('%b')

                        # Group by Month and calculate aggregations
                        grouped_df = df.groupby('Month', as_index=False).agg({
                            'Profit USD': 'sum'
                        })

                        # Ensure the month order is correct
                        month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May',
                                       'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                        grouped_df['Month'] = pd.Categorical(
                            grouped_df['Month'], categories=month_order, ordered=True)
                        grouped_df = grouped_df.sort_values('Month')

                        # Plotting using Seaborn
                        plt.figure(figsize=(10, 6))
                        ax = sns.barplot(
                            data=grouped_df, x='Month', y='Profit USD', color="#346E9F")

                        # Annotate the bars with values above the bars using the text() function
                        for i, value in enumerate(grouped_df['Profit USD']):
                            plt.text(i, value + 0.02 * max(grouped_df['Profit USD']), f"{
                                     value/1_000_000:.1f}M", ha='center', fontsize=12)

                        # Hide the y-axis labels and ticks
                        ax.yaxis.set_visible(False)
                        # Make the month labels bigger
                        ax.set_xticklabels(
                            ax.get_xticklabels(), fontsize=14)

                        # Remove the x-axis label "Month"
                        ax.set_xlabel('')

                        # Remove the border
                        sns.despine(left=True, bottom=True)
                        st.pyplot(plt)

                    
                    st.markdown(
                    """
                    <h3 style='font-size:20px; color: #164871; border-bottom: 1px solid #164871; padding-bottom: 3px;'>
                        Profit per Month
                    </h3>
                    """, unsafe_allow_html=True)
                    monthly_profit_viz(filtered_df)

                with row3_col3:
                    # Monthly Margin %
                    def monthly_margin_viz(df):
                        # Convert 'Shipped Date' to datetime and extract the month names
                        df['Month'] = pd.to_datetime(
                            df['Shipped Date']).dt.strftime('%b')

                        # Group by Month and calculate aggregations
                        grouped_df = df.groupby('Month', as_index=False).agg({
                            'Sales USD': 'sum',
                            'Profit USD': 'sum'
                        })

                        # Calculate Margin %
                        grouped_df['Margin %'] = (
                            grouped_df['Profit USD'] / grouped_df['Sales USD']) * 100

                        # Ensure the month order is correct
                        month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May',
                                       'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                        grouped_df['Month'] = pd.Categorical(
                            grouped_df['Month'], categories=month_order, ordered=True)
                        grouped_df = grouped_df.sort_values('Month')

                        # Plotting using Seaborn
                        plt.figure(figsize=(10, 6))
                        ax = sns.barplot(
                            data=grouped_df, x='Month', y='Margin %', color="#549CD8")

                        # Annotate the bars with values above the bars using the text() function
                        for i, value in enumerate(grouped_df['Margin %']):
                            plt.text(
                                i, value + 0.02 * max(grouped_df['Margin %']), f"{value:.1f}%", ha='center', fontsize=12)

                        # Hide the y-axis labels and ticks
                        ax.yaxis.set_visible(False)
                        # Make the month labels bigger
                        ax.set_xticklabels(
                            ax.get_xticklabels(), fontsize=14)

                        # Remove the x-axis label "Month"
                        ax.set_xlabel('')

                        # Remove the border
                        sns.despine(left=True, bottom=True)
                        st.pyplot(plt)

                    st.markdown(
                    """
                    <h3 style='font-size:20px; color: #164871; border-bottom: 1px solid #164871; padding-bottom: 3px;'>
                        Margin %
                    </h3>
                    """, unsafe_allow_html=True)
                    monthly_margin_viz(filtered_df)
    else:
     st.markdown("<h5 style='color: #5480A5;'>Upload Files in the Tab: Load Your Data</h5>", unsafe_allow_html=True)
    # Tab 3: AI Insights
    if 'data_uploaded' in st.session_state and st.session_state.data_uploaded:
        df = st.session_state.df

        with tabs[2]:
            

            # Ensure that data is loaded
            if 'df' not in st.session_state:
                st.error(
                    "Please upload the data before generating AI insights.")
                return

            load_dotenv()
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                st.error(
                    "API key not found. Please set your OPENAI_API_KEY environment variable.")
            else:
                # Set the OpenAI API key
                openai.api_key = api_key

            # Function to calculate percentage change

            def calculate_percentage_change(current, previous):
                if previous == 0:
                    return 0  # or return "N/A" to indicate no comparison possible
                return ((current - previous) / previous) * 100

            # Function to clean and convert relevant columns to numeric
            def clean_and_convert_to_numeric(df):
                numeric_cols = [
                    'Sales USD', 'Profit USD', 'Margin %',
                    'Unit Price USD', 'Total Cost USD', 'Quantity'
                ]
                for col in numeric_cols:
                    df[col] = pd.to_numeric(
                        df[col], errors='coerce').fillna(0)
                return df

            # Function to generate AI insights
            def generate_ai_insights(query, start_date=None, end_date=None, previous_start_date=None, previous_end_date=None,
                                     market_segment_1=None, market_segment_2=None, sales_rep_1=None, sales_rep_2=None,
                                     customer_1=None, customer_2=None):
                if query:
                    # Initialize variables to avoid NameErrors
                    total_sales = total_profit = avg_margin = 0
                    sales_change = profit_change = margin_change = "N/A"
                    site_changes = {}
                    best_market_segment = potential_market_segment = best_sales_rep = most_improved_sales_rep = "N/A"
                    best_customer = most_improved_customer = "N/A"

                    # Convert dates to datetime64[ns] for comparison
                    if start_date and end_date and previous_start_date and previous_end_date:
                        start_date = pd.to_datetime(start_date)
                        end_date = pd.to_datetime(end_date)
                        previous_start_date = pd.to_datetime(
                            previous_start_date)
                        previous_end_date = pd.to_datetime(
                            previous_end_date)

                        filtered_df = df[(df['Shipped Date'] >= start_date) & (
                            df['Shipped Date'] <= end_date)]
                        previous_df = df[(df['Shipped Date'] >= previous_start_date) & (
                            df['Shipped Date'] <= previous_end_date)]
                    else:
                        filtered_df = df
                        previous_df = pd.DataFrame()  # Empty DataFrame if no previous period is provided

                    # Apply additional filters
                    if market_segment_1 and market_segment_1 != "All":
                        filtered_df = filtered_df[filtered_df['Market Segment']
                                                  == market_segment_1]
                    if market_segment_2 and market_segment_2 != "All" and not previous_df.empty:
                        previous_df = previous_df[previous_df['Market Segment']
                                                  == market_segment_2]

                    if sales_rep_1 and sales_rep_1 != "All":
                        filtered_df = filtered_df[filtered_df['Sales Rep Name']
                                                  == sales_rep_1]
                    if sales_rep_2 and sales_rep_2 != "All" and not previous_df.empty:
                        previous_df = previous_df[previous_df['Sales Rep Name']
                                                  == sales_rep_2]

                    if customer_1 and customer_1 != "All":
                        filtered_df = filtered_df[filtered_df['Customer Name']
                                                  == customer_1]
                    if customer_2 and customer_2 != "All" and not previous_df.empty:
                        previous_df = previous_df[previous_df['Customer Name']
                                                  == customer_2]

                    # Clean and convert data to numeric
                    filtered_df = clean_and_convert_to_numeric(filtered_df)
                    if not previous_df.empty:
                        previous_df = clean_and_convert_to_numeric(
                            previous_df)

                    # Summary calculations for current period
                    if not filtered_df.empty:
                        total_sales = filtered_df['Sales USD'].sum()
                        total_profit = filtered_df['Profit USD'].sum()
                        avg_margin = filtered_df['Margin %'].mean()

                    # Summary calculations for previous period
                    if not previous_df.empty:
                        previous_total_sales = previous_df['Sales USD'].sum(
                        )
                        previous_total_profit = previous_df['Profit USD'].sum(
                        )
                        previous_avg_margin = previous_df['Margin %'].mean(
                        )

                        sales_change = calculate_percentage_change(
                            total_sales, previous_total_sales)
                        profit_change = calculate_percentage_change(
                            total_profit, previous_total_profit)
                        margin_change = calculate_percentage_change(
                            avg_margin, previous_avg_margin)

                    # Site comparison
                    if not filtered_df.empty:
                        site_sales = filtered_df.groupby(
                            'Site')['Sales USD'].sum()
                        previous_site_sales = previous_df.groupby(
                            'Site')['Sales USD'].sum() if not previous_df.empty else pd.Series()
                        site_changes = {site: calculate_percentage_change(
                            site_sales[site], previous_site_sales.get(site, 0)) for site in site_sales.index}

                    # Market segment analysis
                    if not filtered_df.empty:
                        market_segment_profit = filtered_df.groupby('Market Segment')[
                            'Profit USD'].sum()
                        best_market_segment = market_segment_profit.idxmax()
                        potential_market_segment = market_segment_profit.idxmin()
                        
                    # Customer analysis
                    if not filtered_df.empty:
                        customer_sales = filtered_df.groupby('Customer Name')['Sales USD'].sum()
                        best_customer = customer_sales.idxmax()  # Best Customer
                        previous_customer_sales = previous_df.groupby('Customer Name')['Sales USD'].sum() if not previous_df.empty else pd.Series()
                        customer_changes = {cust: calculate_percentage_change(
                            customer_sales[cust], previous_customer_sales.get(cust, 0)) for cust in customer_sales.index}

                        # Ensure all values in customer_changes are numeric
                        customer_changes = {cust: change for cust, change in customer_changes.items() if isinstance(change, (int, float))}

                        if customer_changes:
                            most_improved_customer = max(customer_changes, key=customer_changes.get)

                    # Sales representative analysis
                    if not filtered_df.empty:
                        sales_rep_sales = filtered_df.groupby('Sales Rep Name')[
                            'Sales USD'].sum()
                        best_sales_rep = sales_rep_sales.idxmax()  # Best Sales Rep
                        previous_sales_rep_sales = previous_df.groupby(
                            'Sales Rep Name')['Sales USD'].sum() if not previous_df.empty else pd.Series()
                        sales_rep_changes = {rep: calculate_percentage_change(
                            sales_rep_sales[rep], previous_sales_rep_sales.get(rep, 0)) for rep in sales_rep_sales.index}

                        # Ensure all values in sales_rep_changes are numeric
                        sales_rep_changes = {rep: change for rep, change in sales_rep_changes.items(
                        ) if isinstance(change, (int, float))}

                        if sales_rep_changes:
                            most_improved_sales_rep = max(
                                sales_rep_changes, key=sales_rep_changes.get)

                    # Create a detailed summary
                    summary = (
                        "1. **Total Sales, Margin, Profit:**\n"
                        "   - **Total Sales:** ${:,.0f} USD (Change from last period: {}%)\n".format(total_sales, sales_change) +
                        "   - **Total Profit:** ${:,.0f} USD (Change from last period: {}%)\n".format(total_profit, profit_change) +

                        "2. **Per Site Sales Comparison:**\n" +
                        ''.join(["   - **{}:** {}%\n".format(site, site_changes.get(site, 'N/A')) for site in site_sales.index]) + "\n\n" +

                        "3. **Per Market Segment:**\n"
                        "   - **Best Performing Segment:** {} with Profit ${:,.0f} USD\n".format(best_market_segment, market_segment_profit.get(best_market_segment, 0)) +
                        
                        "4. **Per Customer:**\n"
                        "   - **Best Customer:** {} with Sales ${:,.0f} USD\n".format(best_customer, customer_sales.get(best_customer, 0)) +
                        "   - **Most Improved Customer:** {} with an increase of ${:,.0f} USD\n\n".format(
                        most_improved_customer, customer_changes.get(most_improved_customer, 'N/A')) +


                        "5. **Per Sales Representative:**\n"
                        "   - **Best Sales Rep:** {} with Sales ${:,.0f} USD\n".format(best_sales_rep, sales_rep_sales.get(best_sales_rep, 0)) +
                        "   - **Most Improved Sales Rep:** {} with an increase of {:.2f}%\n\n".format(
                            most_improved_sales_rep, sales_rep_changes.get(most_improved_sales_rep, 'N/A'))
                    )

                    # Extend the query to include this summary
                    extended_query = f"{
                        query}. Here is a summary of the data: {summary}"

                    try:
                        response = openai.ChatCompletion.create(
                            model="gpt-4o-mini",  # Replace with the appropriate model version
                            messages=[
                                {"role": "system", "content": "You are an assistant specialized in analyzing sales data."},
                                {"role": "user", "content": extended_query}
                            ]
                        )

                        # Process the response to remove any unintended artifacts
                        response_content = response.choices[0].message['content']
                        response_content = response_content.replace(
                            '*', '').replace('_', '').strip()

                        # Format the response to maintain the numbering and structure
                        formatted_response = response_content.replace(
                            '\n', '\n\n')

                        return formatted_response

                    except Exception as e:
                        return f"An error occurred: {e}"
                return ""

            # AI Insights Section in Streamlit
            def display_ai_insights():

                # Replace these column names with the correct ones if different
                market_segments = df["Market Segment"].unique(
                ).tolist()
                sales_rep_names = df["Sales Rep Name"].unique(
                ).tolist()
                customers = df["Customer Name"].unique().tolist()

                # Add "All" option at the beginning
                market_segments.insert(0, "All")
                sales_rep_names.insert(0, "All")
                customers.insert(0, "All")

                with st.form(key='date_filters'):
                    col1, col2 = st.columns(2)

                    with col1:
                        
                        st.markdown(
                        """
                        <h3 style='font-size:20px; color: #164871; border-bottom: 1px solid #164871; padding-bottom: 3px; margin-bottom: 10px;'>
                            Select Filters for Current Period
                        </h3>
                        """, unsafe_allow_html=True) 
                        
                        date_col1, date_col2 = st.columns(2)
                        with date_col1:
                            start_date = st.date_input(
                                "Start Date", value=pd.to_datetime("2023-01-01"))
                        with date_col2:
                            end_date = st.date_input(
                                "End Date", value=pd.to_datetime("2023-12-31"))
                        st.markdown("---")
                        market_segment_1 = st.selectbox(
                            "Market Segment 1", options=market_segments, key="market_segment_1")

                        sales_rep_1 = st.selectbox(
                            "Sales Rep Name 1", options=sales_rep_names, key="sales_rep_1")

                        customer_1 = st.selectbox(
                            "Customer 1", options=customers, key="customer_1")

                    with col2:

                        st.markdown(
                        """
                        <h3 style='font-size:20px; color: #164871; border-bottom: 1px solid #164871; padding-bottom: 3px; margin-bottom: 10px;'>
                            Select Filters for Past Period to Compare
                        </h3>
                        """, unsafe_allow_html=True)
                        date_col3, date_col4 = st.columns(2)
                        with date_col3:
                            prev_start_date = st.date_input(
                                "Previous Start Date", value=pd.to_datetime("2022-01-01"))
                        with date_col4:
                            prev_end_date = st.date_input(
                                "Previous End Date", value=pd.to_datetime("2022-12-31"))
                        st.markdown("---")
                        market_segment_2 = st.selectbox(
                            "Market Segment 2 to compare", options=market_segments, key="market_segment_2")

                        sales_rep_2 = st.selectbox(
                            "Sales Rep Name 2 to compare", options=sales_rep_names, key="sales_rep_2")

                        customer_2 = st.selectbox(
                            "Customer 2 to compare", options=customers, key="customer_2")

                    # Submit button within the form context
                    submit_button = st.form_submit_button(
                        label='Generate AI Insights')

                # Process the form submission
                if submit_button:
                    with st.spinner(text='In progress'):
                    # Hardcoded query
                        user_query = "Analyze the sales trends from this dashboard."

                        ai_response = generate_ai_insights(
                            user_query, start_date, end_date, prev_start_date, prev_end_date,
                            market_segment_1, market_segment_2, sales_rep_1, sales_rep_2, customer_1, customer_2
                        )
                        st.markdown(ai_response)

            # Call this function where the AI Insights section is supposed to be
            display_ai_insights()


if __name__ == "__main__":
    main()
