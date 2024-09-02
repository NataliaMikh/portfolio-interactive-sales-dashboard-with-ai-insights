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
import matplotlib.ticker as ticker

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
st.markdown('<div class="fixed-title">Sales Analytics Dashboard</div>',
            unsafe_allow_html=True)

# Function to load default data
# Function to load data (default or user-uploaded)


# Function to load data (default or user-uploaded)
# Function to load data (default or user-uploaded)
def load_data(uploaded_files):
    # Default paths
    paths = {
        'BTN': 'https://raw.githubusercontent.com/NGravereaux/interactive-sales-dashboard-with-ai-insights/main/default_Site-BTN_Sales_History.CSV',
        'DOR': 'https://raw.githubusercontent.com/NGravereaux/interactive-sales-dashboard-with-ai-insights/main/default_Site-DOR_Sales_History.CSV',
        'TOR': 'https://raw.githubusercontent.com/NGravereaux/interactive-sales-dashboard-with-ai-insights/main/default_Site-TOR_Sales_History.CSV'
    }

    # Function to read CSV from either a file or URL
    def read_csv_or_default(uploaded_file, default_url):
        if uploaded_file is not None:
            return pd.read_csv(uploaded_file, encoding='ISO-8859-1', low_memory=False)
        else:
            return pd.read_csv(default_url, encoding='ISO-8859-1', low_memory=False)

    # Load data from either uploaded files or default URLs
    dataframes = {site: read_csv_or_default(
        uploaded_files.get(site), path) for site, path in paths.items()}

    # Add site names
    for site, df in dataframes.items():
        df['Site'] = site

    # Concatenate all dataframes
    df = pd.concat(dataframes.values(), ignore_index=True)
    return df


def main():
    # Initialize session state for data upload tracking and current tab
    if 'data_uploaded' not in st.session_state:
        st.session_state.data_uploaded = False

    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = 0  # Default tab

    # Create tabs for navigation
    tabs = st.tabs(["1.Load Updated Data", "2.Dashboard",
                   "3.Generate AI Insights"])

    # Tab 0 for loading updated data
    with tabs[0]:
        st.markdown("### Upload Sales History CSV Files")

        # Display file uploaders in one line
        col1, col2, col3 = st.columns(3)
        with col1:
            uploaded_btn = st.file_uploader(
                "Upload Site-BTN CSV", type="csv", key="btn_upload")
        with col2:
            uploaded_dor = st.file_uploader(
                "Upload Site-DOR CSV", type="csv", key="dor_upload")
        with col3:
            uploaded_tor = st.file_uploader(
                "Upload Site-TOR CSV", type="csv", key="tor_upload")

        uploaded_files = {
            'BTN': uploaded_btn,
            'DOR': uploaded_dor,
            'TOR': uploaded_tor
        }

        if st.button("Submit"):
            if any(uploaded_files.values()):
                df = load_data(uploaded_files)
                st.session_state.df = df
                st.session_state.data_uploaded = True
                st.success("Data uploaded and combined successfully!")
            else:
                st.error(
                    "Please upload all three CSV files or proceed to use default data.")

        # If no files are uploaded and the user has not clicked submit, load default data automatically
        if not any(uploaded_files.values()) and not st.session_state.data_uploaded:
            df = load_data(uploaded_files)  # This will load default data
            st.session_state.df = df
            st.session_state.data_uploaded = True
            st.success("Default data loaded successfully!")

    # Tab 1: Dashboard
    with tabs[1]:
        if 'data_uploaded' in st.session_state and st.session_state.data_uploaded:
            df = st.session_state.df

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

            df['Total Cost USD'] = df.apply(calculate_total_cost_usd, axis=1)

            def add_profit_usd_column(df):
                df['Profit USD'] = df.apply(
                    lambda row: row['Sales USD'] - row['Total Cost USD']
                    if pd.notnull(row['Sales USD']) and pd.notnull(row['Total Cost USD'])
                    else None,
                    axis=1
                )
                return df

            df = add_profit_usd_column(df)

            def calculate_margin(df):
                df['Margin %'] = df['Profit USD'] / df['Sales USD']
                df['Margin %'] = df['Margin %'].fillna(
                    0).replace([float('inf'), -float('inf')], 0)
                return df

            df = calculate_margin(df)

            # create the 'Customer Name' column based on 'Ship to Name'
            def create_customer_name_column(df):
                df['Customer Name'] = df['Ship to Name']
                return df
            df = create_customer_name_column(df)

            # create the 'Sales Rep Name' column based on 'Name.1'
            def create_customer_name_column(df):
                df['Sales Rep Name'] = df['Name.1']
                return df
            df = create_customer_name_column(df)

            def delete_rows_with_value(df, column_name, value_to_delete):
                df = df[df[column_name] != value_to_delete]
                return df

            df = delete_rows_with_value(df, 'Family', 'D9')

            # st.success("Data cleaned successfully!")

            # Convert the 'Shipped Date' to Year
            df.loc[:, 'Year'] = pd.to_datetime(
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
            if not filtered_df.empty:
                # Step 3: Build visualizations
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
                            formatted_margin = round(total_margin * 100, 2)
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
                            summary_df.loc[:, 'Sales USD(M)'] = summary_df['Sales USD(M)'].round(
                                2)
                            summary_df.loc[:, 'Profit USD(M)'] = summary_df['Profit USD(M)'].round(
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

                    # Viz 3: Sales Distribution per Site (Horizontal Stacked Bar Chart)
                    with row1_col3:
                        st.markdown("""
                        <h3 style='font-size:20px; color: #164871; border-bottom: 1px solid #164871; padding-bottom: 3px;'>
                            Sales Distribution per Site
                        </h3>
                        """, unsafe_allow_html=True)

                        # Filter out only numeric columns before applying the groupby and sum operation
                        filtered_df_numeric = filtered_df[[
                            'Site', 'Sales USD']].copy()
                        filtered_df_numeric = filtered_df_numeric.groupby(
                            'Site').sum().reset_index()

                        # Creating a horizontal bar chart
                        # Creating a horizontal bar chart
                        # Creating a horizontal bar chart
                        fig, ax = plt.subplots(figsize=(6, 1.5), dpi=100)
                        ax.barh(filtered_df_numeric['Site'], filtered_df_numeric['Sales USD'], color=[
                                '#AECDE7', '#7BA8CD', '#5480A5'])

                        # Adding text labels inside the bars
                        for index, value in enumerate(filtered_df_numeric['Sales USD']):
                            ax.text(value * 0.95, index, '${:,.0f}'.format(
                                value), va='center', ha='right', color='black', fontsize=6)

                        # Invert y-axis to have the largest value at the top
                        ax.invert_yaxis()

                        # Remove the x-axis labels and ticks
                        ax.xaxis.set_visible(False)

                        # Remove the border on the left and bottom sides
                        sns.despine(ax=ax, left=True, bottom=True)

                        # Display the plot in Streamlit
                        st.pyplot(fig)

                    # Space reduction after the chart
                    st.markdown(
                        '<style>div.block-container{padding-bottom:0px;}</style>',
                        unsafe_allow_html=True
                    )

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
                        st.dataframe(
                            sales_rep_sales_profit_summary(filtered_df))

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
                            return summary_df.head(5)

                        st.markdown(
                            """
                        <h3 style='font-size:20px; color: #164871; border-bottom: 1px solid #164871; padding-bottom: 3px;'>
                            Per Customer Name (Top 5)
                        </h3>
                        """, unsafe_allow_html=True)
                        st.dataframe(
                            customer_sales_profit_summary(filtered_df))

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
                            plt.figure(figsize=(4, 2), dpi=100)
                            ax = sns.barplot(data=grouped_df, x='Month',
                                             y='Sales USD', color="#2E83CA")

                            # Annotate the bars with values inside the bars using the text() function
                            for i, value in enumerate(grouped_df['Sales USD']):
                                ax.text(i, value - (0.05 * max(grouped_df['Sales USD'])), f"{value/1_000_000:.1f}M",
                                        # Adjust text properties here
                                        ha='center', va='top', color='white', fontsize=8, rotation=90)

                            # Hide the y-axis labels and ticks
                            ax.yaxis.set_visible(False)
                            # Adjust the x-axis labels
                            for label in ax.get_xticklabels():
                                label.set_fontsize(8)

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
                            plt.figure(figsize=(4, 2), dpi=100)
                            ax = sns.barplot(data=grouped_df, x='Month',
                                             y='Profit USD', color="#0D4F86")

                            # Annotate the bars with values above the bars using the text() function
                            for i, value in enumerate(grouped_df['Profit USD']):
                                ax.text(i, value - 0.15 * max(grouped_df['Profit USD']),  # Adjust the multiplier as needed
                                        f"{value/1_000_000:.1f}M", ha='center', va='center',
                                        color='white', fontsize=8, rotation=90)

                            # Hide the y-axis labels and ticks
                            ax.yaxis.set_visible(False)
                            # Adjust the x-axis labels
                            for label in ax.get_xticklabels():
                                label.set_fontsize(8)

                            # Remove the x-axis label "Month"
                            ax.set_xlabel('')

                            # Remove the border
                            sns.despine(left=True, bottom=True)
                            st.pyplot(plt)

                        st.markdown(
                            """
                        <h3 style='font-size:20px; color: #164871; border-bottom: 1px solid #164871; padding-bottom: 3px;'>
                            Gross Profit per Month
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
                            plt.figure(figsize=(4, 2), dpi=100)
                            ax = sns.barplot(data=grouped_df, x='Month',
                                             y='Margin %', color="#617DB4")

                            # Annotate the bars with values above the bars using the text() function
                            # Annotate the bars with values inside the bars using the text() function
                            for i, value in enumerate(grouped_df['Margin %']):
                                ax.text(
                                    # Adjusting position to be slightly lower inside the bar
                                    i, value - 0.15 * \
                                    max(grouped_df['Margin %']),
                                    # Adding vertical alignment
                                    f"{value:.1f}%", ha='center', va='center',
                                    # Adjusting for better visibility and rotation
                                    color='white', fontsize=8, rotation=90)

                            # Hide the y-axis labels and ticks
                            ax.yaxis.set_visible(False)
                            # Adjust the x-axis labels
                            for label in ax.get_xticklabels():
                                label.set_fontsize(8)

                            # Remove the x-axis label "Month"
                            ax.set_xlabel('')

                            # Remove the border
                            sns.despine(left=True, bottom=True)
                            st.pyplot(plt)

                        st.markdown(
                            """
                        <h3 style='font-size:20px; color: #164871; border-bottom: 1px solid #164871; padding-bottom: 3px;'>
                            Margin
                        </h3>
                        """, unsafe_allow_html=True)
                        monthly_margin_viz(filtered_df)
            else:
                st.warning("The filtered data is empty. Adjust your filters.")
        else:
            st.warning(
                "Please upload data in the 'Load Updated Data' tab first.")
    # Tab 3: AI Insights
    with tabs[2]:
        if 'data_uploaded' in st.session_state and st.session_state.data_uploaded:
            df = st.session_state.df

        else:
            st.warning(
                "Please upload data in the 'Load Updated Data' tab first.")

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
            numeric_cols = ['Sales USD', 'Profit USD', 'Margin %',
                            'Unit Price USD', 'Total Cost USD', 'Quantity']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
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
                    previous_start_date = pd.to_datetime(previous_start_date)
                    previous_end_date = pd.to_datetime(previous_end_date)

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
                    previous_df = clean_and_convert_to_numeric(previous_df)

                # Summary calculations for current period
                if not filtered_df.empty:
                    total_sales = filtered_df['Sales USD'].sum()
                    total_profit = filtered_df['Profit USD'].sum()
                    avg_margin = filtered_df['Margin %'].mean()

                # Summary calculations for previous period
                if not previous_df.empty:
                    previous_total_sales = previous_df['Sales USD'].sum()
                    previous_total_profit = previous_df['Profit USD'].sum()
                    previous_avg_margin = previous_df['Margin %'].mean()

                    sales_change = calculate_percentage_change(
                        total_sales, previous_total_sales)
                    profit_change = calculate_percentage_change(
                        total_profit, previous_total_profit)
                    margin_change = calculate_percentage_change(
                        avg_margin, previous_avg_margin)

                # Site comparison
                if not filtered_df.empty:
                    site_sales = filtered_df.groupby('Site')['Sales USD'].sum()
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
                    customer_sales = filtered_df.groupby('Customer Name')[
                        'Sales USD'].sum()
                    best_customer = customer_sales.idxmax()  # Best Customer
                    previous_customer_sales = previous_df.groupby(
                        'Customer Name')['Sales USD'].sum() if not previous_df.empty else pd.Series()
                    customer_changes = {cust: calculate_percentage_change(
                        customer_sales[cust], previous_customer_sales.get(cust, 0)) for cust in customer_sales.index}

                    # Ensure all values in customer_changes are numeric
                    customer_changes = {cust: change for cust, change in customer_changes.items(
                    ) if isinstance(change, (int, float))}

                    if customer_changes:
                        most_improved_customer = max(
                            customer_changes, key=customer_changes.get)

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
                    "   - **Most Improved Customer:** {} with an increase of ${:,.0f} USD\n\n".format(most_improved_customer, customer_changes.get(most_improved_customer, 'N/A')) +

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
                    formatted_response = response_content.replace('\n', '\n\n')

                    return formatted_response

                except Exception as e:
                    return f"An error occurred: {e}"
            return ""

        # AI Insights Section
        def display_ai_insights():
            # Check if the data has been uploaded
            if 'df' in st.session_state:
                df = st.session_state.df

                market_segments = df["Market Segment"].unique().tolist()
                sales_rep_names = df["Sales Rep Name"].unique().tolist()
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
                        <h3 style='font-size:20px; color: #164871; border-bottom: 1px solid #164871; padding-bottom: 10px; margin-bottom: 10px;'>
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
                        <h3 style='font-size:20px; color: #164871; border-bottom: 1px solid #164871; padding-bottom: 10px; margin-bottom: 10px;'>
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
                        label='3.Generate AI Insights')

                # Process the form submission
                if submit_button:
                    with st.spinner(text='In progress'):
                        # Hardcoded query
                        user_query = "Analyze the sales trends from this dashboard."

                        ai_response = generate_ai_insights(
                            user_query, start_date, end_date, prev_start_date, prev_end_date,
                            market_segment_1, market_segment_2, sales_rep_1, sales_rep_2, customer_1, customer_2
                        )

                        # Reduce line spacing in the AI Insights output, including bullet points
                        # Display the AI response with custom formatting
                        st.markdown(f"""
                            <div style='line-height:1.3;'>
                                {ai_response}
                            </div>
                            <style>
                                h4 {{
                                    font-size: 4px;
                                    margin-bottom: 2px;
                                }}
                                ul {{
                                    margin-bottom: 1px;
                                }}
                                li {{
                                    line-height: 1.2;
                                    margin-bottom: 1px;
                                }}
                            </style>
                            """, unsafe_allow_html=True)
            else:
                st.warning(
                    "Data not loaded. Please upload data in the 'Load Updated Data' tab first.")
        # Call this function where the AI Insights section is supposed to be
        display_ai_insights()


if __name__ == "__main__":
    main()
