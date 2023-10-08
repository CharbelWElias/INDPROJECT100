import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.subplots as sp
import statsmodels.api as sm
import plotly.express as px
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from numpy.linalg import LinAlgError
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
import time


# Set Page Title and Layout
st.set_page_config(page_title="Financial Report", layout="wide")

# SIDE BAR
# Set Title
st.sidebar.title('Financial Report')

# Navigation buttons that act like sections
selected_page = st.sidebar.radio('Navigate to:', 
                                  ['The Dataset', 'Correlation Analysis', 'Data Exploration', 'Stock Price Prediction', 'End of the Report'])


# Load the CSV file
file_path = "FINANCIAL_DATA_NEW.csv"
try:
    data = pd.read_csv(file_path)
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])
        data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')  
        data['Year'] = pd.to_datetime(data['Date']).dt.year
except Exception as e:
    st.error(f"Error parsing the CSV file: {e}")
    data = pd.DataFrame()

# Navigation Pages (Based on radio button selection)
if selected_page == 'The Dataset':
    st.markdown('# The Dataset')
    st.markdown(
        """
        ### The current report is divided into 4 main parts:  
        - The first part <u>**Data Playroom**</u> will allow you to explore the underlying dataset by selecting different fields and different time frames
        - The second part <u>**Correlation Analysis**</u> will show you how different variables in the dataset are related, which would help in understanding whether those variables tend to move in the same direction or opposite ones
        - The third part <u>**Stock Price Prediction**</u> will enable you to predict by how much the stock price would change (in %) when the other variables that are related to it and affect its movement change *(in % as well)*
            """,
    unsafe_allow_html=True)

    with st.expander("#### The Variables Used: :slot_machine:"):
    # Inside the expander, write the content you want
        st.markdown("""
        All variables are denominated in USD!

        - **BITCOIN**: a cryptocurrency traded online globally
        - **10YIELD**: the yield of the 10-year US treasury bond
        - **GOOGLE**: the stock price of Google
        - **CSI**: consumer Sentiment Index in the US (**This is not in USD; It is an index**)
        - **IBEX35**: the stock index of Spain
        - **NASDAQ**: the ***technology*** stock index of the US
        - **DAX40**: the stock index of Germany
        - **SP500**: the stock index of the US
        - **FTSE100**: the stock index of the UK
        - **EURO**: the price of 1 Euro in USD
        """)

    st.markdown("---")

    st.markdown("""
        ## Data Playroom :table_tennis_paddle_and_ball:  
        Note that you can visualize all the data by selecting "All" years and sliding the bar to the maximum.  
        Enjoy playing with the data before diving into some statistical analysis! :coffee:
    """)

    # Get the unique years from the data
    years = sorted(data['Year'].unique())

    # Add an option for "All" years
    year_options = ['All'] + [str(year) for year in years if 2014 <= year <= 2023]

    # User Select Year
    selected_year = st.selectbox('Select the Year', year_options)
    if selected_year != 'All':
        data = data[data['Year'] == int(selected_year)]

    # User Selects Number of Rows
    num_rows = st.slider("Select the number of rows to display", min_value=1, max_value=len(data), value=5)

    # User Select Columns
    # Exclude 'Year' column from default columns to display
    default_columns = [col for col in data.columns if col != 'Year']
    columns = st.multiselect("Select columns to display", list(data.columns), default=default_columns)

    # Format all the columns to numbers without decimal points except for "BITCOIN" and "EURO"
    for col in data.columns:
        if col not in ["10YIELD", "EURO"]:
            data[col] = data[col].apply(lambda x: f"{x:.0f}" if isinstance(x, (int, float)) else x)

    # If columns are selected, display them, otherwise show error
    if columns:
        # User selects order of sorting
        sort_order = st.selectbox('Sort Order', ['Ascending', 'Descending'])
        ascending = sort_order == 'Ascending'
        
        # User selects column to sort by
        sort_by = st.selectbox('Sort By', columns, index=columns.index('Date') if 'Date' in columns else 0)
        
        # Display the selected data
        st.table(data[columns].sort_values(by=sort_by, ascending=ascending).head(num_rows))
    else:
        st.error("Please select at least one column to display.")
        



##################################################################################################

elif selected_page == 'Correlation Analysis':
    st.title("Correlation Analysis :twisted_rightwards_arrows:")
    st.subheader("***I have used the stock/index return in % for this analysis (and for subsequent sections/pages) and not the stock prices!***")
    st.markdown("##### ***the reason for using the return in % is to study the variations and not the stocks/index levels!***")

    # Define available years based on the data
    available_years = list(range(2014, 2024))

    # Create two columns using st.columns
    col1, col2 = st.columns(2)

    # Perform Correlation Analysis in the First Column

    with col1:
        st.markdown("## Correlation Matrix")
        st.markdown("""select the years to visualize  
            ***For example, select start year to be 2014 and the end year to be 2023:***""")

        # Let the user select the start and end year for the first column
        start_year1 = st.selectbox('Select Start Year', available_years, index=0, key='start_year1')
        end_year1 = st.selectbox('Select End Year', available_years, index=len(available_years) - 1, key='end_year1')

        # Check if the start year is less than or equal to the end year
        if start_year1 <= end_year1:
            data_analyze1 = data[data['Year'].between(start_year1, end_year1)]

            if not data_analyze1.empty:
                # Calculate the stock return for the first column
                columns_to_calculate1 = [col for col in data_analyze1.columns if col not in ['Date', 'Year']]
                return_table1 = data_analyze1[columns_to_calculate1].pct_change()  # Calculate stock return

                # Drop the first row of return_table1, which contains NaN values
                return_table1 = return_table1.iloc[1:]

                # Add Date and Year columns to the return_table1
                return_table1['Date'] = data_analyze1['Date'].iloc[1:]
                return_table1['Year'] = data_analyze1['Year'].iloc[1:]

                # Reorder the columns to have 'Date' as the first column
                ordered_columns1 = ['Date'] + columns_to_calculate1 + ['Year']
                return_table1 = return_table1[ordered_columns1]

                # Calculate the correlation matrix for all variables except 'Date' and 'Year' for the first column
                correlation_matrix1 = return_table1[columns_to_calculate1].corr()

                # Convert the correlation matrix to a 2D array and round it to two decimal places
                corr_matrix_array1 = correlation_matrix1.round(2).to_numpy()

                # Create labels for the plot (column names)
                labels1 = correlation_matrix1.columns.tolist()

                # Create the heatmap using plotly.figure_factory with a diverging colorscale 'Blues'
                heatmap1 = ff.create_annotated_heatmap(corr_matrix_array1, x=labels1, y=labels1, colorscale='Blues', showscale=True)

                # Adjusting the color scale and the figure size
                heatmap1['data'][0]['zmin'] = -1
                heatmap1['data'][0]['zmax'] = 1
                heatmap1.update_layout(title='Correlation Matrix Heatmap', width=600, height=600)

                # Displaying the plot
                st.plotly_chart(heatmap1)

            else:
                st.warning("No data available for the selected year range in Column 1.")
        else:
            st.error("Invalid year range in Column 1. Start year should be less than or equal to end year.")

    # Perform Correlation Analysis in the Second Column
    with col2:
        st.markdown("## Correlation Differential")
        st.markdown("""select a **DIFFERENT** year combination than Correlation Matrix to visualize the results  
            ***For example select start year to be 2019 and end year to be 2023:***""""")
        
        inferred_start_year2 = st.selectbox('Select Start Year', available_years, index=0, key='inferred_start_year2')

        # Let the user select the end year for the second column
        end_year2 = st.selectbox('Select End Year', available_years, index=len(available_years) - 1, key='end_year2')
        
        # Check if the inferred start year is less than or equal to the end year
        if inferred_start_year2 <= end_year2:
            data_analyze2 = data[data['Year'].between(inferred_start_year2, end_year2)]

            if not data_analyze2.empty:
                # Calculate the stock return for the second column
                columns_to_calculate2 = [col for col in data_analyze2.columns if col not in ['Date', 'Year']]
                return_table2 = data_analyze2[columns_to_calculate2].pct_change()  # Calculate stock return

                # Calculate the correlation matrix for the second column
                correlation_matrix2 = return_table2[columns_to_calculate2].corr()

                # Calculate the correlation differential matrix
                correlation_diff_matrix = correlation_matrix2 - correlation_matrix1

                # Create labels for the plot (column names)
                labels = correlation_diff_matrix.columns.tolist()

                # Create a mask for significant differences
                mask_significant = np.abs(correlation_diff_matrix) > 0.1

                # Apply the mask to the correlation_diff_matrix and replace insignificant values with np.nan
                z = correlation_diff_matrix.where(mask_significant, np.nan).round(2).to_numpy()

                # Create the heatmap using plotly.figure_factory with a diverging colorscale
                heatmap_diff = ff.create_annotated_heatmap(z, x=labels, y=labels, colorscale='hot', showscale=True)

                # Update annotations to avoid displaying 'nan' and to ensure two decimal places for other values
                for i in range(len(heatmap_diff.layout.annotations)):
                    text = heatmap_diff.layout.annotations[i].text
                    if text == 'nan':
                        heatmap_diff.layout.annotations[i].text = ''
                    else:
                        heatmap_diff.layout.annotations[i].text = f'{float(text):.2f}'

                # Adjusting the color scale and the figure size
                heatmap_diff['data'][0]['zmin'] = -1
                heatmap_diff['data'][0]['zmax'] = 1
                heatmap_diff.update_layout(title='Correlation Differential Matrix (for correlation changes > 0.1 or < -0.1)', width=600, height=600)

                # Displaying the plot
                st.plotly_chart(heatmap_diff)

            else:
                st.warning("No data available for the selected year range in Column 2.")
        else:
            st.error("Invalid year range in Column 2. Start year should be less than or equal to end year.")

        

    st.markdown("---")

    st.markdown("## Moving Average Correlation Analysis")
    st.markdown("##### ***Visualize the change in the 12-month trailing average correlation over time from 2016 to 2023***")

    # Ensure that 'Date' is in datetime format
    data['Date'] = pd.to_datetime(data['Date'])

    # Compute 12-month moving average correlation
    moving_avg_corr = {}
    columns_to_calculate = [col for col in data.columns if col not in ['Date', 'Year']]
    for i in range(len(columns_to_calculate)):
        for j in range(i + 1, len(columns_to_calculate)):
            col1 = columns_to_calculate[i]
            col2 = columns_to_calculate[j]
            pair_data = data[[col1, col2, 'Date']].dropna()
            pair_data = pair_data.set_index('Date')  # Set 'Date' as index
            rolling_corr = pair_data[col1].rolling(window=12).corr(pair_data[col2])
            moving_avg_corr[f"{col1} and {col2}"] = rolling_corr

    # Convert the moving average correlation data to a DataFrame
    moving_avg_corr_df = pd.DataFrame(moving_avg_corr).dropna()

    # Filter the moving_avg_corr_df to include data only from 2016 to 2023
    start_date = pd.Timestamp(year=2016, month=1, day=1)
    end_date = pd.Timestamp(year=2023, month=12, day=31)
    moving_avg_corr_df = moving_avg_corr_df[(moving_avg_corr_df.index >= start_date) & (moving_avg_corr_df.index <= end_date)]

    # Allow the user to select two pairs of correlations
    selected_pair1 = st.selectbox('Select the first pair of correlations', list(moving_avg_corr_df.columns), key='pair1')
    selected_pair2 = st.selectbox('Select the second pair of correlations', list(moving_avg_corr_df.columns), key='pair2')

    # Create a Plotly figure
    fig = go.Figure()

    # Add traces to the figure (line plots for selected pairs)
    fig.add_trace(go.Scatter(x=moving_avg_corr_df.index, 
                             y=moving_avg_corr_df[selected_pair1], 
                             mode='lines',
                             name=selected_pair1))

    fig.add_trace(go.Scatter(x=moving_avg_corr_df.index, 
                             y=moving_avg_corr_df[selected_pair2], 
                             mode='lines',
                             name=selected_pair2))

    # Update the layout to place the legend above the chart and expand the chart
    fig.update_layout(
        autosize=True,
        margin=dict(l=0, r=0, b=0, t=40),  # Added top margin for the legend
        legend=dict(y=1.1, orientation='h'),  # Positioning the legend above the chart
        template="plotly"
    )

    # Display the chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)



    st.markdown('---')



    # Adding a new line chart for price visualization
    st.markdown("## Price Visualization")
    st.markdown("##### ***Select the stock of interest and visualize how the movement of the stock affected the correlation results above***")

    # Give the user the option to choose which column to visualize
    columns_to_visualize = [col for col in data.columns if col not in ['Date', 'Year']]
    selected_column = st.selectbox('Select a column to visualize', columns_to_visualize)

    # Filter the DataFrame to include data only from 2016 to 2023
    start_date = pd.Timestamp(year=2016, month=1, day=1)
    end_date = pd.Timestamp(year=2023, month=12, day=31)
    filtered_data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]

    # Create a line chart for the selected column
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filtered_data['Date'], y=filtered_data[selected_column], mode='lines', name=selected_column))

    # Update the layout to expand the chart
    fig.update_layout(
        autosize=True,
        margin=dict(l=0, r=0, b=0, t=0),
        template="plotly",
        xaxis_title='Date',
        yaxis_title='Price',
    )

    # Display the chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)



    st.markdown('---')


##################################################################################################


# Detailed Financial Data Section
#st.markdown("## Detailed Financial Data")
#data_T = data.set_index("Date").transpose() if 'Date' in data.columns else data.transpose()
#st.table(data_T)


# Visualization Section
elif selected_page == 'Data Exploration':
    st.title("Data Exploration :female-detective:")


    columns_to_calculate_pct = [col for col in data.columns if col not in ['Date', 'Year']]
    pct_change_data = data[columns_to_calculate_pct].pct_change()
    pct_change_data['Date'] = data['Date']
    pct_change_data['Year'] = data['Year']

    pct_change_data['Date'] = pd.to_datetime(pct_change_data['Date'])

    start_date_pct = pd.Timestamp(year=2014, month=1, day=1)
    end_date_pct = pd.Timestamp(year=2024, month=12, day=31)

    filtered_pct_data = pct_change_data[(pct_change_data['Date'] >= start_date_pct) & (pct_change_data['Date'] <= end_date_pct)]




    st.markdown('## Normality Study')
    with st.expander("#### How were variables classified?"):
        st.markdown("""
            - Usually, for a distribution to be considered normal:
                - 68% of the data should be within 1 standard deviation away from the mean
                - 95% should be within 2 standard deviation away from the mean
                - 99.7% should be within 3 standard deviation away from the mean
            - In this exercise, I considered a data to be nearly-normal if:
                - 90% of the data is within 2 standard deviation away from the mean
                - 98% within 3 standard deviation away from the mean
            - Otherwise, distributions are not normally distributed""")

    data_numeric = pct_change_data.drop(columns=['Date', 'Year'], errors='ignore')

    z_scores = (data_numeric - data_numeric.mean()) / data_numeric.std()

    ranges = {
        'mean - 3 std': (-3, -2),
        'mean - 2 std': (-2, -1),
        'mean ± 1 std': (-1, 1.01),
        'mean + 2 std': (1.01, 2.01),
        'mean + 3 std': (2.01, 3.01),
    }

    distribution_df = pd.DataFrame(index=data_numeric.columns, columns=ranges.keys())

    for column in z_scores.columns:
        for range_name, (lower, upper) in ranges.items():
            count_in_range = z_scores[column].apply(lambda x: lower <= x < upper).sum()
            percentage_in_range = (count_in_range / len(z_scores)) * 100
            distribution_df.loc[column, range_name] = f"{percentage_in_range:.2f}%"

    distribution_df_numeric = distribution_df.apply(lambda x: x.str.rstrip('%').astype('float'), axis=1)

    # Calculate % Volatility (stdev) for each column and add it to the DataFrame
    volatility = data_numeric.std() * 100
    distribution_df['% Volatility (stdev)'] = volatility.map(lambda x: f"{x:.2f}%")

    distribution_df['% of data within 3 stdev'] = distribution_df_numeric.sum(axis=1).map(lambda x: f"{x:.2f}%")

    if 'Normal Distribution' in distribution_df.columns:
        cols = distribution_df.columns.tolist()
        cols = cols[:-2] + [cols[-1]] + [cols[-2]]
        distribution_df = distribution_df[cols]

    for index, row in distribution_df_numeric.iterrows():
        within_1_std = row['mean ± 1 std']
        within_2_std = row['mean - 2 std'] + row['mean ± 1 std'] + row['mean + 2 std']
        within_3_std = row.sum()
        
        if (row['mean - 3 std'] + row['mean - 2 std']) - (row['mean + 2 std'] + row['mean + 3 std']) > 3:
            skewness = ' - Right Skewed'
        elif (row['mean + 3 std'] + row['mean + 2 std']) - (row['mean - 2 std'] + row['mean - 3 std']) > 3:
            skewness = ' - Left Skewed'
        else:
            skewness = ''
        
        if within_1_std >= 68 and within_2_std >= 94.5 and within_3_std >= 99:
            distribution_df.at[index, 'Normal Distribution'] = 'Normally Distributed' + skewness
        elif within_2_std >= 90 and within_3_std >= 98:
            distribution_df.at[index, 'Normal Distribution'] = 'Near-Normal' + skewness
        else:
            distribution_df.at[index, 'Normal Distribution'] = 'NON-Normal' + skewness

    # Sort the DataFrame by % Volatility (stdev) in descending order
    distribution_df['% Volatility (stdev)'] = distribution_df['% Volatility (stdev)'].str.rstrip('%').astype('float')
    distribution_df = distribution_df.sort_values(by='% Volatility (stdev)', ascending=False)
    distribution_df['% Volatility (stdev)'] = distribution_df['% Volatility (stdev)'].map(lambda x: f"{x:.2f}%")

    st.table(distribution_df)



    st.markdown("---")

    # Visualization of the distributions
    st.markdown("## Visualize the Distributions")

    st.markdown("##### ***See it for yourself! :male-scientist:***")

    # Create two columns
    col1, col2 = st.columns(2)

    # Select column to visualize return variation in the first column
    selected_column_pct1 = col1.selectbox('Select a column to visualize the return variation', distribution_df.index.tolist(), key='selectbox1')


    # Extracting the selected column from the original dataframe for the first column
    selected_data1 = filtered_pct_data[selected_column_pct1].dropna()

    # Calculate mean and standard deviation of the selected data for the first column
    mean1 = selected_data1.mean()
    std1 = selected_data1.std()

    # Create a density plot for the selected column's return variation in the first column
    plt.figure(figsize=(10, 6))
    sns.kdeplot(selected_data1, fill=True)
    plt.title(f'Return Variation Density Plot for {selected_column_pct1}', y=1.05)
    plt.xlabel('% change in price / index level')
    plt.ylabel('Density')

    # Add vertical lines and labels representing the standard deviations from the mean for the first column
    for i in range(-3, 4):
        if i == 0:
            label = 'mean'
        else:
            label = f'mean {i:+d} std'
        plt.axvline(x=mean1 + i * std1, color='r', linestyle='--')
        plt.text(mean1 + i * std1, plt.ylim()[1] * 0.9, label, rotation=45, color='r', ha='right')

    # Display the density plot in the first column
    col1.pyplot(plt.gcf())

    # Select column to visualize return variation in the second column
    selected_column_pct2 = col2.selectbox('Select a column to visualize the return variation', distribution_df.index.tolist(), key='selectbox2')

    # Extracting the selected column from the original dataframe for the second column
    selected_data2 = filtered_pct_data[selected_column_pct2].dropna()

    # Calculate mean and standard deviation of the selected data for the second column
    mean2 = selected_data2.mean()
    std2 = selected_data2.std()

    # Create a density plot for the selected column's return variation in the second column
    plt.figure(figsize=(10, 6))
    sns.kdeplot(selected_data2, fill=True)
    plt.title(f'Return Variation Density Plot for {selected_column_pct2}', y=1.05)
    plt.xlabel('% change in price / index level')
    plt.ylabel('Density')

    # Add vertical lines and labels representing the standard deviations from the mean for the second column
    for i in range(-3, 4):
        if i == 0:
            label = 'mean'
        else:
            label = f'mean {i:+d} std'
        plt.axvline(x=mean2 + i * std2, color='r', linestyle='--')
        plt.text(mean2 + i * std2, plt.ylim()[1] * 0.9, label, rotation=45, color='r', ha='right')

    # Display the density plot in the second column
    col2.pyplot(plt.gcf())



    st.markdown('---')

    st.markdown("## Number of Outliers - per Variable")
    with st.expander("#### Explanation of the below table:"):
        st.markdown("""
        Here I am showing at what cut-offs (in % change):

        - The lower 25% of the data lies (for Q1)
        - The upper 25% of the data lies (for Q3)
        - The lower limit for the outliers (below which the % change becomes outlier)
        - The upper limit for the outliers (above which the % change becomes outlier)
        """)

    # Filter the dataframe for the given years
    pct_change_data = pct_change_data[(pct_change_data['Date'] >= '2014-01-01') & (pct_change_data['Date'] <= '2023-12-31')]

    # Exclude 'Date' and 'Year' columns for the IQR calculations
    columns = [col for col in pct_change_data.columns if col not in ['Date', 'Year']]

    # Initialize a DataFrame to store the IQR statistics and outlier counts
    iqr_stats_df = pd.DataFrame(columns=['Variable', 'Q1', 'Q3', 'Lower Outlier Limit', 'Upper Outlier Limit', 'Lower Outliers Count', 'Upper Outliers Count', 'Total Outliers Count'])

    for col in columns:
        Q1_value = pct_change_data[col].quantile(0.25)
        Q3_value = pct_change_data[col].quantile(0.75)
        IQR = Q3_value - Q1_value
        
        lower_outlier_limit_value = Q1_value - 1.5 * IQR
        upper_outlier_limit_value = Q3_value + 1.5 * IQR
        
        lower_outliers_count = pct_change_data[pct_change_data[col] < lower_outlier_limit_value].shape[0]
        upper_outliers_count = pct_change_data[pct_change_data[col] > upper_outlier_limit_value].shape[0]
        
        total_outliers_count = lower_outliers_count + upper_outliers_count
        
        # Format the values as percentages for display
        Q1 = f"{Q1_value:.2%}"
        Q3 = f"{Q3_value:.2%}"
        lower_outlier_limit = f"{lower_outlier_limit_value:.2%}"
        upper_outlier_limit = f"{upper_outlier_limit_value:.2%}"
        
        new_row = pd.DataFrame({
            'Variable': [col],
            'Q1': [Q1],
            'Q3': [Q3],
            'Lower Outlier Limit': [lower_outlier_limit],
            'Upper Outlier Limit': [upper_outlier_limit],
            'Lower Outliers Count': [lower_outliers_count],
            'Upper Outliers Count': [upper_outliers_count],
            'Total Outliers Count': [total_outliers_count]
        })
        iqr_stats_df = pd.concat([iqr_stats_df, new_row], ignore_index=True)

    # Display the IQR statistics and outliers DataFrame
    st.table(iqr_stats_df)




    st.markdown('---')

    data['Date'] = pd.to_datetime(data['Date'])

    # Visualization
    st.markdown("## Outliers Visualization")
    st.markdown("##### ***Visualize the locations of those outliers on the price charts of each variable:***")

    # Dropdown to select the variable to study
    selected_variable = st.selectbox('Select a variable to study', columns)

    # Set the desired height for the chart
    chart_height = 600  # for example, 600 pixels

    # Create a Plotly line chart with the specified height
    fig = px.line(data, x='Date', y=selected_variable, labels={'Price': selected_variable}, height=chart_height)

    # Adding marks for the outliers
    lower_limit = iqr_stats_df.loc[iqr_stats_df['Variable'] == selected_variable, 'Lower Outlier Limit'].values[0]
    upper_limit = iqr_stats_df.loc[iqr_stats_df['Variable'] == selected_variable, 'Upper Outlier Limit'].values[0]

    # Convert formatted string limits back to float for comparison
    lower_limit = float(lower_limit.strip('%')) / 100
    upper_limit = float(upper_limit.strip('%')) / 100

    outliers_df = pct_change_data[(pct_change_data[selected_variable] < lower_limit) | (pct_change_data[selected_variable] > upper_limit)]

    # Create a scatter plot for outliers and then update the marker size
    scatter_fig = px.scatter(data, x=outliers_df['Date'], y=data.loc[outliers_df.index, selected_variable], color_discrete_sequence=['red'])
    scatter_fig.update_traces(marker=dict(size=10))  # adjust the size as needed
    fig.add_trace(scatter_fig.data[0])

    fig.update_layout(title=f"{selected_variable} Price with Outliers", xaxis_title='Year', yaxis_title='Index')

    # Display the Plotly chart in Streamlit and set use_container_width to True
    st.plotly_chart(fig, use_container_width=True)

########################################################################################################


elif selected_page == 'Stock Price Prediction':
    st.title("Stock Price Prediction :crystal_ball:")
    st.markdown("""
    ##### In this section you can build your own linear regression model!

    - Select the Confidence Levels
    - Correlation thresholds to filter out high correlation variables
    - The timeframe for the study
    """)

    columns_to_calculate_pct = [col for col in data.columns if col not in ['Date', 'Year']]
    pct_change_data = data[columns_to_calculate_pct].pct_change()
    pct_change_data['Date'] = data['Date']
    pct_change_data['Year'] = data['Year']

    pct_change_data['Date'] = pd.to_datetime(pct_change_data['Date'])

    confidence_interval = st.slider("Select Confidence Interval:", min_value=90, max_value=99, value=95)
    threshold_in = threshold_out = 1 - confidence_interval / 100
    correlation_threshold = st.slider("Select Correlation Threshold:", min_value=0.5, max_value=1.0, value=0.6, step=0.05)
    training_size = st.slider("Select Training %:", min_value=0.8, max_value=0.95, value=0.9, step=0.05)

    start_year = st.slider("Select Start Year:", min_value=2013, max_value=2021, value=2014)
    end_year = st.slider("Select End Year:", min_value=2016, max_value=2024, value=2024)

    if end_year - start_year < 3:
        st.warning("The analysis may not be effective with less than three years range.")

    # Filtering data based on the selected year range
    start_date_pct = pd.Timestamp(year=start_year, month=1, day=1)
    end_date_pct = pd.Timestamp(year=end_year, month=12, day=31)

    filtered_pct_data = pct_change_data[(pct_change_data['Date'] >= start_date_pct) & (pct_change_data['Date'] <= end_date_pct)]

    target_variable = st.selectbox("Select Target Variable:", ["SP500", "IBEX35", "FTSE100", "BITCOIN", "GOOGLE"])
    filtered_pct_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    filtered_pct_data.fillna(filtered_pct_data.mean(), inplace=True)

    X = filtered_pct_data.drop(columns=[target_variable, 'Date', 'Year'])
    y = filtered_pct_data[target_variable]

    X_all = sm.add_constant(X)

    testing_size = 1 - training_size
    X_train_all, X_test_all, y_train, y_test = train_test_split(X_all, y, test_size=testing_size, random_state=42)

    model_all = sm.OLS(y_train, X_train_all).fit()
    st.markdown("## Model Summary (All Variables):")
    st.markdown("##### This is not the final output yet! ***For the final one check the next section!***")
    st.markdown("###### Check all variables tested, if they are relevant or not, and their contribution:")
    summary_texts = model_all.summary().tables[0].as_text() + "\n" + model_all.summary().tables[1].as_text()
    st.text(summary_texts)

    st.markdown('---')

    def calculate_vif(X):
        vif_data = pd.DataFrame()
        vif_data["Variable"] = X.columns
        try:
            vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        except LinAlgError:
            # Handle the singular matrix case
            vif_data["VIF"] = [float('inf')] * X.shape[1]
        
        return vif_data if not vif_data.empty else pd.DataFrame(columns=["Variable", "VIF"])

    # Check for correlation between independent variables
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]
    X = X.drop(X[to_drop], axis=1)

    alpha = 1 - confidence_interval

    def stepwise_selection(X, y, initial_list=[], threshold_in=alpha, threshold_out=alpha, correlation_threshold=correlation_threshold, verbose=True):
        included = list(initial_list)
        while True:
            changed = False
            
            # Forward step
            excluded = list(set(X.columns) - set(included))
            new_pval = pd.Series(index=excluded)
            for new_column in excluded:
                model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included + [new_column]]))).fit()
                new_pval[new_column] = model.pvalues[new_column]
                if verbose:
                    print(f'Testing {new_column}, p-value: {model.pvalues[new_column]}')  # Print the p-value of the variable being tested
                
            best_pval = new_pval.min()
            if best_pval < threshold_in:
                best_feature = new_pval.idxmin()
                included.append(best_feature)
                changed = True
                if verbose:
                    print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))
            
            # Backward step
            if len(included) > 0:
                model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
                pvalues = model.pvalues.iloc[1:]
                worst_pval = pvalues.max()
                if worst_pval > threshold_out:
                    changed = True
                    worst_feature = pvalues.idxmax()
                    included.remove(worst_feature)
                    if verbose:
                        print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
            
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
            for variable in included:
                if model.pvalues[variable] > threshold_out:
                    included.remove(variable)
                    changed = True
                    if verbose:
                        print('Drop {:30} with adjusted p-value {:.6}'.format(variable, model.pvalues[variable]))

            if not changed:
                break
        return included



    significant_vars = stepwise_selection(X, y, threshold_in=threshold_in, threshold_out=threshold_out, correlation_threshold=correlation_threshold)

    if not significant_vars:
        st.error("No significant variables were selected by the stepwise selection method.")
    else:
        X_significant = X_train_all[significant_vars + ['const']]
        X_test_significant = X_test_all[significant_vars + ['const']]

        model_significant = sm.OLS(y_train, X_significant).fit()
        st.markdown("## Model Summary (Significant Variables):")
        st.markdown("##### Now, this is the final model output! :white_check_mark:")
        st.markdown("###### This model output was obtained after running the stepwise method that filters out all irrelevant variables")
        summary_text = model_significant.summary().tables[0].as_text() + "\n" + model_significant.summary().tables[1].as_text()
        st.text(summary_text)

        # Displaying the regression formula
        equation = f"{target_variable} = "
        equation += " + ".join([f"{coef:.4f} * {var}" for var, coef in model_significant.params.items()])
        
        st.markdown("## Final Regression Equation:")
        st.latex(equation)

        st.markdown("---")

        # User input for prediction
        st.markdown("## Input Percentage Change for Significant Variables:")
        st.markdown("##### ***Check how much the target variable would change the next month, if the significant variables change:***")
        user_inputs = {}
        for var in significant_vars:
            user_input = st.number_input(f"Enter percentage change for {var} (%):", value=0.0)
            user_inputs[var] = user_input / 100  # Convert the percentage to a fraction

        user_inputs['const'] = 1  # add the constant
        user_inputs_df = pd.DataFrame([user_inputs])
        prediction = model_significant.predict(user_inputs_df)

        st.markdown("## Prediction:")
        st.write(f"Expected change in {target_variable} (%): {prediction[0]*100:.2f}")


########################################################################################################


elif selected_page == "End of the Report":
    st.title("You have reached the end of the report :airplane_arriving:")
    time.sleep(3)
    st.balloons()
    st.balloons()
    st.balloons()
    st.balloons()
    st.balloons()

    image_url = "https://media.giphy.com/media/hc0qyUylKYz2N8MFSd/giphy.gif?cid=ecf05e47xgb1pkqliiwv072tebkhz1iglafny7pq36gn5qzt&ep=v1_gifs_search&rid=giphy.gif&ct=g"
    st.image(image_url, caption='Your caption if needed', use_column_width=True)








