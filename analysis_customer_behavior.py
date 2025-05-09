import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import Counter # For counting item occurrences

# --- Display Settings for Pandas ---
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.2f}'.format) # Format floats to 2 decimal places

# --- File Paths ---
# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the project root directory (one level up from 'scripts')
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
VISUALIZATIONS_DIR = os.path.join(PROJECT_ROOT, 'visualizations')

# Create the visualizations directory if it doesn't already exist
if not os.path.exists(VISUALIZATIONS_DIR):
    os.makedirs(VISUALIZATIONS_DIR)
    print(f"Created directory: {VISUALIZATIONS_DIR}")

# --- 1. Load and Clean Data ---
def load_and_clean_data(file_name="order_history_kaggle_data.csv"):
    """
    Loads the dataset, performs initial cleaning, and prepares it for analysis.
    - Handles file loading errors.
    - Removes duplicate orders based on 'Order ID'.
    - Converts 'Order Placed At' to datetime objects.
    - Extracts time-based features (hour, day of week, date).
    - Fills missing 'Items in order' with 'Unknown'.
    - Filters for 'Delivered' orders for most behavioral analyses.
    """
    print("--- 1. Starting Data Loading and Cleaning ---")
    data_path = os.path.join(DATA_DIR, file_name)

    try:
        df = pd.read_csv(data_path)
        print(f"Successfully loaded data from {data_path}")
    except FileNotFoundError:
        print(f"Error: The file {file_name} was not found in {DATA_DIR}")
        print("Please ensure the CSV file is in the 'data' directory and try again.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
        return None

    # Initial data overview
    print("\nInitial data information:")
    df.info() # Provides a concise summary of the DataFrame

    # Remove duplicate orders: assuming 'Order ID' should be unique for each order.
    # Keep the first occurrence if duplicates exist.
    initial_rows = len(df)
    df.drop_duplicates(subset=['Order ID'], keep='first', inplace=True)
    print(f"\nRemoved {initial_rows - len(df)} duplicate rows based on 'Order ID'.")
    print(f"Number of rows after duplicate removal: {len(df)}")

    # Convert 'Order Placed At' to datetime.
    # The format '%I:%M %p, %B %d %Y' matches "11:38 PM, September 10 2024"
    # `errors='coerce'` will turn unparseable dates into NaT (Not a Time)
    df['Order Placed At'] = pd.to_datetime(df['Order Placed At'], format='%I:%M %p, %B %d %Y', errors='coerce')

    # Drop rows where date conversion failed (NaT values) as they are crucial for time-based analysis
    if df['Order Placed At'].isnull().any():
        print(f"\nWarning: Found {df['Order Placed At'].isnull().sum()} rows with unparseable dates. These rows will be removed.")
        df.dropna(subset=['Order Placed At'], inplace=True)

    # Feature Engineering: Extract useful date/time components
    df['Order Hour'] = df['Order Placed At'].dt.hour
    df['Order Day of Week Name'] = df['Order Placed At'].dt.day_name()
    df['Order Date'] = df['Order Placed At'].dt.date # For daily trend analysis

    # Handle missing values in 'Items in order' before filtering.
    # This is important if we want to analyze cancellation reasons for orders that had items.
    df['Items in order'].fillna('Unknown', inplace=True)

    # Filter for 'Delivered' orders for behavioral analysis of successful transactions.
    # We create a copy to avoid SettingWithCopyWarning if we modify it later.
    df_delivered = df[df['Order Status'] == 'Delivered'].copy()
    print(f"\nNumber of 'Delivered' orders: {len(df_delivered)}")

    if df_delivered.empty:
        print("Warning: No 'Delivered' orders found. Subsequent analyses might be affected or empty.")
        # Return the original df if no delivered orders, maybe for analyzing other statuses.
        # However, for customer *behavior* focus, delivered orders are key.
        return df_delivered # Or decide if you want to return original `df`

    print("--- Finished Data Loading and Cleaning ---")
    return df_delivered


# --- 2. Customer Behavior Analysis ---
def analyze_customer_behavior(df):
    """
    Analyzes customer behavior patterns from the cleaned (delivered) orders data.
    - Identifies top customers by order count.
    - Visualizes order distribution by hour and day of the week.
    - Parses and visualizes the most popular ordered items.
    - Shows daily order trends.
    - Includes a placeholder for payment method analysis (as data is not available).
    """
    if df is None or df.empty:
        print("No data available for customer behavior analysis (DataFrame is None or empty).")
        return

    print("\n--- 2. Starting Customer Behavior Analysis ---")

    # 2.1. Who is the top customer (most orders)?
    if 'Customer ID' in df.columns:
        # Using value_counts() is efficient for this
        most_frequent_customers = df['Customer ID'].value_counts()
        if not most_frequent_customers.empty:
            top_customer_id = most_frequent_customers.index[0]
            top_customer_orders = most_frequent_customers.iloc[0]
            print(f"\nTop customer (by ID): {top_customer_id} with {top_customer_orders} orders.")

            # Display top 10 customers
            top_10_customers = most_frequent_customers.nlargest(10)
            print("\nTop 10 Customers by Number of Orders:")
            print(top_10_customers)

            plt.figure(figsize=(12, 7))
            top_10_customers.plot(kind='bar', color=sns.color_palette("viridis", len(top_10_customers)))
            plt.title('Top 10 Customers by Number of Orders')
            plt.xlabel('Customer ID')
            plt.ylabel('Number of Orders')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout() # Adjust plot to prevent labels from overlapping
            plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'top_10_customers_by_orders.png'))
            plt.close() # Close the plot to free up memory
            print("Saved plot: top_10_customers_by_orders.png")
        else:
            print("Could not determine top customers (no customer data or orders).")
    else:
        print("Column 'Customer ID' not found. Cannot analyze top customers.")


    # 2.2. What are the peak ordering times (hour of day / day of week)?
    # Orders by Hour of Day
    if 'Order Hour' in df.columns:
        plt.figure(figsize=(12, 6))
        sns.countplot(data=df, x='Order Hour', palette='coolwarm', hue='Order Hour', legend=False)
        plt.title('Order Distribution by Hour of Day')
        plt.xlabel('Hour of Day (0-23)')
        plt.ylabel('Number of Orders')
        plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'orders_by_hour_of_day.png'))
        plt.close()
        print("Saved plot: orders_by_hour_of_day.png")

    # Orders by Day of Week
    if 'Order Day of Week Name' in df.columns:
        # Define the order of days for a more logical plot
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        plt.figure(figsize=(12, 6))
        sns.countplot(data=df, x='Order Day of Week Name', order=day_order, palette='Spectral', hue='Order Day of Week Name', legend=False)
        plt.title('Order Distribution by Day of Week')
        plt.xlabel('Day of Week')
        plt.ylabel('Number of Orders')
        plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'orders_by_day_of_week.png'))
        plt.close()
        print("Saved plot: orders_by_day_of_week.png")


    # 2.3. What are the most ordered meals/items?
    # This requires parsing the 'Items in order' column.
    if 'Items in order' in df.columns:
        print("\nAnalyzing most ordered items...")
        all_items_list = []
        # Iterate through each order's item string
        for item_string in df['Items in order'].dropna(): # dropna just in case, though we filled earlier
            if item_string == 'Unknown': # Skip entries we marked as 'Unknown'
                continue
            
            # Items in an order are typically comma-separated, e.g., "1 x Pizza, 2 x Coke"
            # We need to extract the item name, ignoring the quantity for this analysis.
            try:
                individual_items_in_order = item_string.split(',')
                for item_entry in individual_items_in_order:
                    item_entry_cleaned = item_entry.strip()
                    if ' x ' in item_entry_cleaned:
                        # Extract item name after "N x " part
                        item_name = item_entry_cleaned.split(' x ', 1)[1].strip()
                        all_items_list.append(item_name)
                    elif item_entry_cleaned: # If no " x " but string is not empty, take as is
                        all_items_list.append(item_entry_cleaned) # This might catch malformed entries
            except Exception as e:
                # This broad exception catch is for robustness if string format is very inconsistent.
                print(f"Warning: Could not parse item string: '{item_string}'. Error: {e}")
                continue # Skip to the next item string

        if all_items_list:
            item_counts = Counter(all_items_list)
            top_n_items = 15 # Number of top items to display
            most_common_items = item_counts.most_common(top_n_items)

            print(f"\nTop {top_n_items} Most Ordered Items:")
            for item, count in most_common_items:
                print(f"- {item}: {count} times")

            # Bar Chart: Number of orders per item
            top_items_df = pd.DataFrame(most_common_items, columns=['Item', 'Count'])
            plt.figure(figsize=(12, 8)) # Adjusted figure size for better readability
            sns.barplot(data=top_items_df, y='Item', x='Count', palette='rocket', hue='Item', dodge=False, legend=False)
            plt.title(f'Top {top_n_items} Most Ordered Items')
            plt.xlabel('Number of Times Ordered')
            plt.ylabel('Item Name')
            plt.tight_layout()
            plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'top_ordered_items.png'))
            plt.close()
            print("Saved plot: top_ordered_items.png")
        else:
            print("No items found after parsing 'Items in order' column, or all were 'Unknown'.")
    else:
        print("Column 'Items in order' not found. Cannot analyze popular items.")


    # 2.4. Line Plot: Orders over time (daily trend)
    if 'Order Date' in df.columns:
        # Group by 'Order Date' and count the number of unique 'Order ID's
        daily_orders = df.groupby('Order Date')['Order ID'].count()
        plt.figure(figsize=(15, 7))
        daily_orders.plot(kind='line', marker='.', linestyle='-', linewidth=1) # Subtle marker
        plt.title('Trend of Daily Orders')
        plt.xlabel('Date')
        plt.ylabel('Number of Orders')
        plt.grid(True, linestyle='--', alpha=0.7) # Add a light grid for readability
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'daily_orders_trend.png'))
        plt.close()
        print("Saved plot: daily_orders_trend.png")


    # 2.5. Pie Chart: Payment Methods (Cash vs Online) - Data NOT AVAILABLE in current dataset
    print("\n--- Payment Method Analysis ---")
    print("Note: The current dataset does not have a direct column for payment methods (e.g., 'Cash', 'Online').")
    print("Therefore, a pie chart for payment methods cannot be generated with the available data.")
    print("If such a column (e.g., 'Payment_Method') were available, the analysis would look like this (example code commented out):")
    #
    # payment_method_column_name = 'Payment_Method' # Example column name
    # if payment_method_column_name in df.columns:
    #     payment_counts = df[payment_method_column_name].value_counts()
    #     if not payment_counts.empty:
    #         plt.figure(figsize=(8, 8))
    #         plt.pie(payment_counts, labels=payment_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
    #         plt.title('Payment Method Distribution')
    #         plt.axis('equal') # Ensures the pie chart is circular
    #         plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'payment_methods_pie_chart.png'))
    #         plt.close()
    #         print("Saved (example) plot: payment_methods_pie_chart.png")
    #     else:
    #         print(f"No data found in example column '{payment_method_column_name}'.")
    # else:
    #     print(f"Example column '{payment_method_column_name}' for payment methods not found.")

    print("\n--- Finished Customer Behavior Analysis ---")


# --- 3. Insights and Recommendations ---
def generate_insights_and_recommendations(df):
    """
    Generates simple insights and recommendations based on the analysis.
    This is a starting point; deeper insights would require more context or complex analysis.
    """
    if df is None or df.empty:
        print("No data available to generate insights (DataFrame is None or empty).")
        return

    print("\n--- 3. Insights and Recommendations (Based on Delivered Orders) ---")

    # Insights from peak order times
    if 'Order Day of Week Name' in df.columns:
        order_counts_by_day = df['Order Day of Week Name'].value_counts()
        if not order_counts_by_day.empty:
            busiest_day = order_counts_by_day.idxmax()
            slowest_day = order_counts_by_day.idxmin()
            print(f"\n- Peak Day: The busiest day for orders is {busiest_day}. Consider ensuring adequate staffing (kitchen, delivery) on this day.")
            print(f"- Off-Peak Day: The slowest day for orders is {slowest_day}. Promotions or special offers on {slowest_day} could help increase order volume.")

            # Specific recommendation for Fridays (as requested)
            if 'Friday' in order_counts_by_day.index:
                friday_orders = order_counts_by_day.loc['Friday']
                avg_orders_per_day = order_counts_by_day.mean()
                if friday_orders < avg_orders_per_day * 0.9: # If Friday is significantly slower
                    print("- Recommendation for Friday: Since Friday orders seem lower than average, consider targeted promotions to boost activity on Fridays.")
                elif friday_orders > avg_orders_per_day * 1.1: # If Friday is busier
                     print("- Insight for Friday: Friday is a relatively busy day. Ensure operational readiness. Loyalty offers could be effective.")
                else:
                    print("- Insight for Friday: Friday order volume is around average. Standard operations are likely appropriate, but seasonal promotions could be tested.")
        else:
            print("- Could not determine peak/off-peak days (no data on order counts per day).")

    # Recommendation regarding payment methods (theoretical, as data is missing)
    print("\n- Regarding Payment Methods:")
    print("  - Data on payment methods (Cash, Online, etc.) is crucial for understanding customer preferences and potential friction points.")
    print("  - Recommendation: We strongly recommend collecting and including payment method data in future datasets.")
    print("  - If future data shows low adoption of online payments, consider:")
    print("    * Offering small discounts or loyalty points for online payments.")
    print("    * Highlighting the security and convenience of online payment options.")
    print("    * Ensuring a seamless and user-friendly online payment process within the app.")

    # General recommendations based on popular items
    # (Assuming item analysis was performed and `all_items_list` logic was successful)
    if 'Items in order' in df.columns : # A proxy to check if item analysis was attempted
         print("\n- Based on Popular Items (if analysis was successful):")
         print("  * Ensure consistent availability of top-selling items with partner restaurants.")
         print("  * Consider creating combo deals or bundles featuring popular items.")
         print("  * Leverage popular items in marketing campaigns to attract customers.")

    print("\n--- Finished Insights and Recommendations ---")


# --- Main Execution Block ---
if __name__ == "__main__":
    print("--- Starting Food Delivery Customer Behavior Analysis Script ---")

    # Load and clean the data
    # Using a copy for analysis functions to prevent unintended modifications to the cleaned_df
    # if functions were to modify DataFrames in-place (though current ones don't heavily).
    cleaned_df = load_and_clean_data()

    if cleaned_df is not None and not cleaned_df.empty:
        # Perform analyses only if data loading and cleaning were successful and resulted in data
        analyze_customer_behavior(cleaned_df.copy())
        generate_insights_and_recommendations(cleaned_df.copy())
    else:
        print("\nData loading or cleaning failed, or resulted in an empty dataset. Cannot proceed with analysis.")

    print("\n--- End of Food Delivery Customer Behavior Analysis Script ---")
    print(f"Visualizations (if any were generated) are saved in: {VISUALIZATIONS_DIR}")
