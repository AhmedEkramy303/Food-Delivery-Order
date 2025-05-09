# Food Delivery App Customer Behavior Analysis

## Objective

This project aims to understand the habits and preferences of customers using a food delivery application. We seek to answer questions such as:
* Who are the most active customers (highest order frequency)?
* What are the peak ordering times (days of the week and hours of the day)?
* What are the most popular meals or items?
* What are the usage ratios of different payment methods (if data is available)?

The analysis intends to derive actionable insights that can help improve user experience, personalize offers, and enhance operational efficiency.

## Project Steps

1.  **Load and Clean Data:**
    * Load the order history dataset (`order_history_kaggle_data.csv`).
    * Handle missing values.
    * Remove duplicate orders.
    * Convert the order timestamp column (`Order Placed At`) to a proper datetime format.
    * Filter orders to focus on completed/delivered orders for relevant analyses.

2.  **Customer Behavior Analysis:**
    * Identify the customer(s) with the highest number of orders.
    * Analyze the distribution of orders throughout the hours of the day.
    * Analyze the distribution of orders across the days of the week.
    * Analyze the most frequently ordered items by processing the `Items in order` column.

3.  **Visualizations:**
    * **Bar Chart:** To display the number of orders for the most popular items.
    * **Line Plot:** To show the general trend of order volume over the available period (daily).
    * **Pie Chart (Conditional):** To display the proportions of different payment methods (e.g., Cash vs. Online). **Note:** This data was not found in the current dataset. The script will include a placeholder for how this analysis would be performed if the data were available.

4.  **Insights and Recommendations:**
    * Suggest recommendations based on observed patterns, such as:
        * Offering promotions during off-peak days or hours to boost sales.
        * Ensuring high availability of popular meal items.
        * If payment data were available and showed a preference, strategies to encourage other methods or improve the experience of less-used ones could be suggested.

## Setup and Execution

1.  **Clone the Repository (if you upload it to GitHub first):**
    ```bash
    git clone <your_github_repository_url>
    cd food_delivery_customer_behavior_analysis
    ```
    (If you are setting it up locally first, just create the project directory as described below).

2.  **Create a Virtual Environment (Optional but Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On macOS/Linux
    # venv\Scripts\activate    # On Windows
    ```

3.  **Install Required Libraries:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Place Data:**
    * Create a `data` folder in the project's root directory.
    * Place your `order_history_kaggle_data.csv` file inside the `data/` folder.

5.  **Run the Analysis Script:**
    ```bash
    python scripts/analysis_customer_behavior.py
    ```
    The script will create a `visualizations/` folder (if it doesn't exist) and save the generated plots there.

## Data Used

The data used is `order_history_kaggle_data.csv`. Key columns focused on in this analysis include:
* `Customer ID`
* `Order ID`
* `Order Placed At`
* `Order Status`
* `Items in order`

## Tools and Libraries

* Python 3.x
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Collections (for counting items)

---
