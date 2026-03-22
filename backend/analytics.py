import pandas as pd
from datetime import datetime
from recommender import load_full_data

# ---------------------- ANALYTICS ----------------------

def get_kpi_metrics() -> dict:
    df = load_full_data()
    total_revenue = round(float(df['TotalPrice'].sum()), 2)
    total_orders = int(df['InvoiceNo'].nunique())
    total_products = int(df['StockCode'].nunique())
    total_customers = int(df['CustomerID'].nunique())
    avg_order_value = round(float(df.groupby('InvoiceNo')['TotalPrice'].sum().mean()), 2)
    total_countries = int(df['Country'].nunique())

    return {
        "totalRevenue": total_revenue,
        "totalOrders": total_orders,
        "totalProducts": total_products,
        "totalCustomers": total_customers,
        "avgOrderValue": avg_order_value,
        "totalCountries": total_countries,
    }


def get_top_products(n: int = 10) -> dict:
    df = load_full_data()

    product_sales = df.groupby(['StockCode', 'Description']).agg(
        Quantity=('Quantity', 'sum'),
        TotalPrice=('TotalPrice', 'sum')
    ).reset_index()

    top_by_value = product_sales.sort_values('TotalPrice', ascending=False).head(n)
    top_by_quantity = product_sales.sort_values('Quantity', ascending=False).head(n)

    return {
        "byRevenue": top_by_value[['StockCode', 'Description', 'Quantity', 'TotalPrice']].to_dict(orient='records'),
        "byQuantity": top_by_quantity[['StockCode', 'Description', 'Quantity', 'TotalPrice']].to_dict(orient='records'),
    }


def get_geo_sales(product: str = None) -> list:
    df = load_full_data()

    if product:
        df = df[df['Description'].str.lower() == product.lower()]

    country_sales = df.groupby('Country').agg(
        orders=('InvoiceNo', 'nunique'),
        itemsSold=('Quantity', 'sum'),
        revenue=('TotalPrice', 'sum')
    ).reset_index()

    country_sales = country_sales.sort_values('revenue', ascending=False)
    country_sales['revenue'] = country_sales['revenue'].round(2)

    return country_sales.rename(columns={'Country': 'country'}).to_dict(orient='records')


def get_sales_timeline(product: str = None) -> list:
    df = load_full_data()

    if product:
        df = df[df['Description'].str.lower() == product.lower()]

    time_sales = df.groupby(['Year', 'Month']).agg(
        orders=('InvoiceNo', 'nunique'),
        revenue=('TotalPrice', 'sum')
    ).reset_index()

    time_sales['date'] = time_sales.apply(
        lambda x: datetime(year=int(x['Year']), month=int(x['Month']), day=1).isoformat(), axis=1
    )
    time_sales = time_sales.sort_values('date')
    time_sales['revenue'] = time_sales['revenue'].round(2)

    return time_sales[['date', 'orders', 'revenue']].to_dict(orient='records')


def get_customer_stats() -> dict:
    df = load_full_data()

    total_customers = int(df['CustomerID'].nunique())
    total_countries = int(df['Country'].nunique())
    avg_order_value = round(float(df.groupby('InvoiceNo')['TotalPrice'].sum().mean()), 2)

    top_customers = (
        df.groupby('CustomerID').agg(
            orders=('InvoiceNo', 'nunique'),
            totalSpend=('TotalPrice', 'sum')
        )
        .reset_index()
        .sort_values('totalSpend', ascending=False)
        .head(10)
    )
    top_customers['totalSpend'] = top_customers['totalSpend'].round(2)
    top_customers['customerId'] = top_customers['CustomerID'].astype(str)
    top_customers = top_customers.drop(columns=['CustomerID'])

    customers_by_country = (
        df.groupby('Country')['CustomerID'].nunique()
        .reset_index()
        .rename(columns={'Country': 'country', 'CustomerID': 'customers'})
        .sort_values('customers', ascending=False)
        .head(10)
    )

    return {
        "totalCustomers": total_customers,
        "totalCountries": total_countries,
        "avgOrderValue": avg_order_value,
        "topCustomers": top_customers.to_dict(orient='records'),
        "customersByCountry": customers_by_country.to_dict(orient='records'),
    }
