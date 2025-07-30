import pandas as pd
import numpy as np
import streamlit as st
import faiss
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
from datetime import datetime
import plotly.graph_objects as go

st.set_page_config(page_title="Retail Recommendation System", layout="wide")

# ---------------------- DATA LOADING AND PREPARATION ----------------------
@st.cache_resource
def load_full_data():
    # Load the full dataset with all necessary columns
    df = pd.read_csv("OnlineRetail.csv")
    
    # Clean and preprocess data
    df['Description'] = df['Description'].astype(str)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['Year'] = df['InvoiceDate'].dt.year
    df['Month'] = df['InvoiceDate'].dt.month
    
    # Calculate total price
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    
    # Remove cancelled orders (typically indicated by 'C' in Invoice)
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    
    # Remove negative quantities and zero-priced items
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    
    return df

@st.cache_resource
def prepare_sample_data():
    df = load_full_data()
    
    # Create a smaller sample for faster processing in the recommendation system
    sample_size = 5000
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42)
    else:
        df_sample = df
    
    return df_sample

# ---------------------- RECOMMENDATION SYSTEMS ----------------------
@st.cache_resource
def build_content_recommender():
    df_sample = prepare_sample_data()
    
    # Create unique product dataset
    product_data = df_sample.groupby('StockCode').agg({
        'Description': 'first',  # Keep the first description
        'UnitPrice': 'mean',     # Average unit price
        'Quantity': 'sum'        # Total quantity sold
    }).reset_index()
    
    # Calculate popularity scores based on quantity
    scaler = MinMaxScaler()
    product_data['PopularityScore'] = scaler.fit_transform(product_data[['Quantity']])
    product_data['PriceScore'] = scaler.fit_transform(product_data[['UnitPrice']])
    
    # Create feature matrix
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(product_data['Description'])
    
    # Convert to dense array for FAISS and ensure it's C-contiguous
    tfidf_array = np.ascontiguousarray(tfidf_matrix.toarray().astype('float32'))
    
    # Create a FAISS index
    index = faiss.IndexFlatL2(tfidf_array.shape[1])
    index.add(tfidf_array)
    
    # Create mapping from product description to index
    indices = {row['Description']: i for i, row in product_data.iterrows()}
    
    return vectorizer, tfidf_array, index, indices, product_data

def content_based_recommend(product_name, num_recommendations=5):
    vectorizer, tfidf_array, index, indices, product_data = build_content_recommender()
    
    # Check if the product exists
    if product_name not in indices:
        return []
    
    # Get the product index
    idx = indices[product_name]
    
    # Get the product vector and ensure it's C-contiguous
    query_vector = np.ascontiguousarray(tfidf_array[idx].reshape(1, -1))
    
    # Find similar items
    k = min(num_recommendations + 1, len(tfidf_array))
    distances, neighbors = index.search(query_vector, k)
    
    # Remove the input product if it's in the results
    if neighbors[0][0] == idx and len(neighbors[0]) > 1:
        rec_indices = neighbors[0][1:]
        rec_distances = distances[0][1:]
    else:
        rec_indices = neighbors[0]
        rec_distances = distances[0]
    
    # Prepare recommendations
    recommendations = []
    for i, (index, distance) in enumerate(zip(rec_indices, rec_distances)):
        if index < len(product_data):
            product = product_data.iloc[index]
            recommendations.append({
                'StockCode': product['StockCode'],
                'Description': product['Description'],
                'UnitPrice': product['UnitPrice'],
                'PopularityScore': product['PopularityScore'],
                'PriceScore': product['PriceScore'],
                'SimilarityScore': 1 / (1 + distance),  # Convert distance to similarity
            })
    
    return recommendations

@st.cache_resource
def build_user_recommender():
    df = load_full_data()
    
    # Filter customers with enough transactions
    customer_counts = df['CustomerID'].value_counts()
    valid_customers = customer_counts[customer_counts >= 5].index
    df_filtered = df[df['CustomerID'].isin(valid_customers)]
    
    # Create user-item matrix (customers x products)
    user_item_matrix = df_filtered.pivot_table(
        index='CustomerID',
        columns='StockCode',
        values='Quantity',
        aggfunc='sum',
        fill_value=0
    )
    
    # Convert to sparse matrix for FAISS and ensure it's C-contiguous
    user_vectors = np.ascontiguousarray(user_item_matrix.values.astype('float32'))
    
    # Normalize vectors
    faiss.normalize_L2(user_vectors)
    
    # Create FAISS index
    user_index = faiss.IndexFlatIP(user_vectors.shape[1])  # Inner product = cosine similarity for normalized vectors
    user_index.add(user_vectors)
    
    # Store mappings
    user_to_index = {user_id: i for i, user_id in enumerate(user_item_matrix.index)}
    index_to_user = {i: user_id for user_id, i in user_to_index.items()}
    
    # Product mapping
    product_mapping = {col: i for i, col in enumerate(user_item_matrix.columns)}
    reverse_product_mapping = {i: col for col, i in product_mapping.items()}
    
    # Get product info
    product_info = df.groupby('StockCode').agg({
        'Description': 'first',
        'UnitPrice': 'mean'
    }).reset_index()
    
    return user_vectors, user_index, user_to_index, index_to_user, product_mapping, reverse_product_mapping, product_info, user_item_matrix

def personalized_recommend(customer_id, num_recommendations=5):
    try:
        user_vectors, user_index, user_to_index, index_to_user, product_mapping, reverse_product_mapping, product_info, user_item_matrix = build_user_recommender()
        
        # Check if this customer is in our dataset
        if customer_id not in user_to_index:
            return []
        
        # Get user index
        user_idx = user_to_index[customer_id]
        
        # Get user's vector and ensure it's C-contiguous
        user_vector = np.ascontiguousarray(user_vectors[user_idx].reshape(1, -1))
        
        # Find similar users
        k = min(6, len(user_vectors))  # Get 5 similar users + the user itself
        _, similar_user_indices = user_index.search(user_vector, k)
        
        # Remove the user from the results
        similar_user_indices = similar_user_indices[0][1:]
        
        # Get products purchased by similar users but not by this user
        user_purchases = set(user_item_matrix.columns[user_item_matrix.loc[customer_id] > 0])
        
        # Collect recommended products
        potential_recommendations = {}
        
        for sim_user_idx in similar_user_indices:
            sim_user_id = index_to_user[sim_user_idx]
            sim_user_purchases = set(user_item_matrix.columns[user_item_matrix.loc[sim_user_id] > 0])
            
            # Find products the similar user bought that this user didn't
            for product in sim_user_purchases - user_purchases:
                if product in potential_recommendations:
                    potential_recommendations[product] += 1
                else:
                    potential_recommendations[product] = 1
        
        # Sort by frequency
        sorted_recommendations = sorted(potential_recommendations.items(), key=lambda x: x[1], reverse=True)[:num_recommendations]
        
        # Format recommendations
        recommendations = []
        for stock_code, score in sorted_recommendations:
            product_row = product_info[product_info['StockCode'] == stock_code]
            if not product_row.empty:
                recommendations.append({
                    'StockCode': stock_code,
                    'Description': product_row['Description'].values[0],
                    'UnitPrice': product_row['UnitPrice'].values[0],
                    'Score': score / len(similar_user_indices)  # Normalize score
                })
        
        return recommendations
    except Exception as e:
        st.error(f"Error in personalized recommendations: {str(e)}")
        return []

# ---------------------- DATA ANALYSIS FUNCTIONS ----------------------
@st.cache_resource
def analyze_top_products(n=10):
    df = load_full_data()
    
    # Calculate total sales by product
    product_sales = df.groupby(['StockCode', 'Description']).agg({
        'Quantity': 'sum',
        'TotalPrice': 'sum'
    }).reset_index()
    
    # Sort by total sales value
    top_by_value = product_sales.sort_values('TotalPrice', ascending=False).head(n)
    
    # Sort by quantity
    top_by_quantity = product_sales.sort_values('Quantity', ascending=False).head(n)
    
    return top_by_value, top_by_quantity

@st.cache_resource
def analyze_geographic_sales():
    df = load_full_data()
    
    # Calculate sales by country
    country_sales = df.groupby('Country').agg({
        'InvoiceNo': 'nunique',
        'Quantity': 'sum',
        'TotalPrice': 'sum'
    }).reset_index()
    
    country_sales.columns = ['Country', 'Number of Orders', 'Total Items Sold', 'Total Revenue']
    country_sales = country_sales.sort_values('Total Revenue', ascending=False)
    
    return country_sales

@st.cache_resource
def analyze_sales_over_time():
    df = load_full_data()
    
    # Group by month and year
    time_sales = df.groupby(['Year', 'Month']).agg({
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum'
    }).reset_index()
    
    # Create date column for better plotting
    time_sales['Date'] = time_sales.apply(lambda x: datetime(year=int(x['Year']), month=int(x['Month']), day=1), axis=1)
    time_sales = time_sales.sort_values('Date')
    
    return time_sales

# ---------------------- STREAMLIT UI ----------------------
def main():
    st.title("🛒 Advanced Retail Recommendation System")
    
    tab1, tab2, tab3 = st.tabs(["Recommendations", "Customer Analysis", "Sales Analytics"])
    
    with tab1:
        st.header("Product Recommendations")
        
        # Get unique products for selection
        vectorizer, tfidf_array, index, indices, product_data = build_content_recommender()
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Content-Based Recommendations")
            product_list = list(indices.keys())
            selected_product = st.selectbox("Select a product", product_list, key="content_product")
            
            if st.button("Get Similar Products"):
                recommendations = content_based_recommend(selected_product)
                
                if recommendations:
                    st.write("Products similar to:", selected_product)
                    
                    for i, rec in enumerate(recommendations):
                        with st.container():
                            st.markdown(f"**{i+1}. {rec['Description']}**")
                            cols = st.columns(3)
                            cols[0].metric("Price", f"${rec['UnitPrice']:.2f}")
                            cols[1].metric("Popularity", f"{rec['PopularityScore']:.2f}")
                            cols[2].metric("Similarity", f"{rec['SimilarityScore']:.2f}")
                else:
                    st.warning("No recommendations found for this product.")
        
        with col2:
            st.subheader("Personalized Recommendations")
            
            # Get unique customers
            try:
                _, _, user_to_index, _, _, _, _, _ = build_user_recommender()
                customer_ids = list(user_to_index.keys())
                
                selected_customer = st.selectbox("Select a customer ID", customer_ids, key="user_customer")
                
                if st.button("Get Personalized Recommendations"):
                    personal_recs = personalized_recommend(selected_customer)
                    
                    if personal_recs:
                        st.write(f"Recommended products for Customer #{selected_customer}:")
                        
                        for i, rec in enumerate(personal_recs):
                            with st.container():
                                st.markdown(f"**{i+1}. {rec['Description']}**")
                                cols = st.columns(2)
                                cols[0].metric("Price", f"${rec['UnitPrice']:.2f}")
                                cols[1].metric("Match Score", f"{rec['Score']:.2f}")
                    else:
                        st.warning("No personalized recommendations available for this customer.")
            except Exception as e:
                st.error(f"Error in customer recommendations: {str(e)}")
                st.info("Personalized recommendations require customers with sufficient purchase history.")
    
    with tab2:
        st.header("Customer Analysis")
        
        df = load_full_data()
        
        # Customer overview
        st.subheader("Customer Overview")
        total_customers = df['CustomerID'].nunique()
        total_countries = df['Country'].nunique()
        avg_order_value = df.groupby('InvoiceNo')['TotalPrice'].sum().mean()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Customers", f"{total_customers:,}")
        col2.metric("Countries Served", total_countries)
        col3.metric("Avg. Order Value", f"${avg_order_value:.2f}")
        
        # Top customers
        st.subheader("Top Customers by Total Spend")
        top_customers = df.groupby('CustomerID').agg({
            'InvoiceNo': 'nunique',
            'TotalPrice': 'sum'
        }).reset_index().sort_values('TotalPrice', ascending=False).head(10)
        
        top_customers.columns = ['Customer ID', 'Number of Orders', 'Total Spend']
        st.dataframe(top_customers)
        
        # Customer geographic distribution
        st.subheader("Customer Geographic Distribution")
        customers_by_country = df.groupby('Country')['CustomerID'].nunique().reset_index()
        customers_by_country.columns = ['Country', 'Number of Customers']
        customers_by_country = customers_by_country.sort_values('Number of Customers', ascending=False)
        
        fig = px.bar(customers_by_country.head(10), 
                     x='Country', y='Number of Customers',
                     title='Top 10 Countries by Number of Customers',
                     color='Number of Customers',
                     color_continuous_scale=px.colors.sequential.Blues)
        st.plotly_chart(fig)
    
    with tab3:
        st.header("Sales Analytics")
        
        # Overall sales metrics
        df = load_full_data()
        total_revenue = df['TotalPrice'].sum()
        total_orders = df['InvoiceNo'].nunique()
        total_products = df['StockCode'].nunique()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Revenue", f"${total_revenue:,.2f}")
        col2.metric("Total Orders", f"{total_orders:,}")
        col3.metric("Unique Products", f"{total_products:,}")
        
        # Top selling products
        st.subheader("Top Selling Products")
        
        top_by_value, top_by_quantity = analyze_top_products()
        
        metric_choice = st.radio("Sort by:", ["Revenue", "Quantity Sold"])
        
        if metric_choice == "Revenue":
            fig = px.bar(top_by_value, x='TotalPrice', y='Description', 
                         title='Top 10 Products by Revenue',
                         labels={'TotalPrice': 'Total Revenue ($)', 'Description': 'Product'},
                         orientation='h',
                         color='TotalPrice',
                         color_continuous_scale=px.colors.sequential.Blues)
            st.plotly_chart(fig)
        else:
            fig = px.bar(top_by_quantity, x='Quantity', y='Description', 
                         title='Top 10 Products by Quantity Sold',
                         labels={'Quantity': 'Units Sold', 'Description': 'Product'},
                         orientation='h',
                         color='Quantity',
                         color_continuous_scale=px.colors.sequential.Blues)
            st.plotly_chart(fig)
        
        # Geographic sales distribution
        st.subheader("Geographic Sales Distribution")
        country_sales = analyze_geographic_sales()
        
        fig = px.choropleth(country_sales, 
                           locations='Country', 
                           locationmode='country names',
                           color='Total Revenue', 
                           hover_name='Country',
                           color_continuous_scale=px.colors.sequential.Blues,
                           title='Revenue by Country')
        st.plotly_chart(fig)
        
        # Sales over time
        st.subheader("Sales Trends Over Time")
        time_sales = analyze_sales_over_time()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time_sales['Date'], y=time_sales['TotalPrice'],
                                mode='lines+markers', name='Revenue',
                                line=dict(color='royalblue', width=2)))
        
        fig.update_layout(title='Monthly Revenue Over Time',
                         xaxis_title='Date',
                         yaxis_title='Revenue ($)')
        st.plotly_chart(fig)
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=time_sales['Date'], y=time_sales['InvoiceNo'],
                                mode='lines+markers', name='Orders',
                                line=dict(color='darkblue', width=2)))
        
        fig2.update_layout(title='Monthly Number of Orders Over Time',
                         xaxis_title='Date',
                         yaxis_title='Number of Orders')
        st.plotly_chart(fig2)

if __name__ == "__main__":
    main()