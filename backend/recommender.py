import os
import pandas as pd
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# Resolve CSV path relative to this file so it works from any working directory
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_HERE, "..", "OnlineRetail.csv")

# ---------------------- DATA LOADING ----------------------

_full_data_cache = None
_sample_data_cache = None

def load_full_data() -> pd.DataFrame:
    global _full_data_cache
    if _full_data_cache is not None:
        return _full_data_cache

    df = pd.read_csv(DATA_PATH)
    df['Description'] = df['Description'].astype(str)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['Year'] = df['InvoiceDate'].dt.year
    df['Month'] = df['InvoiceDate'].dt.month
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

    _full_data_cache = df
    return df


def prepare_sample_data() -> pd.DataFrame:
    global _sample_data_cache
    if _sample_data_cache is not None:
        return _sample_data_cache

    df = load_full_data()
    sample_size = 5000
    _sample_data_cache = df.sample(n=sample_size, random_state=42) if len(df) > sample_size else df
    return _sample_data_cache


# ---------------------- CONTENT-BASED RECOMMENDER ----------------------

_content_recommender_cache = None

def build_content_recommender():
    global _content_recommender_cache
    if _content_recommender_cache is not None:
        return _content_recommender_cache

    df_sample = prepare_sample_data()

    product_data = df_sample.groupby('StockCode').agg(
        Description=('Description', 'first'),
        UnitPrice=('UnitPrice', 'mean'),
        Quantity=('Quantity', 'sum')
    ).reset_index()

    scaler = MinMaxScaler()
    product_data['PopularityScore'] = scaler.fit_transform(product_data[['Quantity']])
    product_data['PriceScore'] = scaler.fit_transform(product_data[['UnitPrice']])

    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(product_data['Description'])
    tfidf_array = np.ascontiguousarray(tfidf_matrix.toarray().astype('float32'))

    index = faiss.IndexFlatL2(tfidf_array.shape[1])
    index.add(tfidf_array)

    indices = {row['Description']: i for i, row in product_data.iterrows()}

    _content_recommender_cache = (vectorizer, tfidf_array, index, indices, product_data)
    return _content_recommender_cache


def content_based_recommend(product_name: str, num_recommendations: int = 5) -> list:
    vectorizer, tfidf_array, index, indices, product_data = build_content_recommender()

    if product_name not in indices:
        return []

    idx = indices[product_name]
    query_vector = np.ascontiguousarray(tfidf_array[idx].reshape(1, -1))

    k = min(num_recommendations + 1, len(tfidf_array))
    distances, neighbors = index.search(query_vector, k)

    if neighbors[0][0] == idx and len(neighbors[0]) > 1:
        rec_indices = neighbors[0][1:]
        rec_distances = distances[0][1:]
    else:
        rec_indices = neighbors[0]
        rec_distances = distances[0]

    recommendations = []
    for i, (rec_idx, distance) in enumerate(zip(rec_indices, rec_distances)):
        if rec_idx < len(product_data):
            product = product_data.iloc[rec_idx]
            recommendations.append({
                'StockCode': product['StockCode'],
                'Description': product['Description'],
                'UnitPrice': round(float(product['UnitPrice']), 2),
                'PopularityScore': round(float(product['PopularityScore']), 4),
                'PriceScore': round(float(product['PriceScore']), 4),
                'SimilarityScore': round(1 / (1 + float(distance)), 4),
                'Reason': f"Similar product description ({round(1 / (1 + float(distance)) * 100, 1)}% match)",
            })

    return recommendations


# ---------------------- CATEGORY HELPERS ----------------------

# Ordered list of (category_name, keywords). First match wins.
_CATEGORY_RULES = [
    ("Christmas",        ["CHRISTMAS", "XMAS", "NOEL", "SANTA", "REINDEER", "SNOWMAN", "SNOWFLAKE", "ADVENT"]),
    ("Candles & Lights", ["CANDLE", "T-LIGHT", "TLIGHT", "LANTERN", "LIGHT HOLDER", "NIGHTLIGHT", "FAIRY LIGHT"]),
    ("Kitchen & Dining", ["KITCHEN", "CUP", "MUG", "PLATE", "BOWL", "CAKE", "BAKING", "SPOON", "FORK", "KNIFE", "TEAPOT", "COFFEE", "BREAD", "RECIPE"]),
    ("Garden & Plants",  ["GARDEN", "PLANT", "FLOWER", "HERB", "WATERING", "SEED", "POTTING", "TROWEL"]),
    ("Bags & Storage",   ["BAG", "BASKET", "STORAGE", "TIN", "JAR", "BOTTLE", "POUCH", "TOTE", "SHOPPER", "TRUNK", "CHEST"]),
    ("Stationery",       ["CARD", "NOTEBOOK", "PENCIL", "PEN ", "DIARY", "NOTEPAD", "JOURNAL", "STICKER", "STICKY", "POSTCARD", "WRITING"]),
    ("Toys & Games",     ["TOY", "GAME", "PUZZLE", "DOLL", "BEAR", "BUNNY", "RABBIT", "MONKEY", "GIRAFFE", "ELEPHANT", "PLAYSET"]),
    ("Party & Gifting",  ["PARTY", "BIRTHDAY", "BALLOON", "GIFT", "WRAP", "RIBBON", "BOW ", "BUNTING", "BANNER", "CELEBRATION"]),
    ("Clothing & Acc.",  ["HAT", "SCARF", "GLOVES", "APRON", "UMBRELLA", "SOCKS", "PURSE", "WALLET"]),
    ("Home Decor",       ["HEART", "FRAME", "WALL", "CLOCK", "MIRROR", "CUSHION", "CURTAIN", "VASE", "ORNAMENT", "FIGURINE", "STATUE", "SIGN ", "PLAQUE"]),
    ("Vintage & Retro",  ["VINTAGE", "RETRO", "ANTIQUE", "DISTRESSED", "RUSTIC"]),
]

def categorize_product(description: str) -> str:
    desc = description.upper()
    for category, keywords in _CATEGORY_RULES:
        if any(kw in desc for kw in keywords):
            return category
    return "Other"


def get_all_products(category: str = None) -> list:
    _, _, _, indices, product_data = build_content_recommender()
    if category:
        filtered = product_data[product_data['Description'].apply(categorize_product) == category]
        return sorted(filtered['Description'].tolist())
    return sorted(list(indices.keys()))


def get_all_categories() -> list:
    _, _, _, _, product_data = build_content_recommender()
    cats = product_data['Description'].apply(categorize_product).value_counts()
    # Return categories sorted by product count, with "Other" last
    ordered = [c for c in cats.index if c != "Other"] + (["Other"] if "Other" in cats.index else [])
    return [{"name": c, "count": int(cats[c])} for c in ordered]


# ---------------------- COLLABORATIVE FILTERING ----------------------

_user_recommender_cache = None

def build_user_recommender():
    global _user_recommender_cache
    if _user_recommender_cache is not None:
        return _user_recommender_cache

    df = load_full_data()

    customer_counts = df['CustomerID'].value_counts()
    valid_customers = customer_counts[customer_counts >= 5].index
    df_filtered = df[df['CustomerID'].isin(valid_customers)]

    user_item_matrix = df_filtered.pivot_table(
        index='CustomerID',
        columns='StockCode',
        values='Quantity',
        aggfunc='sum',
        fill_value=0
    )

    user_vectors = np.ascontiguousarray(user_item_matrix.values.astype('float32'))
    faiss.normalize_L2(user_vectors)

    user_index = faiss.IndexFlatIP(user_vectors.shape[1])
    user_index.add(user_vectors)

    user_to_index = {user_id: i for i, user_id in enumerate(user_item_matrix.index)}
    index_to_user = {i: user_id for user_id, i in user_to_index.items()}

    product_mapping = {col: i for i, col in enumerate(user_item_matrix.columns)}
    reverse_product_mapping = {i: col for col, i in product_mapping.items()}

    product_info = df.groupby('StockCode').agg(
        Description=('Description', 'first'),
        UnitPrice=('UnitPrice', 'mean')
    ).reset_index()

    _user_recommender_cache = (
        user_vectors, user_index, user_to_index, index_to_user,
        product_mapping, reverse_product_mapping, product_info, user_item_matrix
    )
    return _user_recommender_cache


def personalized_recommend(customer_id, num_recommendations: int = 5) -> list:
    try:
        user_vectors, user_index, user_to_index, index_to_user, \
            product_mapping, reverse_product_mapping, product_info, user_item_matrix = build_user_recommender()

        if customer_id not in user_to_index:
            return []

        user_idx = user_to_index[customer_id]
        user_vector = np.ascontiguousarray(user_vectors[user_idx].reshape(1, -1))

        k = min(6, len(user_vectors))
        _, similar_user_indices = user_index.search(user_vector, k)
        similar_user_indices = similar_user_indices[0][1:]

        user_purchases = set(user_item_matrix.columns[user_item_matrix.loc[customer_id] > 0])

        potential_recommendations = {}
        for sim_user_idx in similar_user_indices:
            sim_user_id = index_to_user[sim_user_idx]
            sim_user_purchases = set(user_item_matrix.columns[user_item_matrix.loc[sim_user_id] > 0])
            for product in sim_user_purchases - user_purchases:
                potential_recommendations[product] = potential_recommendations.get(product, 0) + 1

        sorted_recommendations = sorted(
            potential_recommendations.items(), key=lambda x: x[1], reverse=True
        )[:num_recommendations]

        recommendations = []
        for stock_code, score in sorted_recommendations:
            product_row = product_info[product_info['StockCode'] == stock_code]
            if not product_row.empty:
                norm_score = round(score / len(similar_user_indices), 4)
                recommendations.append({
                    'StockCode': stock_code,
                    'Description': product_row['Description'].values[0],
                    'UnitPrice': round(float(product_row['UnitPrice'].values[0]), 2),
                    'Score': norm_score,
                    'Reason': f"Bought by {score} of your similar customers",
                })

        return recommendations
    except Exception as e:
        print(f"Error in personalized recommendations: {e}")
        return []


def get_all_customers() -> list:
    _, _, user_to_index, _, _, _, _, _ = build_user_recommender()
    return [str(c) for c in sorted(user_to_index.keys())]


# ---------------------- HYBRID RECOMMENDER ----------------------

def hybrid_recommend(
    product_name: str,
    customer_id=None,
    num_recommendations: int = 5,
    content_weight: float = 0.6,
    collab_weight: float = 0.4,
) -> list:
    content_recs = content_based_recommend(product_name, num_recommendations * 2)
    collab_recs = personalized_recommend(customer_id, num_recommendations * 2) if customer_id else []

    scores: dict = {}
    details: dict = {}

    for rec in content_recs:
        key = rec['StockCode']
        scores[key] = scores.get(key, 0) + rec['SimilarityScore'] * content_weight
        details[key] = rec

    for rec in collab_recs:
        key = rec['StockCode']
        scores[key] = scores.get(key, 0) + rec['Score'] * collab_weight
        if key not in details:
            details[key] = rec

    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:num_recommendations]

    results = []
    for stock_code, hybrid_score in sorted_items:
        item = details[stock_code].copy()
        item['HybridScore'] = round(hybrid_score, 4)
        item['Reason'] = "Matched by content similarity and purchase patterns"
        results.append(item)

    return results
