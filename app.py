# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score
import warnings
import os
from flask import Flask, jsonify, request

warnings.filterwarnings('ignore')

# =============================================
# 1. Load Data
# =============================================
def load_data():
    try:
        base_path = os.path.dirname(__file__)
    except NameError:
        base_path = os.getcwd()

    file_path = os.path.join(base_path, "data.xlsx")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"ملف data.xlsx مش موجود في المسار: {file_path}")

    df = pd.read_excel(file_path, sheet_name='Sheet1')

    df['order_id'] = df['order_id'].astype(int)
    df['product_id'] = df['product_id'].astype(str)
    df['interaction'] = df['quantity']

    return df


# =============================================
# 2. Build User-Item Matrix
# =============================================
def build_user_item_matrix(df):
    user_item = df.pivot_table(
        index='order_id',
        columns='product_id',
        values='interaction',
        fill_value=0
    )
    user_item_sparse = csr_matrix(user_item.values)
    return user_item, user_item_sparse


# =============================================
# 3. Item Similarity
# =============================================
def compute_item_similarity(user_item_sparse, columns):
    similarity = cosine_similarity(user_item_sparse.T)
    return pd.DataFrame(similarity, index=columns, columns=columns)


# =============================================
# 4. Prepare Features
# =============================================
def prepare_features(df, user_item):
    product_popularity = user_item.sum(axis=0)
    product_avg_price = df.groupby('product_id')['price_egp'].mean()

    data = []

    for order_id in user_item.index:
        row_data = user_item.loc[order_id]
        bought_items = row_data[row_data > 0].index

        # Positive samples
        for item in bought_items:
            data.append({
                'order_id': order_id,
                'item_id': item,
                'label': 1,
                'popularity': product_popularity[item],
                'avg_price': product_avg_price[item],
                'user_activity': row_data.sum()
            })

        # Negative samples
        not_bought = row_data[row_data == 0].index
        sample_size = min(5 * len(bought_items), len(not_bought))

        if sample_size > 0:
            sampled = np.random.choice(not_bought, size=sample_size, replace=False)
            for item in sampled:
                data.append({
                    'order_id': order_id,
                    'item_id': item,
                    'label': 0,
                    'popularity': product_popularity[item],
                    'avg_price': product_avg_price[item],
                    'user_activity': row_data.sum()
                })

    return pd.DataFrame(data), product_popularity, product_avg_price


# =============================================
# 5. Train Model
# =============================================
def train_model(features_df):
    X = features_df.drop(['order_id', 'item_id', 'label'], axis=1)
    y = features_df['label']

    train_orders, test_orders = train_test_split(
        features_df['order_id'].unique(), test_size=0.2, random_state=42
    )

    train_mask = features_df['order_id'].isin(train_orders)
    test_mask = features_df['order_id'].isin(test_orders)

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    groups_train = features_df[train_mask].groupby('order_id').size().values
    groups_test = features_df[test_mask].groupby('order_id').size().values

    model = xgb.XGBRanker(
        objective='rank:pairwise',
        learning_rate=0.1,
        n_estimators=200,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='ndcg',
        random_state=42
    )

    model.fit(
        X_train, y_train,
        group=groups_train,
        eval_set=[(X_test, y_test)],
        eval_group=[groups_test],
        verbose=False
    )

    return model


# =============================================
# 6. Recommendation Function
# =============================================
def recommend(product_id, item_sim_df, model, user_item,
              product_popularity, product_avg_price, df,
              order_id=None, top_k=5):

    if product_id not in item_sim_df.index:
        return [{"error": "Product not found"}]

    # Get similar items
    sim_scores = item_sim_df[product_id].sort_values(ascending=False)[1:]
    candidates = sim_scores.index.tolist()

    rows = []
    for item in candidates:
        user_activity = user_item.loc[order_id].sum() if order_id in user_item.index else user_item.sum(axis=1).mean()

        rows.append({
            'popularity': product_popularity[item],
            'avg_price': product_avg_price[item],
            'user_activity': user_activity
        })

    X = pd.DataFrame(rows)
    scores = model.predict(X)

    # Get top K
    top_idx = np.argsort(scores)[::-1][:top_k]

