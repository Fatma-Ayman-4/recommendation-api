import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import xgboost as xgb
from sklearn.model_selection import train_test_split
import warnings
import os
import joblib
from flask import Flask, jsonify, request

warnings.filterwarnings('ignore')

# =============================================
# Global Variables & Model Saving Paths
# =============================================
MODEL_PATH = "xgboost_ranker_model.pkl"
SIMILARITY_PATH = "item_sim_df.pkl"
POPULARITY_PATH = "product_popularity.pkl"
AVG_PRICE_PATH = "product_avg_price.pkl"
USER_ITEM_PATH = "user_item.pkl"

# =============================================
# 1. Load Data
# =============================================
print("🚀 Loading data...")
file_path = "data.xlsx"
df = pd.read_excel(file_path, sheet_name='Sheet1')
df['order_id'] = df['order_id'].astype(int)
df['product_id'] = df['product_id'].astype(str)
df['interaction'] = df['quantity']

print(f"Data loaded successfully! Shape: {df.shape}")

# =============================================
# 2. Build User-Item Matrix
# =============================================
user_item = df.pivot_table(
    index='order_id',
    columns='product_id',
    values='interaction',
    fill_value=0
)
user_item_sparse = csr_matrix(user_item.values)

print(f"User-Item Matrix Ready: {user_item.shape}")

# =============================================
# 3. Item Similarity
# =============================================
item_sim_df = pd.DataFrame(
    cosine_similarity(user_item_sparse.T),
    index=user_item.columns,
    columns=user_item.columns
)

# =============================================
# 4. Features for Ranking
# =============================================
product_popularity = user_item.sum(axis=0)
product_avg_price = df.groupby('product_id')['price_egp'].mean()

def create_rank_features(df, user_item_matrix):
    data = []
    for order_id in user_item_matrix.index:
        bought_items = user_item_matrix.loc[order_id][user_item_matrix.loc[order_id] > 0].index

        # Positive samples
        for item in bought_items:
            data.append({
                'order_id': order_id,
                'item_id': item,
                'label': 1,
                'popularity': product_popularity[item],
                'avg_price': product_avg_price[item],
                'user_activity': user_item_matrix.loc[order_id].sum()
            })

        # Negative samples (قللنا لـ 2x عشان السرعة)
        not_bought = user_item_matrix.loc[order_id][user_item_matrix.loc[order_id] == 0].index
        sample_size = min(2 * len(bought_items), len(not_bought))
        if sample_size > 0:
            sampled = np.random.choice(not_bought, size=sample_size, replace=False)
            for item in sampled:
                data.append({
                    'order_id': order_id,
                    'item_id': item,
                    'label': 0,
                    'popularity': product_popularity[item],
                    'avg_price': product_avg_price[item],
                    'user_activity': user_item_matrix.loc[order_id].sum()
                })

    return pd.DataFrame(data)

# =============================================
# Load or Train Model (مهم جداً)
# =============================================
if (os.path.exists(MODEL_PATH) and os.path.exists(SIMILARITY_PATH)):
    print("✅ Loading pre-trained model...")
    model = joblib.load(MODEL_PATH)
    item_sim_df = joblib.load(SIMILARITY_PATH)
    product_popularity = joblib.load(POPULARITY_PATH)
    product_avg_price = joblib.load(AVG_PRICE_PATH)
    user_item = joblib.load(USER_ITEM_PATH)
else:
    print("⚠️ Training model for the first time (this may take some time)...")
    features_df = create_rank_features(df, user_item)
    
    X = features_df.drop(['order_id', 'item_id', 'label'], axis=1)
    y = features_df['label']

    train_idx, test_idx = train_test_split(features_df['order_id'].unique(), test_size=0.2, random_state=42)
    train_mask = features_df['order_id'].isin(train_idx)
    test_mask = features_df['order_id'].isin(test_idx)

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    groups_train = features_df[train_mask].groupby('order_id').size().values
    groups_test = features_df[test_mask].groupby('order_id').size().values

    model = xgb.XGBRanker(
        objective='rank:pairwise',
        random_state=42,
        learning_rate=0.1,
        n_estimators=200,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='ndcg'
    )

    model.fit(
        X_train, y_train,
        group=groups_train,
        eval_set=[(X_test, y_test)],
        eval_group=[groups_test],
        verbose=True
    )

    # حفظ كل حاجة
    joblib.dump(model, MODEL_PATH)
    joblib.dump(item_sim_df, SIMILARITY_PATH)
    joblib.dump(product_popularity, POPULARITY_PATH)
    joblib.dump(product_avg_price, AVG_PRICE_PATH)
    joblib.dump(user_item, USER_ITEM_PATH)
    print("✅ Model trained and saved!")

print("✅ System Initialized Successfully!")

# =============================================
# Recommendation Function (من كودك)
# =============================================
def recommend_similar_products(product_id, order_id=None, top_k=5, min_similarity=0.02):
    if product_id not in item_sim_df.index:
        return [{"error": f"Product {product_id} not found in data"}]

    sim_scores = item_sim_df[product_id].sort_values(ascending=False)
    sim_scores = sim_scores[sim_scores.index != product_id]
    sim_scores = sim_scores[sim_scores >= min_similarity]

    if sim_scores.empty:
        return [{"message": "No sufficiently similar products were found."}]

    candidate_items = sim_scores.index.tolist()

    feature_rows = []
    for item in candidate_items:
        user_activity = user_item.loc[order_id].sum() if order_id and order_id in user_item.index else user_item.sum(axis=1).mean()
        feature_rows.append({
            'popularity': product_popularity[item],
            'avg_price': product_avg_price[item],
            'user_activity': user_activity
        })

    X_candidates = pd.DataFrame(feature_rows)
    scores = model.predict(X_candidates)

    ranked_idx = np.argsort(scores)[::-1][:top_k]
    result = []

    for idx in ranked_idx:
        item_id = candidate_items[idx]
        score = scores[idx]
        name_row = df[df['product_id'] == item_id]['product_name']
        product_name = name_row.iloc[0] if not name_row.empty else "Name not available"

        result.append({
            "product_id": item_id,
            "product_name": product_name,
            "ranking_score": round(float(score), 4)
        })

    return result

# =============================================
# Flask API
# =============================================
app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"message": "Recommendation API is running 🚀"})

@app.route("/recommend", methods=["POST"])
def recommend_endpoint():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        product_id = data.get("product_id")
        order_id = data.get("order_id")
        top_k = data.get("top_k", 5)

        if not product_id:
            return jsonify({"error": "product_id is required"}), 400

        results = recommend_similar_products(
            product_id=product_id,
            order_id=order_id,
            top_k=top_k
        )

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
