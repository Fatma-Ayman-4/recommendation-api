import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import xgboost as xgb
from sklearn.model_selection import train_test_split
import warnings
import os
import joblib

warnings.filterwarnings('ignore')

# =============================================
# Global Paths
# =============================================
MODEL_PATH = "recommendation_model.pkl"
SIMILARITY_PATH = "item_sim_df.pkl"
POPULARITY_PATH = "product_popularity.pkl"
AVG_PRICE_PATH = "product_avg_price.pkl"

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
# 4. Prepare Features (قللنا الـ negative samples عشان يبقى أسرع)
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

        # Negative samples - قللناها من 5x إلى 2x عشان السرعة
        not_bought = row_data[row_data == 0].index
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
# 6. Recommendation Function (محسنة وأسرع)
# =============================================
def recommend(product_id, item_sim_df, model, user_item,
              product_popularity, product_avg_price, df,
              order_id=None, top_k=5):

    if product_id not in item_sim_df.index:
        return [{"error": "Product not found"}]

    # خد أول 80 منتج مشابه فقط (مش كلهم) عشان السرعة
    sim_scores = item_sim_df[product_id].sort_values(ascending=False)[1:81]
    candidates = sim_scores.index.tolist()

    rows = []
    default_user_activity = user_item.sum(axis=1).mean() if not user_item.empty else 0

    for item in candidates:
        user_activity = user_item.loc[order_id].sum() if order_id and order_id in user_item.index else default_user_activity
        rows.append({
            'popularity': float(product_popularity.get(item, 0)),
            'avg_price': float(product_avg_price.get(item, 0)),
            'user_activity': float(user_activity)
        })

    X = pd.DataFrame(rows)
    scores = model.predict(X)

    # Get top K
    top_idx = np.argsort(scores)[::-1][:top_k]
    results = []

    for i in top_idx:
        item_id = candidates[i]
        name_row = df[df['product_id'] == item_id]
        name = name_row['product_name'].iloc[0] if not name_row.empty else "Unknown"
        results.append({
            "product_id": item_id,
            "product_name": name,
            "score": float(scores[i])
        })

    return results


# =============================================
# Load or Train Model (الجزء المهم جداً)
# =============================================
print("  Initializing Recommendation System...")

df = load_data()
user_item, user_item_sparse = build_user_item_matrix(df)
item_sim_df = compute_item_similarity(user_item_sparse, user_item.columns)

# تحميل الموديل المحفوظ إذا كان موجود
if (os.path.exists(MODEL_PATH) and 
    os.path.exists(SIMILARITY_PATH) and 
    os.path.exists(POPULARITY_PATH) and 
    os.path.exists(AVG_PRICE_PATH)):
    
    print(" Loading pre-trained model...")
    model = joblib.load(MODEL_PATH)
    item_sim_df = joblib.load(SIMILARITY_PATH)
    product_popularity = joblib.load(POPULARITY_PATH)
    product_avg_price = joblib.load(AVG_PRICE_PATH)
    
else:
    print("⚠️ Training model for the first time (this may take a few minutes)...")
    features_df, product_popularity, product_avg_price = prepare_features(df, user_item)
    model = train_model(features_df)
    
    # حفظ الموديل عشان المرات الجاية تكون أسرع
    joblib.dump(model, MODEL_PATH)
    joblib.dump(item_sim_df, SIMILARITY_PATH)
    joblib.dump(product_popularity, POPULARITY_PATH)
    joblib.dump(product_avg_price, AVG_PRICE_PATH)
    print(" Model trained and saved successfully!")

print(" System Initialized Successfully!")

# =============================================
# Flask API
# =============================================
from flask import Flask, jsonify, request

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

        results = recommend(
            product_id=product_id,
            item_sim_df=item_sim_df,
            model=model,
            user_item=user_item,
            product_popularity=product_popularity,
            product_avg_price=product_avg_price,
            df=df,
            order_id=order_id,
            top_k=top_k
        )

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Run the app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
