import streamlit as st
import pandas as pd
import numpy as np
import math

# ---------------- Entropy Function ----------------
def entropy(col):
    values, counts = np.unique(col, return_counts=True)
    return -sum((counts[i]/len(col)) * math.log2(counts[i]/len(col)) for i in range(len(counts)))

# ---------------- Information Gain ----------------
def info_gain(df, attr, target):
    total_entropy = entropy(df[target])
    values, counts = np.unique(df[attr], return_counts=True)
    
    weighted_entropy = sum(
        (counts[i]/len(df)) * entropy(df[df[attr] == values[i]][target])
        for i in range(len(values))
    )
    
    return total_entropy - weighted_entropy

# ---------------- Build Decision Tree ----------------
def build_tree(df, target, features):
    if len(np.unique(df[target])) == 1:
        return np.unique(df[target])[0]
    
    if len(features) == 0:
        return df[target].mode()[0]
    
    gains = [info_gain(df, feature, target) for feature in features]
    best_feature = features[np.argmax(gains)]
    
    tree = {best_feature: {}}
    
    for value in np.unique(df[best_feature]):
        sub_data = df[df[best_feature] == value]
        remaining_features = [f for f in features if f != best_feature]
        tree[best_feature][value] = build_tree(sub_data, target, remaining_features)
    
    return tree

# ---------------- Prediction ----------------
def predict(tree, input_data):
    if not isinstance(tree, dict):
        return tree
    
    root = next(iter(tree))
    value = input_data[root]
    
    if value in tree[root]:
        return predict(tree[root][value], input_data)
    else:
        return "Unknown"

# ---------------- Streamlit UI ----------------
st.title("🌳 Decision Tree Classifier (Without Matplotlib)")

# Sample Dataset
data = {
    "outlook": ['sunny', 'sunny', 'overcast', 'rain', 'rain', 'rain', 
                'overcast', 'sunny', 'sunny', 'rain'],
    "temperature": ['hot', 'hot', 'hot', 'mild', 'cool', 'cool',
                    'mild', 'cool', 'mild', 'mild'],
    "humidity": ['high', 'high', 'high', 'high', 'normal', 'normal',
                 'normal', 'high', 'normal', 'normal'],
    "wind": ['weak', 'strong', 'weak', 'weak', 'weak', 'strong',
             'strong', 'weak', 'weak', 'strong'],
    "play": ['no', 'no', 'yes', 'yes', 'yes', 'no',
             'yes', 'no', 'yes', 'yes']
}

df = pd.DataFrame(data)

target = "play"
features = [col for col in df.columns if col != target]

# Build Tree
tree = build_tree(df, target, features)

# ---------------- User Input ----------------
st.sidebar.header("Input Features")

input_data = {}
for feature in features:
    input_data[feature] = st.sidebar.selectbox(
        feature, df[feature].unique()
    )

# ---------------- Prediction ----------------
result = predict(tree, input_data)

st.subheader("Prediction Result")
st.success(f"Prediction: {result}")

# ---------------- Graph (No matplotlib) ----------------
st.subheader("Dataset Class Distribution")

class_counts = df[target].value_counts()
st.bar_chart(class_counts)

st.subheader("Decision Tree Structure")
st.json(tree)
