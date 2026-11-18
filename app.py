import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, mean_absolute_error

from mlxtend.frequent_patterns import apriori, association_rules

# PAGE CONFIGURATION
st.set_page_config(
    page_title="SmartBite AI",
    page_icon="🍔",
    layout="wide"
)

# LOAD DATASET
data = pd.read_csv("dataset.csv")

st.title("🍔 SmartBite AI")
st.subheader("Intelligent Food Demand Prediction & Waste Management System")

st.write(
    "An AI-powered system that helps restaurants predict food demand, "
    "identify food waste risks and understand food purchasing patterns."
)

# SIDEBAR
st.sidebar.title("SmartBite Navigation")

page = st.sidebar.radio(
    "Select Module",
    [
        "Dashboard",
        "Demand Prediction",
        "Food Waste Analysis",
        "Food Clustering",
        "Food Combo Recommendations"
    ]
)

# DASHBOARD
if page == "Dashboard":

    st.header("📊 Restaurant Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Orders", int(data["Orders"].sum()))
    col2.metric("Food Prepared", int(data["Prepared"].sum()))
    col3.metric("Food Wasted", int(data["Wasted"].sum()))

    st.subheader("Food Item Demand")

    chart = data.groupby("Food_Item")["Orders"].sum()
    st.bar_chart(chart)

    st.subheader("Dataset")
    st.dataframe(data, use_container_width=True)


# DEMAND PREDICTION
elif page == "Demand Prediction":

    st.header("📈 Food Demand Prediction")
    st.write("Algorithm: Linear Regression")

    df = data.copy()

    encoders = {}

    for col in ["Day", "Food_Item", "Weather"]:
        encoders[col] = LabelEncoder()
        df[col] = encoders[col].fit_transform(df[col])

    X = df[["Day", "Food_Item", "Prepared", "Weekend", "Weather"]]
    y = df["Orders"]

    model = LinearRegression()
    model.fit(X, y)

    food = st.selectbox(
        "Select Food Item",
        data["Food_Item"].unique()
    )

    prepared = st.number_input(
        "Enter Planned Food Quantity",
        min_value=1,
        value=150
    )

    day = st.selectbox(
        "Select Day",
        data["Day"].unique()
    )

    weather = st.selectbox(
        "Select Weather",
        data["Weather"].unique()
    )

    weekend = 1 if day in ["Saturday", "Sunday"] else 0

    if st.button("Predict Demand"):

        sample = pd.DataFrame([{
            "Day": encoders["Day"].transform([day])[0],
            "Food_Item": encoders["Food_Item"].transform([food])[0],
            "Prepared": prepared,
            "Weekend": weekend,
            "Weather": encoders["Weather"].transform([weather])[0]
        }])

        prediction = model.predict(sample)[0]

        st.success(
            f"Predicted orders for {food}: {max(0, round(prediction))}"
        )


# FOOD WASTE ANALYSIS
elif page == "Food Waste Analysis":

    st.header("🗑️ Food Waste Risk Prediction")
    st.write("Algorithm: Decision Tree Classifier")

    df = data.copy()

    df["Waste_Ratio"] = df["Wasted"] / df["Prepared"]

    df["Waste_Risk"] = np.where(
        df["Waste_Ratio"] >= 0.15,
        1,
        0
    )

    for col in ["Day", "Food_Item", "Weather"]:
        df[col] = LabelEncoder().fit_transform(df[col])

    X = df[["Orders", "Prepared", "Day", "Food_Item", "Weather"]]
    y = df["Waste_Risk"]

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)

    col1, col2 = st.columns(2)

    with col1:
        orders = st.number_input("Expected Orders", 1, 1000, 100)

    with col2:
        prepared = st.number_input("Food Prepared", 1, 1000, 150)

    if st.button("Check Waste Risk"):

        sample = pd.DataFrame([{
            "Orders": orders,
            "Prepared": prepared,
            "Day": 0,
            "Food_Item": 0,
            "Weather": 0
        }])

        result = model.predict(sample)[0]

        if result == 1:
            st.error("High Food Waste Risk")
        else:
            st.success("Low Food Waste Risk")

    waste = data.groupby("Food_Item")["Wasted"].sum()

    st.subheader("Food Waste by Item")
    st.bar_chart(waste)


# KMEANS CLUSTERING
elif page == "Food Clustering":

    st.header("👥 Food Item Clustering")
    st.write("Algorithm: K-Means Clustering")

    grouped = data.groupby("Food_Item")[["Orders", "Wasted"]].mean()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(grouped)

    model = KMeans(n_clusters=2, random_state=42, n_init=10)

    grouped["Cluster"] = model.fit_predict(X_scaled)

    st.write("Food items grouped according to demand and waste.")

    st.dataframe(grouped)

    st.subheader("Clustering Visualization")

    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=100)
    
    ax.scatter(
        grouped["Orders"],
        grouped["Wasted"],
        c=grouped["Cluster"]
    )

    for item in grouped.index:
        ax.annotate(
            item,
            (grouped.loc[item, "Orders"], grouped.loc[item, "Wasted"])
        )

    ax.set_xlabel("Average Orders")
    ax.set_ylabel("Average Waste")
    ax.set_title("Food Item Clusters")

    st.pyplot(fig, use_container_width=False)
    plt.close(fig)
# APRIORI
elif page == "Food Combo Recommendations":

    st.header("🍔 Food Combo Recommendations")
    st.write("Algorithm: Apriori Association Rule Mining")

    transactions = [
        ["Burger", "Fries", "Coke"],
        ["Burger", "Fries"],
        ["Pizza", "Coke"],
        ["Pizza", "Garlic Bread"],
        ["Burger", "Coke"],
        ["Pizza", "Coke", "Garlic Bread"],
        ["Sandwich", "Juice"],
        ["Sandwich", "Juice", "Fries"],
        ["Burger", "Fries", "Coke"],
        ["Pizza", "Garlic Bread"]
    ]

    items = sorted(set(item for basket in transactions for item in basket))

    basket_data = pd.DataFrame([
        {item: item in basket for item in items}
        for basket in transactions
    ])

    frequent_items = apriori(
        basket_data,
        min_support=0.2,
        use_colnames=True
    )

    st.subheader("Frequently Purchased Combinations")

    st.dataframe(frequent_items)

    if not frequent_items.empty:

        rules = association_rules(
            frequent_items,
            metric="confidence",
            min_threshold=0.5
        )

        st.subheader("Association Rules")

        if not rules.empty:
            st.dataframe(rules[[
                "antecedents",
                "consequents",
                "support",
                "confidence",
                "lift"
            ]])
        else:
            st.info("No association rules found.")

    else:
        st.info("No frequent item combinations found.")

st.sidebar.success("SmartBite AI is running!")