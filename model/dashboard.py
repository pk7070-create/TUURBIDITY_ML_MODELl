import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


st.set_page_config(
    page_title="Water Quality Dashboard",
    layout="centered"
)

st.title("💧 Water Quality Monitoring Dashboard")
st.markdown("""
# 💧🌱 Smart Irrigation Water Quality Dashboard
**AI-powered turbidity analysis for agricultural water**
""")

st.caption(
    "This system evaluates irrigation water quality and provides risk-based recommendations using Machine Learning."
)




df = pd.read_csv("data/turbidity_data.csv")

X = df[['turbidity']]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc  = accuracy_score(y_test, model.predict(X_test))



st.subheader("📊 Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        label="📘 Training Accuracy",
        value=f"{train_acc:.2f}"
    )

with col2:
    st.metric(
        label="📗 Testing Accuracy",
        value=f"{test_acc:.2f}"
    )



st.markdown("## 🧪 Live Water Quality Check")
st.markdown("Enter the turbidity value obtained from the sensor or sample test.")


turbidity = st.number_input(
    "Enter Turbidity Value",
    min_value=0,
    max_value=2000,
    step=10
)


if st.button("🔍 Analyze Water Quality"):
    result = model.predict([[turbidity]])[0]

    st.subheader("📌 Water Quality Assessment Result")

    if result == "Good":
        st.success("✅💧 Status: SAFE FOR IRRIGATION")
        st.info("🌱 Water quality is suitable for crops. No treatment required.")

    elif result == "Medium":
        st.warning("⚠️💧 Status: MODERATE RISK")
        st.warning(
            "🚧 **Caution Required**\n\n"
            "- Water is not ideal for direct irrigation\n"
            "- Filtration or settling is recommended\n"
            "- Monitor soil and crop response"
        )

    else:
        st.error("🚫💧 Status: UNSAFE FOR IRRIGATION")
        st.error(
            "☠️ **High Risk Detected**\n\n"
            "- May clog drip irrigation systems\n"
            "- Can damage crops and soil quality\n"
            "- Treatment is strongly recommended"
        )



st.markdown("## 📈 Dataset Insights")

st.markdown("### Water Quality Class Distribution")
st.bar_chart(df["label"].value_counts())


st.markdown("---")
st.caption(
    "🔬 Academic prototype | Future scope:\n"
    "- Real-time turbidity sensor integration (ESP32/Arduino)\n"
    "- Automated alert system for farmers\n"
    "- Mobile app or web API support"
)





