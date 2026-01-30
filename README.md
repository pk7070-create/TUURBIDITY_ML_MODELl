# Smart Turbidity-Based Water Quality Prediction using Machine Learning
 Project Overview

This project is a Machine Learning–based water quality monitoring system that predicts irrigation water quality using turbidity values.

The system classifies water into:

 Good – Safe for irrigation

 Medium – Moderate risk, treatment recommended

 Bad – Unsafe for irrigation

A Decision Tree Classifier is used to learn patterns from historical turbidity data.
The model is deployed using Streamlit to provide an interactive dashboard for real-time analysis.

This project is designed as an academic + practical prototype for smart agriculture and water management systems.


# Problem Statement

In agriculture, poor water quality (high turbidity) can:

Clog drip irrigation systems

Reduce crop yield

Damage soil health

Manual water testing is slow and not scalable.
This project uses Machine Learning to automatically assess water quality based on turbidity values.


# Working Principle 
 Data Collection

Dataset contains turbidity values and corresponding water quality labels

Example:

Low turbidity → Good

Medium turbidity → Medium

High turbidity → Bad


#  Data Preprocessing

Input (X): Turbidity value

Output (y): Water quality label

Dataset is split into:

80% Training data


#  Model Training

A Decision Tree Classifier is trained on the dataset

The model learns decision rules like:

If turbidity < threshold → Good

If turbidity is moderate → Medium

If turbidity is high → Bad


 Model Evaluation

Accuracy is calculated for both:

Training data

Testing data


Real-Time Prediction

User enters a turbidity value manually (or via sensor in future)

Model predicts water quality instantly

 Streamlit Dashboard

The Streamlit app provides:

📊 Training & Testing Accuracy

🧪 Live turbidity input

🚦 Water quality status with recommendations

📈 Dataset label distribution
