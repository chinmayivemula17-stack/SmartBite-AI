# SmartBite AI – Intelligent Food Demand Prediction and Waste Management System

## Project Overview
SmartBite AI is a machine learning-based web application designed to help food businesses predict food demand, analyze food waste, and make better inventory decisions.

The application uses machine learning techniques to study food sales patterns and provide useful insights.

## Objectives
- Predict food demand based on selected inputs.
- Analyze food waste and identify high-waste food items.
- Group food items using K-Means clustering.
- Recommend frequently purchased food combinations.
- Support better food inventory management.

## Technologies Used
- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Mlxtend

## Machine Learning Algorithms
1. Linear Regression – Demand prediction.
2. Decision Tree Classification – Food waste analysis.
3. K-Means Clustering – Grouping food items.
4. Apriori Algorithm – Food combination recommendations.

## Project Structure

```text
SmartBite_AI/
│
├── app.py
├── dataset.csv
├── requirements.txt
└── README.md
```

## Installation and Execution

### Step 1: Install Python
Install Python on your computer.

### Step 2: Install Required Libraries
Open the terminal and run:

```bash
pip install -r requirements.txt
```

### Step 3: Run the Application

```bash
streamlit run app.py
```

The application will open in your web browser.

## Dataset
The project currently uses a small simulated food dataset containing:
- Day
- Food Item
- Orders
- Prepared Quantity
- Wasted Quantity
- Weather
- Weekend Information

The dataset is intended for demonstration and academic learning.

## Future Enhancements
- Use real restaurant sales datasets.
- Improve prediction accuracy with larger datasets.
- Add real-time inventory tracking.
- Develop a database for storing sales information.
- Deploy the application online.

## Conclusion
SmartBite AI demonstrates how machine learning can support food demand prediction and waste management. It aims to reduce food wastage and improve inventory planning through data-driven insights.

## Developed For
Machine Learning Mini Project  
B.Tech – CSE (AI & ML)