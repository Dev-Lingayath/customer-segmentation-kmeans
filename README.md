# Customer Segmentation Using K-Means Clustering

## Overview
This project implements the K-Means clustering algorithm to segment retail customers based on their purchasing behavior.  
Customer segmentation helps businesses understand customer patterns, optimize marketing strategies, and improve decision-making.

The project demonstrates a complete machine learning workflow, including data preprocessing, feature scaling, cluster selection, model training, and visualization.

---

## Objective
- Segment customers into distinct groups based on spending behavior
- Identify high-value and low-value customer segments
- Visualize customer clusters for business insights

---

## Dataset
- **Source:** Kaggle  
- **Dataset Name:** Customer Segmentation Tutorial in Python  
- **File Used:** `Mall_Customers.csv`  

Dataset link:  
https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

---

## Features Used
- Annual Income (k$)
- Spending Score (1–100)

---

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Git
- GitHub
- Visual Studio Code

---

## Methodology
1. Load and inspect the dataset
2. Select relevant numerical features
3. Scale features using StandardScaler
4. Determine optimal number of clusters using the Elbow Method
5. Apply K-Means clustering
6. Assign cluster labels to customers
7. Visualize clusters using scatter plots

---

## Results
- Customers are grouped into distinct segments based on income and spending behavior
- Clear separation of clusters is observed
- The model provides meaningful insights for targeted marketing and customer analysis

---

## How to Run
1. Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
