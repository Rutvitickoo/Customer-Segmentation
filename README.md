# Customer-Segmentation
# Customer Segmentation using K-Means Clustering

This project demonstrates how to group customers based on their purchasing behavior and demographics using unsupervised machine learning (K-Means Clustering). The goal is to help businesses effectively target marketing strategies based on customer segments.

---

## 📌 Features

- Loads customer demographic and behavioral data from a CSV file
- Preprocesses and encodes features
- Scales features using Standard Scaler
- Uses the Elbow Method to determine the optimal number of clusters
- Applies K-Means clustering to segment customers
- Visualizes clusters using PCA (Principal Component Analysis)
- Outputs cluster-specific summaries
- Saves the final segmented data into a CSV

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

---

## 📁 Dataset Format

The input CSV file (`customers.csv`) should contain the following columns:

| CustomerID | Age | Gender | Annual Income (k$) | Spending Score (1-100) |
|------------|-----|--------|--------------------|-------------------------|
| 1          | 19  | Male   | 15                 | 39                      |

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/customer-segmentation.git
cd customer-segmentation
