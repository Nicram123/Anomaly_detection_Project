# Credit Card Fraud Detection

Here you can find a detailed analysis and machine learning project focused on **credit card transaction anomaly detection**.  
The dataset contains real or simulated transaction records, where the main goal is to identify **fraudulent operations** among a vast majority of normal transactions.

The project has three main parts:

1. **Exploratory Data Analysis (EDA)**  
2. **Class Balancing and Feature Engineering**  
3. **Model Training and Evaluation**

---

## 🧩 Stack

**Pandas**, **NumPy**, **Matplotlib**, **Seaborn**, **Scikit-Learn**, **Imbalanced-learn (SMOTE)**, **XGBoost**, **TensorFlow**

---

## 📊 Exploratory Data Analysis (EDA)

In this section, I analysed general trends and correlations between different factors such as transaction amount, time, and balance.  
Several visualizations were prepared to better understand the structure and imbalance of the data.

### Main visualizations:
- **Distribution of Transaction Amounts**  
 <img width="1207" height="315" alt="image" src="https://github.com/user-attachments/assets/1a44b8da-974e-4c96-b28d-6a5a980f24aa" />
- **Correlation Heatmap**  
<img width="850" height="772" alt="image" src="https://github.com/user-attachments/assets/5d71f6d0-7d03-43b9-9754-50e3431a2d4c" />
- **Anomaly Score Visualization `confusion matrix`**  
 <img width="852" height="386" alt="image" src="https://github.com/user-attachments/assets/00dfe010-7354-45c1-bb7c-e394ecd1be0c" />


These plots allowed identifying which variables are most informative for detecting anomalies and which show noise or redundancy.

---

## ⚖️ Data Preprocessing and Class Balancing

Fraud detection datasets are typically **highly imbalanced**.  
To handle this, several resampling techniques were applied:

- **Random Under Sampling (RUS)** – reducing the majority class  
- **SMOTE (Synthetic Minority Oversampling Technique)** – generating synthetic minority samples  
- **NearMiss** – removing easy examples from the majority class  

Additionally:
- Data were **scaled** using `StandardScaler` and `RobustScaler`
- **Outliers** were removed based on statistical thresholds (IQR / z-score)
- **PCA** and **t-SNE** were used for dimensionality reduction and visualization

---

## 🤖 Model Training and Evaluation

Multiple machine learning models were trained and compared:

- **Logistic Regression**
- **Support Vector Machine (SVM)**
- **Decision Tree**
- **Random Forest**
- **XGBoost**

The training process was carefully evaluated using:
- **Precision**, **Recall**, and **F1-score**
- **ROC-AUC**
- **Precision-Recall curves**
- **Confusion Matrices**

A grid search was used to optimize hyperparameters for each model.  
Balanced and weighted versions of algorithms were tested to minimize bias toward the majority class.

---

## 🎯 Results and Observations

- The **SMOTE-balanced datasets** significantly improved recall (fraud detection rate) without sacrificing too much precision.  
- **XGBoost** and **Random Forest** performed best overall, achieving high AUC scores.  
- Weighted SVM also provided stable results, especially in recall-oriented settings.  
- Some models showed overfitting on the balanced data — this was investigated via learning curves.

---

## 🧠 Further Experiments

Additional experiments included:
- Threshold optimization for recall vs precision trade-off  
- Testing the impact of different scaling methods  
- Using **t-SNE** visualizations to observe separability of fraudulent vs non-fraudulent transactions  

---





