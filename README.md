# Loan-status

## LendingClub Loan Risk Prediction & Offline RL Decisioning

In this project, I present a complete pipeline for automated loan approval using LendingClub’s publicly available loan dataset. I combine deep learning for loan default prediction with an offline reinforcement learning agent to maximize expected profit. The workflow is fully reproducible and designed for easy deployment on platforms like Kaggle.

---

## 🚀 Project Steps

### 1. Exploratory Data Analysis (EDA) & Preprocessing

- Loaded and inspected the raw LendingClub data (over 2.2M loans, 151 features).
- Cleaned the dataset by removing data leaks and post-loan features.
- Retained only applications labeled as **"Fully Paid"** (`target=0`) or defaulter status like **"Charged Off"/"Default"** (`target=1`).
- Eliminated columns with more than 30% missing values, encoded categorical features, and scaled numeric columns using `StandardScaler`.

### 2. Deep Learning Model (DL) Training

- Trained a multi-layer perceptron (MLP) to predict loan outcomes (default vs fully paid).
- Utilized a stratified train/validation/test split (**70%** / **15%** / **15%**).
- Monitored metrics suitable for imbalanced data: **AUC**, **F1-score**, **Precision**, **Recall**.
- Addressed class imbalance using Keras’ `class_weight` parameter.

### 3. RL Environment & Offline Agent Training

- Defined a tabular, single-step RL environment:
    - **State**: applicant’s preprocessed features
    - **Action**: approve/deny
    - **Reward**: historical outcome (+interest for fully paid, -principal for default)
- Constructed an offline RL dataset using d3rlpy’s `MDPDataset`.
- Trained a Conservative Q-Learning (**CQL**) agent to learn profit-maximizing approval policy.

### 4. Evaluation & Reporting

- Presented DL model’s **AUC/F1** and RL agent’s **estimated policy value**.
- Compared policies and highlighted cases where RL made counter-intuitive (but profitable) approvals.
- Discussed business impact, limitations, and recommended future steps.
- Generated visualizations: ROC curves, action distributions, and reward histograms.

---

## 💻 How to Run the Project on Kaggle

### 1. Import the Dataset

- Download LendingClub data (e.g. `accepted_2007_to_2018.csv`) from [Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club).
- Upload to Kaggle Notebook or place in `/input/`.

### 2. Kernel/Notebook Setup

- Just run blocks till MLP training once and then seperately run DL Model Training bloack and then run the rest of the code.
- This is done to handle some version issues among some libraries.

### 3. Execute the Pipeline

### 4. Review & Save Outputs


- Use Kaggle’s output section to download generated CSVs, PNGs, and model artifacts for final reporting and submission.

---

## 📦 Project Requirements

- Python 3.8+
- Libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `tensorflow`, `d3rlpy`

---



---

## 👤 Contact & Attribution

- **Author:** *Kunal Choudhary*
- **Data:** LendingClub (via Kaggle)

---


