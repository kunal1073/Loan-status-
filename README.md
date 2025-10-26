Loan-status

LendingClub Loan Risk Prediction & Offline RL Decisioning
In this project, I present a complete pipeline for automated loan approval using LendingClub’s publicly available loan dataset. I combine deep learning for loan default prediction with an offline reinforcement learning agent to maximize expected profit. The workflow is fully reproducible and designed for easy deployment on platforms like Kaggle.

Project Steps
1. Exploratory Data Analysis (EDA) & Preprocessing
I loaded and inspected the raw LendingClub loan data, which includes over 2.2 million loans and 151 features.

I cleaned the dataset by removing leaks and features that are only available post-loan issuance.

I kept only applications labeled as "Fully Paid" (target=0) or with a defaulter status ("Charged Off"/"Default", target=1).

To handle missing data, I eliminated columns with more than 30% missing values, encoded categorical features, and scaled numeric columns using StandardScaler.

2. Deep Learning Model (DL) Training
I trained a multi-layer perceptron (MLP) to predict loan outcomes (default vs fully paid).

I used a robust train/validation/test split (70%/15%/15%) and ensured stratification by the target label.

I monitored metrics suitable for imbalanced data (AUC, F1-score, Precision, and Recall).

I addressed class imbalance using the class_weight parameter in Keras to penalize misclassifications of minority classes.

3. RL Environment & Offline Agent Training
I defined a tabular, single-step RL environment: applicant’s features (state), action space (approve/deny), and reward based on historical outcomes (+interest for fully paid loans, -principal for defaults).

I constructed an offline RL dataset using d3rlpy’s MDPDataset.

I trained a Conservative Q-Learning (CQL) agent to learn a policy that optimizes expected profit per loan.

4. Evaluation & Reporting
I presented the DL model’s AUC and F1 scores, and the RL agent’s estimated policy value as key evaluation metrics.

I compared the policies of both models and highlighted cases where the RL agent made counter-intuitive, but profitable, approvals.

I discussed the potential business impact, limitations of the approach, and recommended future steps.

Throughout the process, I produced visualizations including ROC curves, action distributions, and reward histograms.

How to Run the Project on Kaggle
Import the Dataset

Download LendingClub data (e.g. accepted_2007_to_2018.csv) from Kaggle.

Upload the CSV file to the Kaggle Notebook environment, or place it in /input/.

Kernel/Notebook Setup

Make sure all required libraries are installed. In your Kaggle notebook, run:

python
!pip install d3rlpy scikit-learn tensorflow
Execute the Pipeline

Step 1: Run my EDA and preprocessing notebook/script to save the cleaned CSV.

Step 2: Run the deep learning model notebook/script (train_deep_learning.py) for baseline classifier results.

Step 3: Run the RL agent notebook/script (train_rl_agent.py) to train the offline RL agent.

Be sure to check and update file paths to any processed CSVs, such as data/processed/loan_data_for_rl.csv.

Review & Save Outputs

Visualizations, model weights, action distributions, and key metrics will be automatically saved to the results/ and models/ directories.

You can use Kaggle’s output section to download the generated CSVs, PNGs, and model artifacts for final reporting and submission.

Project Requirements
Python 3.8+

Libraries: pandas, numpy, scikit-learn, matplotlib, tensorflow, d3rlpy

Main Scripts Included
exploratory_data_analysis.py — Loads data, cleans columns, and prepares features for ML.

train_deep_learning.py — Builds and trains the neural net classifier, then outputs test metrics.

train_rl_agent.py — Trains the offline RL agent, evaluates using policy value, and saves visualization graphs.

report.md or report.pdf — Summarizes all findings, model comparisons, and recommendations.

Contact & Attribution
Author: [Your Name]

Data source: LendingClub (via Kaggle)
