# Political Candidate Education Prediction

## Overview

This project predicts the education level of political candidates using a Random Forest Classifier. It analyzes various candidate attributes like criminal record, total assets, party affiliation, and state to make predictions.

## Prerequisites

Ensure you have the following installed:

- Python 3.x
- Required Python libraries: pandas, matplotlib, seaborn, scikit-learn

## File Description

- `train.csv`: Training dataset containing information about political candidates.
- `test.csv`: Test dataset for predicting education levels.
- `my_submission_rf_improved_2.csv`: CSV file with predicted education levels for the test dataset.

## Code Explanation

1. **Data Preprocessing**:
   - Load and clean the training and test datasets.
   - Convert 'Total Assets' values to numeric format for analysis.
   - Perform basic data exploration and visualization to understand the data.

2. **Feature Engineering**:
   - Create new features based on candidate names and constituency preferences to improve model performance.

3. **Data Encoding**:
   - Encode categorical variables into numeric format using LabelEncoder for model compatibility.

4. **Model Training**:
   - Split the training data into training and validation sets for model evaluation.
   - Train a Random Forest Classifier on the training data to predict education levels.

5. **Prediction**:
   - Use the trained model to make predictions on the test set.
   - Convert numeric predictions back to the original education levels for interpretation.

6. **Output**:
   - Save the predictions to a CSV file for further analysis and submission.

## Instructions

1. **Clone the Repository**
git clone <repository_url>
2. **Install Dependencies**:
pip install pandas matplotlib seaborn scikit-learn
3. **Run the Code**:
python3 main.py 
4. **View Results**:
- After running the code, check the generated `my_submission_rf_improved_2.csv` file for the predicted education levels.

## Additional Notes

- Ensure that the training and test datasets are in the same directory as the script.
- Adjust hyperparameters of the Random Forest Classifier or try different machine learning models for experimentation.
- Explore additional feature engineering techniques for potentially improving model performance.

