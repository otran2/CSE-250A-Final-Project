# EM Model Documentation #

## Data Preprocessing ##
1. Split dataset into relevant features and categories pd data frames
2. BMI values are continuous; pgmpy expects discretization. Utilize bins to categorize BMI into underweight, normal weight, overweight, obese, and extremely obese categories. Checked for BMI category sparsity--bins appear to be fine.
3. Shift Age, Education, and Income to start from zero; pgmpy expects integer labels to start from zero.
4. Target classes are imbalanced; Class 0 consists of 218,334 samples and Class 1 consists of 35,346 samples.
5. Applied SMOTE to account for target class imbalance.
6. Per TA office hours, 21 features may be too much. Applied feature selection to identify most impactful features and dropped the rest.

## Feature Selection ##
The objective is to select the 15 most relevant features for model training. 
1. Used mutual information to determine 15 features with highest dependency.
2. Applied RFE (recursive elimination) to eliminate least important features until only 15 remain. 

## Develop Bayesian Network ##
1. Use pgmpy to develop Bayesian network and perform inference.

## Model Evaluation ##
1. Evaluated model performance using accuracy and precision scores.

## EM CPT ##
![CPT](sganti-EM_CPT.png)
