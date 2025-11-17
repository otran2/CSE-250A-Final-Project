from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np 
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import ExpectationMaximization
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#use trained model if set to true
useSaved = True

# fetch dataset 
cdc_diabetes_health_indicators = fetch_ucirepo(id=891) 
  
# data (as pandas dataframes) 
X = cdc_diabetes_health_indicators.data.features 
y = cdc_diabetes_health_indicators.data.targets 

#drop unused data
X = X.drop(columns=["GenHlth", "PhysHlth", "MentHlth"])

#Data cleaning (so it plays nice with pgmpy)
#convert BMI into underweight, normal, overweight, obese, and extremely obese (respectively 0 through 4)
X.loc[:, "BMI"] = pd.cut(
    X.loc[:, "BMI"],
    bins=[0, 18.5, 25, 30, 35, float("inf")],
    labels=[0, 1, 2, 3, 4]
).astype(int)

#shift age to be 0 through 12 instead of 1 through 13
X.loc[:, "Age"] = X.loc[:, "Age"] - 1

#shift education to be 0 through 5 instead of 1 through 6
X.loc[:, "Education"] = X.loc[:, "Education"] - 1

#shift income to be 0 through 7 instead of 1 through 8 
X.loc[:, "Income"] = X.loc[:, "Income"] - 1

#split data into train-test-validate
X_train, X_test_validate, y_train, y_test_validate = train_test_split(X, y, train_size=0.8, random_state=42)
X_test, X_validate, y_test, y_validate = train_test_split(X_test_validate, y_test_validate, train_size=0.5, random_state=42)

if useSaved == False:
    #establish BN 
    model = DiscreteBayesianNetwork(
        [
            #variables connecting to physical health latent variable
            ("HighBP", "PhysicalHealth"),
            ("HighChol", "PhysicalHealth"),
            ("BMI", "PhysicalHealth"), 
            ("HeartDiseaseorAttack", "PhysicalHealth"),
            ("Stroke", "PhysicalHealth"),
            ("PhysActivity", "PhysicalHealth"),
            ("DiffWalk", "PhysicalHealth"),

            #varaibles connecting to diet latent variable
            ("Fruits", "Diet"),
            ("Veggies", "Diet"),
            ("HvyAlcoholConsump", "Diet"),
            ("Smoker", "Diet"),

            #variables connecting to healthcare access latent variable
            ("AnyHealthcare", "HealthcareAccess"),
            ("NoDocbcCost", "HealthcareAccess"),
            ("CholCheck", "HealthcareAccess"),

            #variables connecting to socioeconomic status latent variable
            ("Sex", "SocioeconomicStatus"),
            ("Age", "SocioeconomicStatus"),
            ("Education", "SocioeconomicStatus"),
            ("Income", "SocioeconomicStatus"),

            #latent variables connecting to final diagnosis
            ("PhysicalHealth", "Diabetes_binary"),
            ("Diet", "Diabetes_binary"),
            ("HealthcareAccess", "Diabetes_binary"),
            ("SocioeconomicStatus", "Diabetes_binary"),
            #("SurveyResponses", "Diabetes_binary"),
        ],
        latents={"PhysicalHealth", "Diet", "HealthcareAccess", "SocioeconomicStatus"} 
    )

    #get cardinalities from data
    cardinalities = {col: X[col].nunique() for col in X.columns} #cardinalities taken from number of unique values in data
    cardinalities.update({"Diabetes_binary" : 2}) #No diabetes or prediabetes, no diabetes or prediabetes
    cardinalities.update({"PhysicalHealth" : 3}) #bad, neutral, good
    cardinalities.update({"Diet" : 3}) #bad, neutral, good
    cardinalities.update({"HealthcareAccess" : 3}) #None, poor, good 
    cardinalities.update({"SocioeconomicStatus" : 3}) #Lower class, middle class, upper class

    #generate CPTs for each node
    cpts = []
    for var in model.nodes(): 
        var_card = cardinalities[var]
        parents = model.get_parents(var) #read the parents of the node

        #variable has no parents
        if len(parents) == 0:
            probs = np.random.dirichlet(np.ones(var_card)) #uniform distribution initialization
            cpts.append(TabularCPD(variable = var, variable_card = var_card, values = np.transpose([probs])))
        #variable has parents
        else:
            parent_cards = [cardinalities[parent] for parent in parents] #get the cardinalities of each parent of the node into a list
            num_parent_configs = np.prod(parent_cards)

            probs = np.zeros((var_card, num_parent_configs))

            for col in range(num_parent_configs): #define a uniform distribution for each parent configuration
                probs[:, col] = np.random.dirichlet(np.ones(var_card))

            cpts.append(TabularCPD(variable = var, variable_card = var_card, values = probs, evidence=parents, evidence_card = parent_cards))

    model.add_cpds(*cpts)

    #check model for faults
    model.check_model()

#train EM 
if useSaved == False: 
    train_combined = X_train.join(y_train) #combine X_train and y_train together
    small_train_combined = train_combined.sample(20000)
    model.fit(data=small_train_combined, estimator=ExpectationMaximization)

    model.save("model_ben.bif", filetype="bif")
else: 
    model = DiscreteBayesianNetwork.load("model_ben.bif", fyletype ="bif")

model_infer = VariableElimination(model)

#test set
preds_test = []

for _, row in X_test.iterrows():
    evidence = {k: str(v) for k, v in row.to_dict().items()} #convert from int to str because pgmpy is fussy
    q = model_infer.query(["Diabetes_binary"], evidence=evidence)
    preds_test.append(int(q.values.argmax()))
 
print(accuracy_score(y_test, preds_test))

#validation set
preds_val = []

for _, row in X_validate.iterrows():
    evidence = {k: str(v) for k, v in row.to_dict().items()} #convert from int to str because pgmpy is fussy
    q = model_infer.query(["Diabetes_binary"], evidence=evidence)
    preds_val.append(int(q.values.argmax()))

print(accuracy_score(y_validate, preds_val))