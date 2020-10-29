import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore", category = FutureWarning)
warnings.filterwarnings("ignore", category = DeprecationWarning)

tr_features = pd.read_csv("train_features.csv")
tr_labels = pd.read_csv("train_labels.csv")

#dont pay too much attention to this function
def print_results(results):
    print("BEST PARAMS: {}\n".format(results.best_params_))

    means = results.cv_results_["mean_test_score"]
    stds = results.cv_results_["std_test_score"]
    for mean, std, params in zip(means, stds, results.cv_results_["params"]):
        print("{} (+/-{}) for {}".format(round(mean,3), round(std*2, 3), params))


#Logistic regression is suitable for binary classifications
#here we will try to fit it for number of parameters using GridSearchCV and Cross Validation parameter cv
#the different parameter values will be stored in parameters dictionary in the form of a array whose
#key will be the actual parameter name that is passed in the LogisticRegression()


#load the model
lr = LogisticRegression()
#set the parameters
parameters = {
      "C":[100]
}
##    "C":[100,200,300,400,500,600,700,800,900,1000]
##} #this is a regularization param, when its value is high the regularization is low and chances of overfitting are high
#similarly when the value is low the chances of underfitting are high and regularization is high

cv = GridSearchCV(lr, parameters, cv = 5) #here we have kept the cross validation parameter as 5
cv.fit(tr_features, tr_labels.values.ravel())
#Our training labels are stored as a column vector type, but what scikit-learn really wants them to be is an array.
#So we're going to go ahead and convert this column vector to an array by just appending
#.values.ravel.

print_results(cv)

#print the best estimator features
print(cv.best_estimator_)

#store the model
joblib.dump(cv.best_estimator_, "LR_model.pkl")

