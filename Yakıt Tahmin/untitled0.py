import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import norm,skew

from sklearn.preprocessing import RobustScaler,StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

#XGBoost
import xgboost as xgb 

#uyarı 
import warnings
warnings.filterwarnings('ignore')

column_name = ["MGP", "Cylinders", "Displacement", "Horsepower", "Weight", "Acceleration", "Model Year", "Origin"]
data = pd.read_csv("auto-mpg.data", names=column_name, na_values= "?", comment= "\t", sep = " ",skipinitialspace = True)

data.rename(columns = {"MPG":"target"})

print(data.head())
print("Data shape:",data.shape)

data.info()

describe = data.describe()

# %% kayıp değerler
print(data.isna().sum())

data["Horsepower"] = data["Horsepower"].fillna(data["Horsepower"].mean()) 
"""
horspowerdeki boştaki mean değerleri doldur
"""
print(data.isna().sum())

sns.displot(data.Horsepower)

# %%

#veri analizleri

corr_matrix = data.corr()
sns.clustermap(corr_matrix, annot = True, fmt = ".2f")
plt.title("özellikler arasındaki korelasyon")
plt.show()

threshold = 0.75
filtre = np.abs(corr_matrix["MGP"])>threshold
corr_features = corr_matrix.columns[filtre].tolist() 
sns.clustermap(data[corr_features].corr(), annot = True, fmt = ".2f")
plt.title("özellikler arasındaki korelasyon")
plt.show()


sns.pairplot(data, diag_kind = "kde", markers = "+")
plt.show()

"""
cylinders ve origin kategörik olabilir
"""


plt.figure()
sns.countplot(data["Cylinders"])
print(data["Cylinders"].value_counts())

plt.figure()
sns.countplot(data["Origin"])
print(data["Origin"].value_counts())

#boxplot

for c in data.columns:
    plt.figure()
    sns.boxplot(x = c, data = data, orient = "v")
"""
alt ve üst çizgi dışındaki noktalar outlier = horsepower ve accelaration
"""

# %%
"""
IQR HESAPLAMA  veri kaybı çok olması istenmediği için treshold 2 olarak belirlendi ve outlier datanın içerisinden çıkarıldı
"""
thr = 2

horsepower_desc = describe["Horsepower"]

q3_hp = horsepower_desc[6]
q1_hp = horsepower_desc[4]
IQR_hp = q3_hp - q1_hp
top_limit_hp = q3_hp + thr * IQR_hp
bottom_limit_hp = q1_hp - thr * IQR_hp
filter_hp_bottom = bottom_limit_hp < data["Horsepower"]
filter_hp_top = data["Horsepower"] < top_limit_hp
filter_hp = filter_hp_bottom & filter_hp_top

data = data[filter_hp]  



acceleration_desc = describe["Acceleration"]

q3_acc = acceleration_desc[6]
q1_acc = acceleration_desc[4]
IQR_acc = q3_acc - q1_acc
top_limit_acc = q3_acc + thr * IQR_acc
bottom_limit_acc = q1_acc - thr * IQR_acc
filter_acc_bottom = bottom_limit_acc < data["Acceleration"]
filter_acc_top = data["Acceleration"] < top_limit_acc
filter_acc = filter_acc_bottom & filter_acc_top

data = data[filter_acc] 
# %%
# çarpıklık değeri

# bağımlı değişken

""" mu:mean  sigma:standart sapma
"""

sns.distplot(data.MGP,fit = norm)

(mu,sigma) = norm.fit(data["MGP"])
print("mu: {}, sigma = {}".format(mu,sigma))
 
#quantile quantile plot
plt.figure()
stats.probplot(data["MGP"],plot = plt)
plt.show()

data["MGP"] = np.log1p(data["MGP"])
plt.figure()
sns.distplot(data.MGP,fit = norm)


plt.figure()
stats.probplot(data["MGP"],plot = plt)
plt.show()

#bağımsız değişken

skewed_feats = data.apply(lambda x: skew(x.dropna())).sort_values(ascending = False)
skewness = pd.DataFrame(skewed_feats, columns = ["çarpıklık"])

# %% one hot encoding

data["Cylinders"] = data["Cylinders"].astype(str)
data["Origin"] = data["Origin"].astype(str)

data = pd.get_dummies(data)

# %% Standartizasyon ve Test

#Split
x = data.drop(["MGP"], axis = 1)
y = data.MGP

test_size = 0.9
X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size = test_size, random_state = 42)
  
#Standartizasyon

scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

"""
mean 0 standart sapması 1 olarak ayarlandı
"""
# %% Regrasyon modelleri

# lineer regrasyon

lr = LinearRegression()
lr.fit(X_train,Y_train)
print("LR. Coef: ",lr.coef_)
y_predicted_dummy = lr.predict(X_test)
lr_mse = mean_squared_error(Y_test,y_predicted_dummy)
print("Lineer Regrasyon MSE: ",lr_mse)

# Ridge Regrasyon

ridge = Ridge(random_state = 42, max_iter = 10000)
alphas = np.logspace(-4,-0.5,30)

tuned_parameters = [{'alpha':alphas}]
n_folds = 5

"""
alfa değerini mutlaka seçmek gerekir bunun için cross validation kullannıldı
"""

clf = GridSearchCV(ridge, tuned_parameters, cv = n_folds, scoring = "neg_mean_squared_error",refit = True)
clf.fit(X_train,Y_train)
scores = clf.cv_results_["mean_test_score"]
scores_std = clf.cv_results_["std_test_score"]

print("Ridge Coef: ",clf.best_estimator_.coef_)
rigde = clf.best_estimator_
print("Rigde Best Estimator: ",rigde)


y_predicted_dummy = clf.predict(X_test)
ridge_mse = mean_squared_error(Y_test,y_predicted_dummy)
print("Rigde Mse: ",ridge_mse)
print("----------------------------------------------------------")
 
"""
alpha çizilirken logspace kullanıldığı için burda da kullanılmalıdır
"""
plt.figure()
plt.semilogx(alphas,scores)
plt.xlabel("alpha")
plt.ylabel("scores")
plt.title("Rigde")

#Lasso Regrassion (L1)

lasso = Lasso(random_state = 42, max_iter = 1000)
alphas = np.logspace(-4,-0.5,30)

tuned_parameters = [{'alpha':alphas}] 
n_folds = 5

clf = GridSearchCV(lasso, tuned_parameters, cv = n_folds, scoring = "neg_mean_squared_error",refit = True)
clf.fit(X_train,Y_train)
scores = clf.cv_results_["mean_test_score"]
scores_std = clf.cv_results_["std_test_score"]

print("Lasso Coef: ",clf.best_estimator_.coef_)
lasso= clf.best_estimator_
print("Lasso Estimator: ",lasso)

y_predicted_dummy = clf.predict(X_test)
lasso_mse = mean_squared_error(Y_test,y_predicted_dummy)
print("Lasso Mse: ",lasso_mse)
print("----------------------------------------------------------")
 

plt.figure()
plt.semilogx(alphas,scores)
plt.xlabel("alpha")
plt.ylabel("scores")
plt.title("Lasso")

#Elastic Net

parametersGrid = {"alpha":alphas, "l1_ratio": np.arange(0.0, 1.0, 0.05)}
eNet = ElasticNet(random_state = 42, max_iter = 1000)
clf = GridSearchCV(eNet, parametersGrid, cv = n_folds, scoring = "neg_mean_squared_error",refit = True)
clf.fit(X_train,Y_train)

print("ElasticNet Coef: ",clf.best_estimator_.coef_)
eNet = clf.best_estimator_
print("ElasticNet En iyi Estimator: ",eNet)

y_predicted_dummy = clf.predict(X_test)
elastic_mse = mean_squared_error(Y_test,y_predicted_dummy)
print("ElasticNet: ",elastic_mse)


# %% XGBoost
parametersGrid = {'nthreat':[4],
                  'objective':['reg:linear'],
                  'learnin_rate':[.03, 0.05, .07],
                  'max_depth':[5,6,7],
                  'min_child_weight':[4],
                  'silent':[1],
                  'subsample':[0.7],
                  'colsample_bytree':[0.7],
                  'n_estimators':[500,1000]}
model_xgb = xgb.XGBRFRegressor()

clf =GridSearchCV(model_xgb, parametersGrid, cv = n_folds, scoring = "neg_mean_squared_error",refit = True, n_jobs=5) 
clf.fit(X_train,Y_train)
model_xgb = clf.best_estimator_


y_predicted_dummy = clf.predict(X_test)
xgb_mse = mean_squared_error(Y_test,y_predicted_dummy)
print("XGBRegressor MSE:",xgb_mse)

# %% Ortalama Model  
class AveragingModels():
 def __init__(self,models):
     self.models = models
       
 def fit(self, X, y):
     self.models_ = [clone(x) for x in self.models]
     
     for model in self.models_:
         model.fit(X, y)
         
         return self
     
 def predict(self, X):
     predictions = np.column_stack([model.predict(X) for model in self.models_])
     return np.mean(predictions, axis = 1)
 
averaged_models = AveragingModels(models = (model_xgb , lasso))
averaged_models.fit(X_train, Y_train)

y_predicted_dummy = averaged_models.predict(X_test)
mse = mean_squared_error(Y_test, y_predicted_dummy)
print("Ortalama Model MSE:",mse)



#%% Tüm Modellerin gösterimi

model_gosterim = np.array([lr_mse,ridge_mse,lasso_mse,elastic_mse,xgb_mse]).reshape(1,5)
model_gosterim

df = pd.DataFrame(model_gosterim, columns = ["Lineer Regression","Ridge Regression","Lasso Regression","Elastic Net","XgbBoost"])
df.head()

