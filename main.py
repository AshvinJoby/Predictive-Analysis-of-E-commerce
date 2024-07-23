import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.feature_selection import SelectKBest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

# Read from excel file
df = pd.read_excel('Dataset.xlsx', sheet_name = 'E Comm')
df.head()
dt=df.copy()
df.describe()
df.info()

#Boxplot to visualize distribution
sns.set_theme(style="darkgrid", palette="muted")
fig, ax = plt.subplots(figsize=(16,7))
sns.boxplot(data=df).set_title("Box Plot to Visualize Distribution")
plt.xticks(rotation=45)
plt.show()

#Handling outliers
cat = df.select_dtypes(include='object').columns
num = list(df.select_dtypes(exclude='object').columns)
num.remove('Churn')
for cols in num:
    Q1 = df[cols].quantile(0.25)
    Q3 = df[cols].quantile(0.75)
    IQR=Q3-Q1 #interquartile range
    lr= Q1-(1.5 * IQR)
    ur= Q3+(1.5 * IQR)
    df[cols] = df[cols].mask(df[cols]<lr, lr, )
    df[cols] = df[cols].mask(df[cols]>ur, ur, )

#fill missing values with median/mean of respective columns
df.isnull().sum()
df.fillna({'Tenure' : df.Tenure.median()}, inplace=True)
df.fillna({'WarehouseToHome' : df.WarehouseToHome.median()}, inplace=True)
df.fillna({'HourSpendOnApp' : df.HourSpendOnApp.median()}, inplace=True)
df.fillna({'OrderAmountHikeFromlastYear' : round(df.OrderAmountHikeFromlastYear.mean())}, inplace=True)
df.fillna({'CouponUsed' : df.CouponUsed.median()}, inplace=True)
df.fillna({'OrderCount' : df.OrderCount.median()}, inplace=True)
df.fillna({'DaySinceLastOrder' : df.DaySinceLastOrder.median()}, inplace=True)
df.isnull().sum()

#Categorical Analysis
for col in cat:
    print(df[col].value_counts())
fig,ax = plt.subplots(nrows=2,ncols=3,figsize=(26,12))
fig.suptitle("Count plot for Categorical Analysis")
for col,subplot in zip(cat, ax.flatten()):
  sns.countplot(x = df[col], hue=df.Churn, ax=subplot) #count plots
fig, ax = plt.subplots(2, 3, figsize=(30, 20))
fig.suptitle("Pie Charts for Categorical Analysis")
plt.rcParams['font.size'] = '16'
for col,subplot in zip(cat, ax.flatten()):
    temp = df.groupby(by=df[col]).Churn.sum()
    total = df.value_counts(col).sort_index()
    res1 = temp/total*100
    subplot.pie(labels = res1.index, x = res1.values, autopct='%.0f%%',textprops={'fontsize': 16}) #pie charts

#Numerical Analysis
for i, subplot in zip(num, ax.flatten()):
    sns.histplot(df[i], kde = True, ax=subplot)
fig, ax = plt.subplots(2, 6, figsize=(30, 10))
fig.suptitle("Line graphs for Numerical Analysis")
for col,subplot in zip(num, ax.flatten()):
    temp = df.groupby(by=df[col]).Churn.sum()
    total = df.value_counts(col).sort_index()
    res1 = temp/total*100
    sns.lineplot(x = res1.index, y = res1.values, ax=subplot, ) #line graphs
    subplot.tick_params(axis='x')

#Heatmap for correlation matrix
numeric_df = df.select_dtypes(include=[np.number])
mask=np.zeros_like(numeric_df.corr())
mask[np.triu_indices_from(mask)] = True
fig, ax = plt.subplots( figsize=(16, 7))
sns.heatmap(numeric_df.corr(method='pearson'), mask=mask, cmap='rainbow').set_title("Heatmap for Correlation Matrix")

#Encoding categorical data
enc = LabelEncoder()
for col in df.select_dtypes(include='object'):
    df[col]=enc.fit_transform(df[col])

#Split data into training and testing sets
X_train, X_test,y_train, y_test = train_test_split(df.drop('Churn', axis=1), df.Churn)

#Class for wrapping a base estimator
class my_classifier(BaseEstimator,):
    def __init__(self, estimator=None):
        self.estimator = estimator
    def fit(self, X, y=None):
        self.estimator.fit(X,y)
        return self
    def predict(self, X, y=None):
        return self.estimator.predict(X,y)
    def predict_proba(self, X):
        return self.estimator.predict_proba(X)
    def score(self, X, y):
        return self.estimator.score(X, y)

#Creation of pipeline    
pipe = Pipeline([('scaler', StandardScaler()), ('clf', my_classifier())])
#Parameters for grid search
parameters = [
              {'clf':[LogisticRegression(max_iter=1000)],
               'clf__C':[0.001,0.01,.1,1],
               'clf__solver':['lbfgs','liblinear']
               },
             {'clf':[RandomForestClassifier()],
             'clf__criterion':['gini','entropy'],
             },
             {
               'clf':[DecisionTreeClassifier()],
             'clf__criterion':['gini','entropy'],
             },
             {
              'clf':[XGBClassifier()],
             'clf__learning_rate':[0.01,0.1,0.2,0.3],
             'clf__reg_lambda':[0.01,0.1,1],
             'clf__reg_alpha': [0.01,0.1,0,1],
             }]

grid = GridSearchCV(pipe, parameters, cv=5)
grid.fit(X_train,y_train)
grid.best_estimator_
grid.best_score_
y_pred = grid.predict(X_test,)

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True).set_title("Heatmap for Confusion Matrix and F1 Score")
#confusion matrix determines performance of the operation
print(f1_score(y_test,y_pred))

feature_array = grid.best_estimator_[-1].feature_importances_ #array of feature importances
importance = dict(zip(df.drop('Churn',axis=1).columns,feature_array))
importance = dict(sorted(importance.items(), key= lambda item:item[1],reverse = True) )
fig, ax = plt.subplots(figsize=(26,8))
sns.barplot(x=list(importance.keys()), y=list(importance.values())).set_title("Bar Graphs for Feature Importances") #bar graphs for feature importances
plt.tick_params(axis='x', labelrotation=45)
plt.show()
