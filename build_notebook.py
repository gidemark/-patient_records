import nbformat as nbf
nb = nbf.v4.new_notebook()
cells = []

def md(s): return nbf.v4.new_markdown_cell(s)
def code(s): return nbf.v4.new_code_cell(s)

cells.append(md("# Heart Disease Prediction Analysis"))

cells.append(md("## Part (a)"))
cells.append(code("""import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
df = pd.read_csv('heart.csv')
df = df.rename(columns={'age':'Age','sex':'Sex','cp':'ChestPainType','trestbps':'RestingBP','chol':'Cholesterol','thalach':'MaxHR','exang':'ExerciseAngina','target':'HeartDisease'})
df = df.rename(columns={'age':'Age','sex':'Sex','cp':'ChestPainType','trestbps':'RestingBP','chol':'Cholesterol','thalach':'MaxHR','exang':'ExerciseAngina','target':'HeartDisease'})
print(df.head(10))
print('Shape:', df.shape)
print(df.dtypes)"""))
cells.append(code("""print('Missing values:'); print(df.isnull().sum())
for col in ['Cholesterol','RestingBP']:
    med = df.loc[df[col]!=0, col].median()
    df[col] = df[col].replace(0, med)
print(df[['Cholesterol','RestingBP']].describe())"""))
cells.append(code("""print(df['HeartDisease'].value_counts())
print('Balanced?' , 'Yes' if df['HeartDisease'].value_counts(normalize=True).min()>0.4 else 'Imbalanced')"""))

cells.append(md("## Part (b)"))
cells.append(code("""print(df.describe())"""))
cells.append(code("""num = df.select_dtypes(include=np.number)
plt.figure(figsize=(10,8)); sns.heatmap(num.corr(), annot=True, cmap='coolwarm'); plt.show()"""))
cells.append(code("""for col in ['Age','MaxHR','Cholesterol']:
    plt.figure(); sns.boxplot(x='HeartDisease', y=col, data=df); plt.show()"""))
cells.append(code("""cat_cols = ['ChestPainType','thal','slope','restecg']
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
print(df.head())"""))

cells.append(md("## Part (c)"))
cells.append(code("""from sklearn.model_selection import train_test_split
X = df.drop('HeartDisease', axis=1); y = df['HeartDisease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape)"""))
cells.append(code("""from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
for depth in [3,5,None]:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)
    acc = accuracy_score(y_test, dt.predict(X_test))
    print('max_depth', depth, 'accuracy', acc)"""))
cells.append(code("""from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
print('LR accuracy', accuracy_score(y_test, lr.predict(X_test)))"""))

cells.append(md("## Part (d)"))
cells.append(code("""from sklearn.metrics import confusion_matrix, classification_report
best_dt = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_train,y_train)
for name, model in [('DecisionTree',best_dt),('LogisticRegression',lr)]:
    pred = model.predict(X_test)
    print(name); print(classification_report(y_test,pred))
    cm = confusion_matrix(y_test,pred)
    sns.heatmap(cm, annot=True, fmt='d'); plt.title(name); plt.show()"""))
cells.append(code("""from sklearn.metrics import precision_score, recall_score, f1_score
rows=[]
for name,model in [('DT3',DecisionTreeClassifier(max_depth=3,random_state=42).fit(X_train,y_train)),
                   ('DT5',DecisionTreeClassifier(max_depth=5,random_state=42).fit(X_train,y_train)),
                   ('DTNone',DecisionTreeClassifier(max_depth=None,random_state=42).fit(X_train,y_train)),
                   ('LR',lr)]:
    p=model.predict(X_test)
    rows.append({'Model':name,'Accuracy':accuracy_score(y_test,p),'Precision':precision_score(y_test,p),
                 'Recall':recall_score(y_test,p),'F1':f1_score(y_test,p)})
print(pd.DataFrame(rows))"""))
cells.append(code("""from sklearn.tree import plot_tree
plt.figure(figsize=(16,10)); plot_tree(best_dt, feature_names=X.columns, class_names=['0','1'], filled=True, max_depth=3); plt.show()"""))

nb.cells = cells
with open("C:/Users/opolot/Desktop/patient records/heart_analysis.ipynb","w") as f:
    nbf.write(nb, f)
print("done")
