p="C:/Users/opolot/Desktop/patient records/build_notebook.py"
s=open(p).read()
s=s.replace("df = pd.read_csv('heart.csv')","df = pd.read_csv('heart.csv')\ndf = df.rename(columns={'age':'Age','sex':'Sex','cp':'ChestPainType','trestbps':'RestingBP','chol':'Cholesterol','thalach':'MaxHR','exang':'ExerciseAngina','target':'HeartDisease'})")
s=s.replace("cat_cols = ['Sex','ChestPainType','ExerciseAngina']","cat_cols = ['ChestPainType','thal','slope','restecg']")
open(p,"w").write(s)
print("ok")
