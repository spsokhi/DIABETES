from django.shortcuts import render
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

def diabetes_predict(request):
    if request.method == 'POST':
        pre = float(request.POST.get('pre'))
        glu = float(request.POST.get('glu'))
        blo = float(request.POST.get('blo'))
        ski = float(request.POST.get('ski'))
        ins = float(request.POST.get('ins'))
        bmi = float(request.POST.get('bmi'))
        dia = float(request.POST.get('dia'))
        age = float(request.POST.get('age'))

        path = "/Users/spsokhi/Desktop/AiML/DIABETES/diabetes_project/diabetes.csv"
        data = pd.read_csv(path)
        print(data)
        print(data.info())

        le = LabelEncoder()
        data['Outcome'] = le.fit_transform(data['outcome'])

        inputs = data[['pregnancies','glucose','bloodpressure','skinthickness','insulin','bmi','diabetespedigreefunction','age']]
        output = data['outcome']

        model = DecisionTreeClassifier()
        model.fit(inputs,output)

        prediction = model.predict([[pre,glu,blo,ski,ins,bmi,dia,age]])
        return render(request,'diabetes.html',{'prediction':prediction[0]})
    else:
        return render(request,'diabetes.html')
