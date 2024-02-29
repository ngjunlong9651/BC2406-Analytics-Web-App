import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
st.write("This project was prepared by Group 2 For BC2406")
image = Image.open('ntu-placeholder-d.jpg')

st.image(image,caption='A practical application of Random Forests models',use_column_width=True)

df = pd.read_csv('Cleveland, Hungary, Switzerland, and Long Beach.csv')
x= df.iloc[:,:-1] 
y = df.iloc[:,13]
x_train,x_test,y_train,y_test =train_test_split(x,y,random_state=42, test_size=0.3,shuffle=True)
# sex,cp,trestbps,chol,fbs,restecg,
# thalach,exang,oldpeak,slope,ca,thal,target
def get_user_input():
    age = st.slider('What is your age? ',20,100)
    sex = st.slider('What is your sex? 0 for Male, 1 for Female', 0, 1)
    cp = st.slider('What is your chest pain type? 0 for Typical Anginal, 1 for Atypical Angina, 2 for Non-Anginal, 3 for Non-Anginal Chest Pain',0,3)
    trestbps = st.slider('What is your resting blood pressure? ',0,190)
    chol = st.slider('What is your cholesterol: ',100,400)
    fbs = st.slider('Is your blood sugar more than 120? 0 for No, 1 for Yes: ',0,1)
    restecg = st.slider('What is your resting ECG? ',0,2)
    thalach = st.slider('What is your maximum heart rate? ',0,200)
    exang = st.slider('Do you have exercise induced angina? 0 for No, 1 for Yes: ',0,1)
    oldpeak = st.slider('What is your ST depression induced by exercise relative to rest: ', 0.0,5.0)
    slope = st.slider('What is your slope of peak ST segment? ',0,2)
    ca = st.slider('What is your number of major vessels (0-3) colored by flourosopy',0,4)
    thal = st.slider('What is your thal value? 1 for Normal, 2 for Fixed Defect, 3 for Reversable Defect:-',1,3)
    
    user_data = {'age':age,
                'sex':sex,
                'cp':cp,
                'trestbps':trestbps,
                'chol':chol,
                'fbs':fbs,
                'restecg':restecg,
                'thalach':thalach,
                'exang':exang,
                'oldpeak':oldpeak,
                'slope':slope,
                'ca':ca,
                'thal':thal
    }

    features = pd.DataFrame(user_data, index=[0])
    return features
user_input = get_user_input()
st.subheader('user input:')
st.write(user_input)

from sklearn.ensemble import RandomForestClassifier
dt = RandomForestClassifier(
    n_estimators=100,  # The number of trees in the forest. More trees can improve accuracy but increases computational cost.
    criterion='gini',  # Criterion to measure the quality of a split ('gini' for Gini impurity and 'entropy' for information gain).
    max_depth=None,  # Maximum depth of the tree. If None, nodes are expanded until all leaves are pure or contain less than min_samples_split samples.
    min_samples_split=20,  # The minimum number of samples required to split an internal node.
    min_samples_leaf=10,  # The minimum number of samples required to be at a leaf node.
    max_features='auto',  # The number of features to consider when looking for the best split. 'auto' means max_features=sqrt(n_features).
    bootstrap=True,  # Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree.
    oob_score=True,  # Whether to use out-of-bag samples to estimate the generalization accuracy.
    random_state=42,  # Controls both the randomness of the bootstrapping of the samples and the sampling of the features.
    n_jobs=-1,  # The number of jobs to run in parallel. -1 means using all processors.
)
dt.fit(x_train,y_train)

predict = dt.predict(user_input)

st.subheader('Classification:')
st.write(predict)

if predict==0:
    st.write('The model predicts that you do not have heart disease. Continue to monitor your symptoms and edit this page if you feel worse')
else:
    st.write("Please go to NHCS, this model predicts that you have heart disesase! We wish you all the best!")