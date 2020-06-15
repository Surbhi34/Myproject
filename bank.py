import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVR
from sklearn import metrics
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, accuracy_score, classification_report, confusion_matrix




st.title("BANK CHURN ANALYSIS")
st.write("""
Using different Machine Learning Algorithms for Making Prediction whether person will stay or not in bank
""")
st.write("## Machine Learning Algorithm ")
classifier=st.selectbox("Select ML Algorithm or classifier ",("Decision Tree",'KNN',"Random Forest","Support Vector Machine"))

st.write("Algorithm Selected for Prediction is  ",classifier)
df = st.cache(pd.read_csv,allow_output_mutation=True)("churn.csv")
st.write("## Shape of Dataset")
st.write("Number of rows and columns contained in the following dataset are  ::: ")
st.write(df.shape)
st.write("## View")
st.write("IF you want to view dataset click on the checkbox ")
is_check = st.checkbox("Display Data")
if is_check:
    st.write(df)


df.Geography[df.Geography=='France']=1
df.Geography[df.Geography=='Spain']=2
df.Geography[df.Geography=='Germany']=3
df['Geography']=df['Geography'].astype('int')
st.write("## Credit Score")
creditscore=st.slider("Select Credit Score",350,900)
st.write("The selected credit score for the person is ",creditscore)
st.write("## Geography ")
Geography=st.selectbox("Select Geography",("Germany","France","Spain"))
st.write("Geography Selected is ",Geography)

def con(G):
    data=G
    if data=='France':
        val=1
        return val
    elif data=="Germany":
        val=3
        return val
    else:
        val=2
        return val
geo=con(Geography)
st.write("## Gender")
gen=st.radio("Gender",('Male','Female'))
st.write("Selected Gender is ",gen)
st.write("## Age ")
df.Gender[df.Gender == 'Male'] = 1
df.Gender[df.Gender == 'Female'] = 2
df['Gender']=df['Gender'].astype('int')
def convo(G):
    data=G
    if data=='Male':
        val=1
        return val
    else:
        val=2
        return val
gender=convo(gen)

age=st.slider("Age ",18,95)
st.write("Age selected is ",age)
st.write("## Tenure")
Tenure=st.slider("Tenure ",0,10)
st.write("Tenure is ",Tenure)
st.write("## Balance")
Balance=st.slider("Balance  ",0,260000)
st.write("Balance is ",Balance)
st.write("## Number of Products ")
pro=st.selectbox("Number of products ",('1','2','3','4'))
st.write("Number of Products Selected ",pro)
st.write("## Credit card ")
st.write(" 0 - Doesn't have credit card , 1 - Has Credit card ")
card=st.radio("Has Credit card  ",('0','1'))
st.write("value of credit card",card)
st.write("## Active Member")
st.write(" 0 - Not Active , 1 - Active Member ")
active=st.radio(" Person is Active Member or not ",('0','1'))
st.write(" Value of Active member is ",active)
st.write("## Estimated salary ")
salary=st.slider("Estimated salary ",10,200000)
st.write(" Estimated Salary of Person is ",salary)

df=df.drop(labels=['CustomerId','RowNumber','Surname'], axis=1,inplace=False)
X=df.iloc[:,:-1]
y=df.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


def pre(a,b,c,d,e,f,g,h,i,j,k):
    classs=k
    st.write("## Algorithm used here is ",classs)
    l=['Credit Score','Geography','Gender','Age','Tenure','Balance','Number of Products','Credit card','Active member','Estimated Salary']
    l1=pd.Series(l)
    l2=[a,b,c,d,e,f,g,h,i,j]
    l3=pd.Series(l2)
    l4=pd.concat([l1,l3],axis=1).set_axis(['Keys','Values'],axis=1)
    daa=pd.DataFrame(l4)
    st.write(daa)
    if classs =='Decision Tree':
         
         classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
         classifier.fit(X_train, y_train)
         result=classifier.predict([[a,b,c,d,e,f,g,h,i,j]])
         a=int(result)
         
         if a==0:
             st.write(" After anlysing entered information our model predict that  there are maximum chances of the person to stay in the bank")
         else:
             st.write(" After anlysing entered information our model predict that  there are maximum chances of the person to churn or Exit the bank")
         st.write("## Thank you ..")
    else:
          if classs=='KNN':
              nn = KNeighborsClassifier(5)
              nn.fit(X_train, y_train)
              result=nn.predict([[a,b,c,d,e,f,g,h,i,j]])
              a=int(result)
              if a==0:
                  st.write(" After anlysing entered information our model predict that  there are maximum chances of the person to stay in the bank")
              else:
                  st.write(" After anlysing entered information our model predict that  there are maximum chances of the person to churn or Exit the bank")
              st.write("## Thank you ..")
          else:
              if classs=='Random Forest':
                  regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
                  regressor.fit(X_train, y_train)
                  result=regressor.predict([[a,b,c,d,e,f,g,h,i,j]])
                  a=int(result)
                  if a==0:
                     st.write(" After anlysing entered information our model predict that  there are maximum chances of the person to stay in the bank")
                  else:
                      st.write(" After anlysing entered information our model predict that  there are maximum chances of the person to churn or Exit the bank")
                  st.write("## Thank you ..")
              else:
                    svm = SVR()
                    svm.fit(X_train, y_train)
                    result=svm.predict([[a,b,c,d,e,f,g,h,i,j]])
                    a=int(result)
                    if a==0:
                       st.write(" After anlysing entered information our model predict that  there are maximum chances of the person to stay in the bank")
                    else:
                       st.write(" After anlysing entered information our model predict that  there are maximum chances of the person to churn or Exit the bank")
                    st.write("## Thank you ..")
st.write("prediction whether a person with following details will stay of leave")
z=st.button("PREDICTION")
if z==True:
    pre(creditscore,geo,gender,age,Tenure,Balance,pro,card,active,salary,classifier)
else:
    pass
st.write("For any Querry contact -----> sharmasurbhi1999@yahoo.com")
st.write("contact ----->  9459976116")