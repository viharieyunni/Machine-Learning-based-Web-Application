import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import warnings
import pickle
warnings.filterwarnings("ignore")

heart_df=pd.read_csv("Coronary_heart_disease.csv")
heart_df.drop(['education'],axis=1,inplace=True)
heart_df.rename(columns={'male':'Sex_male'},inplace=True)
heart_df.dropna(axis=0,inplace=True)

new_features=heart_df[['age','Sex_male','cigsPerDay','totChol','sysBP','glucose','TenYearCHD']]
x=new_features.iloc[:,:-1]
y=new_features.iloc[:,-1]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size = 0.4, random_state=5)


logreg=LogisticRegression()
logreg.fit(x_train,y_train)

#y_pred = logreg.predict(X_test) 
#a = [35,0,195,106,77]
#int_features = [int(x) for x in a]
#final = [np.array(int_features)]
#prediction = log_reg.predict_proba(final)
#output = '{0:.{1}f}'.format(prediction[0][1], 2)
#print(output)

#inputt = [int(x) for x in "45 32 60".split(' ')]
#final = [np.array(inputt)]
#b = log_reg.predict_proba(final)

# Using pickle to send the model to a pkl file to use it later on in webapp
pickle.dump(logreg,open('model.pkl','wb'))

