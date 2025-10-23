## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.

STEP 2:Clean the Data Set using Data Cleaning Process.

STEP 3:Apply Feature Encoding for the feature in the data set.

STEP 4:Apply Feature Transformation for the feature in the data set.

STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding

An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.

2. Label Encoding

Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.

3. Binary Encoding

Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.

4. One Hot Encoding

We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:

  # 1. FUNCTION TRANSFORMATION

• Log Transformation

• Reciprocal Transformation

• Square Root Transformation

• Square Transformation

  # 2. POWER TRANSFORMATION

• Boxcox method

• Yeojohnson method

# CODING AND OUTPUT:


import pandas as pd 
import numpy as np 

df = pd.read_csv("Encoding Data.csv")
df

<img width="235" height="315" alt="image" src="https://github.com/user-attachments/assets/c963fe4f-fe8b-46bc-ab97-2f5495029751" />

from sklearn.preprocessing import LabelEncoder,OrdinalEncoder

pm = ["Hot","Warm","Cold"]

e1 = OrdinalEncoder(categories=[pm])

e1.fit_transform(df[["ord_2"]])

<img width="105" height="167" alt="image" src="https://github.com/user-attachments/assets/66bd9715-6292-4678-8f38-d28fdab1d85f" />

le = LabelEncoder()

dfc = df.copy()

dfc["ord_2"] = le.fit_transform(dfc['ord_2'])

dfc

<img width="241" height="324" alt="image" src="https://github.com/user-attachments/assets/bc7abb93-001a-48b4-965f-884de9b0bad4" />


from sklearn.preprocessing  import OneHotEncoder
import pandas as pd
ohe = OneHotEncoder(sparse_output = False)
df2 = df.copy()

<img width="121" height="317" alt="image" src="https://github.com/user-attachments/assets/9db73d98-248e-4697-9098-c1e183732d13" />


enc = pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
enc

<img width="124" height="315" alt="image" src="https://github.com/user-attachments/assets/73db2803-1df0-44b8-bc0f-8f40ff2432ea" />


df2 = pd.concat([df2,enc],axis = 1)'

<img width="312" height="311" alt="image" src="https://github.com/user-attachments/assets/8b4c8c29-3715-4737-a061-1e47d30f8386" />

pd.get_dummies(df2,columns=["nom_0"])

<img width="513" height="323" alt="image" src="https://github.com/user-attachments/assets/711d5ee6-34e8-492e-8217-e5032416f63e" />

import pandas as pd 
from scipy import stats
import numpy as np 
df = pd.read_csv("Data_To_Transform.csv")
df

<img width="610" height="385" alt="image" src="https://github.com/user-attachments/assets/35127329-e2b3-4ab7-87e0-ebf6a8162d9a" />

df.skew()

<img width="252" height="92" alt="image" src="https://github.com/user-attachments/assets/5e034494-77c9-48ac-9c2e-c5a24737b4f2" />

np.log(df["Highly Positive Skew"])

<img width="402" height="207" alt="image" src="https://github.com/user-attachments/assets/17f4f1eb-9d7c-400c-b506-23bedeb09a99" />

np.reciprocal(df["Moderate Positive Skew"])

<img width="410" height="196" alt="image" src="https://github.com/user-attachments/assets/b467504d-e753-4d14-b736-188ba5d298f7" />

np.sqrt(df["Highly Positive Skew"])

<img width="396" height="198" alt="image" src="https://github.com/user-attachments/assets/8f6779c0-70e8-4cff-bf3e-5b43a726cbcd" />

np.square(df["Highly Positive Skew"])

<img width="395" height="196" alt="image" src="https://github.com/user-attachments/assets/7d613be0-1e99-4656-b208-02e257424b7e" />

df["Highly Positive Skew_boxcox"],parameters = stats.boxcox(df["Highly Positive Skew"])


<img width="935" height="360" alt="image" src="https://github.com/user-attachments/assets/51392458-8a1e-4ec2-8e79-00bd912b70d2" />

df.skew()

<img width="281" height="117" alt="image" src="https://github.com/user-attachments/assets/331d2d3c-142f-482b-a00b-89719082ec49" />

df.skew()

<img width="275" height="114" alt="image" src="https://github.com/user-attachments/assets/ed385885-8e1e-4b5f-b014-5132e8d443ba" />


import pandas as pd 
from scipy import stats
import numpy as np 
df = pd.read_csv("Data_To_Transform.csv")
from sklearn.preprocessing import QuantileTransformer 
qt = QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew"] = qt.fit_transform(df[["Moderate Negative Skew"]])
df


<img width="621" height="380" alt="image" src="https://github.com/user-attachments/assets/7a096fcc-5ff6-47f2-8826-c83faf64b538" />

import pandas as pd 
import statsmodels.api as sm
import matplotlib.pyplot as plt 

data = pd.read_csv('Data_to_Transform.csv')

df = pd.DataFrame(data)
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()

<img width="787" height="532" alt="image" src="https://github.com/user-attachments/assets/98b8b6cd-0003-492b-aef6-74850c1e298a" />


import numpy as np
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')


<img width="781" height="532" alt="image" src="https://github.com/user-attachments/assets/e75e8845-a7b6-4bf0-a090-ab26f1d69ba2" />


from sklearn.preprocessing import QuantileTransformer


qt = QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"] = qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"], line='45')
plt.show()


<img width="768" height="532" alt="image" src="https://github.com/user-attachments/assets/ede5d02f-3777-4b23-9ab0-1a6fa68d1441" />




df["Highly Negative Skew_1"] = qt.fit_transform(df[["Highly Negative Skew"]])

sm.qqplot(df["Highly Negative Skew"],line='45')

plt.show()


<img width="768" height="532" alt="image" src="https://github.com/user-attachments/assets/96a2c6fa-5a45-41d4-9750-b6079f0e983c" />


import pandas as pd 
from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution='normal',n_quantiles=981)

df["Moderate Negative Skew"]= qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"], line='45')
plt.show()

<img width="764" height="532" alt="image" src="https://github.com/user-attachments/assets/235cf779-b7ad-463c-a0a6-7b1399e187ae" />

import pandas as pd

dt = pd.read_csv("titanic_dataset.csv")

from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution='normal',n_quantiles=891)

dt["Age_1"] = qt.fit_transform(dt[["Age"]])

sm.qqplot(dt["Age"],line='45')

plt.show()    

<img width="768" height="532" alt="image" src="https://github.com/user-attachments/assets/094846fb-5934-432e-a4c2-cdce7cd8914c" />


       # INCLUDE YOUR CODING AND OUTPUT SCREENSHOTS HERE
# RESULT:
       # INCLUDE YOUR RESULT HERE

       
