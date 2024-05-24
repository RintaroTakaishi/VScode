from ucimlrepo import fetch_ucirepo 
import pandas as pd
  
# fetch dataset 
adult = fetch_ucirepo(id=2) 
  
# data (as pandas dataframes) 
X = adult.data.features 
y = adult.data.targets 
  
print(X)
print(y)
df = pd.concat([X, y], axis=1)

df.to_csv(
    "../input/adult.csv", 
    sep=",",
    index=False, 
    header=True, 
    na_rep="NONE"
    )

# variable information 
print(adult.variables) 