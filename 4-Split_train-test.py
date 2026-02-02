import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

housing = pd.read_csv(Path("datasets/housing/housing.csv"))
housing['income_cat'] = pd.cut(housing['median_income'],
                               bins=[0.,1.5,3.0,4.5,6.,np.inf],
                               labels=[1,2,3,4,5])  


train_set , test_set = train_test_split(housing, 
                                        random_state=42, 
                                        test_size=0.2,
                                        stratify=housing["income_cat"])

print("\n Splitting Done! \n")

print(train_set.head(),"\n")

for set in (train_set, test_set):
    set.drop("income_cat", axis=1, inplace=True)

print("\n Dropping Done! \n")

print(train_set.head(),"\n")

try : 

    if not Path("data").is_dir():
        Path("data").mkdir(parents=True,exist_ok=True)

    train_set.to_csv('data/train.csv',index=False)
    test_set.to_csv('data/test.csv',index=False)

    print('Both Datasets are saved in data!')

except Exception as e :
    print(f"error : {e}")