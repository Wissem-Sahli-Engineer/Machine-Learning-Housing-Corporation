import pandas as pd
from pathlib import Path
import tarfile
import urllib.request

tarball_path = Path("datasets/housing.tgz")

if not tarball_path.is_file():
    Path("datasets").mkdir(parents=True, exist_ok = True)
    
    url = "https://github.com/ageron/data/raw/main/housing.tgz"
    
    try : 
        urllib.request.urlretrieve(url, tarball_path)
        print("\n Downling Done! \n")

        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path= "datasets", filter='data')
    except Exception as e:
        print(f"\n Error: {e}\n")


df = pd.read_csv(Path("datasets/housing/housing.csv"))

print("\n",df.head(),"\n")

print(df.columns,"\n")

print(df.info(),"\n")

print(df.describe(),"\n")

print(df.value_counts("ocean_proximity"),"\n")