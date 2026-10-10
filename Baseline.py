#%% CONFIG
from pathlib import Path

chdir = Path("G:\내 드라이브\04_Project\01_OIBC 2025\OIBC_2025_DATA")
cell = {'DataLoad' : True,
        'EDA' : True,
        'CDA' : True,
        'Model' : True,
        'Result' : True
        }

#%% DATA LOAD
import pandas as pd
col = ['nins', 'coord1', 'coord2']
pd.read_parquet(chdir / "train.parquet", columns=col)

if cell[0]==0:
    print("DataLoad 실행")

#%% EDA

#%% CDA

#%% MODEL

#%% RESULT

# 실제 경로 "G:\내 드라이브\04_Project\01_OIBC 2025\OIBC_2025_DATA\train.parquet"