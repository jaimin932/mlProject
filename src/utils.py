import os
import sys
sys.path.insert(0,'C:/Users/Hp/Desktop/Data Analyst/mlProject/src')

import exception as CustomException
import numpy as np 
import pandas as pd
import pickle
import dill
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
