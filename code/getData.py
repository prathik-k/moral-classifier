import pandas as pd 
import numpy as np 
import requests 
from tqdm import tqdm
import os


def getDataset():
    url = "https://www.dropbox.com/s/qjmj4wq9ywz5tb7/clean_data.csv?dl=1"
    fname = "temp_data.csv"


    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(fname, 'wb') as f:
            pbar = tqdm(total=int(r.headers['Content-Length']))
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    aita_data = pd.read_csv("temp_data.csv")
    os.remove("temp_data.csv")
    
    return aita_data
