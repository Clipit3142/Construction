import os
import numpy as np
import argparse
from scripts.analysis import run_analysis
#from scripts.pipeline import run_pipeline
from scripts.download import run_download


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="Fetch and download data.")
    parser.add_argument("--analysis", action="store_true", help="Analyze coorelation of train data.")
    parser.add_argument("--train", action="store_true", help="Train all models on dataset.")
    parser.add_argument("--pipelinee", action="store_true", help="Reformat train data.")
    args = parser.parse_args()
    if args.analysis:
        run_analysis()
    if args.download:
        url = "https://www.ine.gob.cl/docs/default-source/permisos-de-edificacion/cuadros-estadisticos/series-mensuales/series-históricas-a-mayo-2026.xls?sfvrsn=e96da874_4"
        run_download(url)
    if args.train:
        pass

    
    

