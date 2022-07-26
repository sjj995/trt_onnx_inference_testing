from xmlrpc.client import boolean
import argparse
import json
import pandas as pd
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()

    #config 파일로 상대오차 절대오차 정보 받아서 어떤 기준으로 돌렸는지 정보 나올 수 있도록 처리

    parser.add_argument(
        "--json_path",
        type=str,
        help="json data(=report) Path",
        default="/data2/user/sjj995/TensorRT/local/report.json"
    )

    parser.add_argument(
        "--origin_json_path",
        type=str,
        help="origin json data Path for concat",
        default=None
    )

    parser.add_argument(
        "--concat_json",
        type=boolean,
        help="(T/F) for concat json (default : False)",
        default=False
    )

    parser.add_argument(
        "--csv_path",
        type=str,
        help="csv data path(eg. directory/report.csv)",
        default=None
    )
    return parser.parse_args()


def print_dataframe(csv_path,origin_json_path,concat_flag,json_data):
    t_pd = pd.DataFrame(json_data).transpose()
    t_pd.columns=['Input Name','Input Shape','Output Name','Output Shape','abs Similarity','rel Similarity','Max Diff','Max Diff Pos','Unsupported Op']
    if concat_flag == False:
        print('[[New Report]]')
        print(t_pd)
        if csv_path is not None:
            t_pd.to_csv(csv_path)
    else:
        with open(origin_json_path,'r') as f:
            origin_data = json.load(f)
        
        final_data = dict(origin_data,**json_data)
        final_pd = pd.DataFrame(final_data).transpose()
        final_pd.columns=['Input Name','Input Shape','Output Name','Output Shape','abs Similarity','rel Similarity','Max Diff','Max Diff Pos','Unsupported Op']
        print('[[Final DataFrame]]')
        print(final_pd)
        if csv_path is not None:
            t_pd.to_csv(csv_path)
        with open(origin_json_path,'w') as f:
            json.dump(final_data,f,indent=4)

    return


def main(csv_path,json_path,origin_json_path,concat_flag):

    with open(json_path,'r') as f:
        json_data = json.load(f)

    print_dataframe(csv_path,origin_json_path,concat_flag,json_data)

    return

if __name__ == "__main__":

    args = get_args()
    csv_path = args.csv_path
    json_path = args.json_path
    origin_path = args.origin_json_path
    concat_flag = args.concat_json
    
    main(csv_path,json_path,origin_path,concat_flag)
