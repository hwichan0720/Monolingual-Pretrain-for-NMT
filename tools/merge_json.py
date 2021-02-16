import sys
import json


def load_json(json_file):
    json_open = open(json_file, 'r')
    token_id_dict = json.load(json_open)
    return token_id_dict

if __name__ == "__main__":
    joson_file1 = sys.argv[1]
    token_id_dict_l1 = load_json(joson_file1)
    
    joson_file2 = sys.argv[2]
    token_id_dict_l2 = load_json(joson_file2)

    for key, value in token_id_dict_l2.items():
        if key in token_id_dict_l1:
            token_id_dict_l2[key] = int(token_id_dict_l1[key])
        else:
            token_id_dict_l2[key] = int(token_id_dict_l2[key]) + len(token_id_dict_l1)
    
    for key, value in token_id_dict_l1.items():
        if key not in token_id_dict_l2:
            token_id_dict_l2[key] = int(token_id_dict_l1[key])

    with open(sys.argv[3], "w", encoding="utf-8") as f:
        json.dump(token_id_dict_l2, f)

