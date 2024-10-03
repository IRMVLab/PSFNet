# -- coding: utf-8 --

import json

def json2dict(path: str) -> dict:
    with open(path, 'r', encoding='utf8') as f:
        content = json.load(f)
    return content

def dict2json(data: dict, path: str):
    with open(path, 'w', encoding='utf8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)






