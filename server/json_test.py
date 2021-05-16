import json
from result import *

faceline_index = str(faceline_index)
up = str(up)
center = str(center)
low = str(low)
ratio = str(ratio)
cheek_side = str(cheek_side)
cheek_front = str(cheek_front)

analysis = {
    "FaceAnalysis": [
        {
            "Faceline": faceline_index, 
            "Ratio": [  ratio,
                        up, 
                        center, 
                        low],
            "Cheek": [  cheek_front, 
                        cheek_front]
        }
    ]
} 

with open("test_file.json", "w") as json_file:
    json.dump(analysis, json_file)