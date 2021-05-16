import json
from features import *

up = str(upper_ratio)
center = str(center_ratio)
low = str(lower_ratio)

data1 = {
    "FaceAnalysis": [
        {
            "Major": "Statistics", 
            "Ratio": [up, 
                        center, 
                        low]
        }, 
        {
            "Minor": "ComputerScience", 
            "Classes": ["Data Structure", 
                        "Programming", 
                        "Algorithms"]
        }
    ]
} 

with open("test_file.json", "w") as json_file:

    json.dump(data1, json_file)