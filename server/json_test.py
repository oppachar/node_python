import json
from face_detection import *

low = str(low);

student_data = {
    "1.low": low,
    "2.LastName": "Hong",
    "3.Age": 20, 
    "4.University": "Yonsei University",
    "5.Courses": [
        {
            "Major": "Statistics", 
            "Classes": ["Probability", 
                        "Generalized Linear Model", 
                        "Categorical Data Analysis"]
        }, 
        {
            "Minor": "ComputerScience", 
            "Classes": ["Data Structure", 
                        "Programming", 
                        "Algorithms"]
        }
    ]
} 

with open("student_file.json", "w") as json_file:

    json.dump(student_data, json_file)
