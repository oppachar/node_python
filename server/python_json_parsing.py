import json

# with를 이용해 파일을 연다.
# json 파일은 같은 폴더에 있다고 가정!

with open('student_file.json') as json_file:
    json_data = json.load(json_file)