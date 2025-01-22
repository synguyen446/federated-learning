import os
import datetime as dt

PATH = r"C:\Users\syngu\Downloads"

def get_files():    
    return [x for x in os.listdir(PATH) if os.path.splitext(x)[-1] == ".png"]

def get_most_recent_file_path():
    img_path_list = get_files()
    most_recent_file_info = ["",0]

    for img_path in img_path_list:
        path = f"{PATH}\\{img_path}"
        if os.path.getmtime(path) > most_recent_file_info[-1]:
            most_recent_file_info[0] = img_path
            most_recent_file_info[-1] = os.path.getmtime(path)

    return f"{PATH}\\{most_recent_file_info[0]}"

