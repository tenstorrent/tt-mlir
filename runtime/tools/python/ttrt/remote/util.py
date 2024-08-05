import os

def getFlatbuffers(path):
    fbs = {}
    if os.path.isfile(path):
        fbs[path] =  open(path ,'rb')
    elif os.path.isdir(path):
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith(".ttnn"):
                    fbs[file] = open(os.path.join(root, file), 'rb')
    return fbs