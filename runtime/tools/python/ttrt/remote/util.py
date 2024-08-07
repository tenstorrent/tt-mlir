import os
import re
import json
import zipfile

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

def saveArtifacts(response):
    extract_dir = "extracted"
    content_type = response.headers['Content-Type']
    # print(content_type)
    boundary = re.search(r'boundary=(.*)', content_type).group(1)

    # Split the response content by the boundary
    parts = response.content.split(f'--{boundary}'.encode())

    for part in parts:
        # Each part will have headers followed by content, separated by two newlines
        if b'\r\n\r\n' in part:
            headers, content = part.split(b'\r\n\r\n', 1)

            # Handle JSON part
            if b'application/json' in headers:
                json_data = json.loads(content.decode('utf-8'))
                print("JSON Data:", json_data['output'])

            # Handle ZIP file part
            elif b'application/zip' in headers:
                # Extract the filename from the Content-Disposition header
                disposition_header = re.search(r'filename="(.+)"', headers.decode('utf-8'))
                if disposition_header:
                    zip_file_path = 'artifacts'
                    with open(zip_file_path, 'wb') as f:
                        f.write(content)
                    with zipfile.ZipFile('artifacts', 'r') as zip_ref:
                        zip_ref.extractall(extract_dir)
                        print(f"ZIP file extracted to '{extract_dir}'.")
                    os.remove(zip_file_path)