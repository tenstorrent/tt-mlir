import requests
import json
import os
import zipfile
import io
from ttrt.remote.util import getFlatbuffers
import re

server_url = 'http://172.27.45.166:8080'  # Replace with your server URL

def remote_set(args):
    print(args)
    print("remote_set")

def remote_get(args):
    print(args)
    print("remote_get")

def remote_run(args):
    # print(args)
    
    fbs = getFlatbuffers(args.binary)
    command = "ttrt run"

    # for extensibility
    # for key, value in args.items():
    #     command += f" --{key} {value}"

    # add support for all
    if int(args.program_index) != 0:
        command += f" --program-index {args.program_index}"
    command += " placeholder"
    if int(args.loops) != 1:
        command += f" --loops {args.loops}"
    if args.clean_artifacts:
        command += " --clean-artifacts"
    if args.save_artifacts:
        command += " --save-artifacts"

    json_data = {"command": command}
    response = requests.post(server_url + "/run", files = fbs, data={'json': json.dumps(json_data)})
    print(response)

    extract_dir = "extracted"
    if args.save_artifacts:
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
                        filename = disposition_header.group(1)
                        zip_file_path = 'artifacts'
                        with open(zip_file_path, 'wb') as f:
                            f.write(content)
                        # print(f"ZIP file saved as {zip_file_path}")
                        with zipfile.ZipFile('artifacts', 'r') as zip_ref:
                            zip_ref.extractall(extract_dir)
                            print(f"ZIP file extracted to '{extract_dir}'.")
                        os.remove(zip_file_path)

        # print(json_data, zip_file_path)
        # with open('artifacts', 'wb') as f:
        #     f.write(response.content)

        # with zipfile.ZipFile('artifacts', 'r') as zip_ref:
            
        #     zip_ref.extractall(extract_dir)
        #     print(f"ZIP file extracted to '{extract_dir}'.")
    else:
        print(response.json()['output'])

def remote_query(args):
    # print(args)
    
    command = "ttrt query"
    if args.system_desc:
        command += " --system-desc"
    elif args.system_desc_as_json:
        command += " --system-desc-as-json"
    elif args.system_desc_as_dict:
        command += " --system-desc-as-dict"
    elif args.clean_artifacts:
        command += " --clean-artifacts"
    elif args.save_artifacts:
        command += " --save-artifacts"

    json_data = {"command": command}

    response = requests.post(server_url + "/query", json = json_data)
    print(response)
    print(response.json()['output'])

    # zip_file = zipfile.ZipFile(io.BytesIO(response.content))
    
    # # List the files in the ZIP archive
    # print("Files in the ZIP archive:")
    # print(zip_file.namelist())

    # # Extract all files to a directory
    # zip_file.extractall('extracted_files')
    # print("Files extracted to 'extracted_files' directory.")

def remote_perf(args):
    # print(args)

    fbs = getFlatbuffers(args.binary)
    command = "ttrt perf"
    if int(args.program_index) != 0:
        command += f" --program-index {args.program_index}"
    if int(args.device):
        command += " --device"
    command += " placeholder"
    if int(args.loops) != 1:
        command += f" --loops {args.loops}"
    if args.clean_artifacts:
        command += " --clean-artifacts"
    if args.save_artifacts:
        command += " --save-artifacts"

    json_data = {"command": command}
    response = requests.post(server_url + "/perf", files = fbs, data={'json': json.dumps(json_data)})
    print(response)
    print(response.json()['output'])

def remote_download(args):
    print(args)
    
    response = requests.get(args.url)

    filepath = "download.ttnn"
    if args.n != None:
        filepath = args.n
    
    if response.status_code == 200:
        with open(filepath, 'wb') as file:
            file.write(response.content)
        print(f'File downloaded successfully as {filepath}')

def remote_upload(args):
    # print(args)

    command = "ttrt upload"
    json_data = {"command": command}
    file = {args.binary: open(args.binary ,'rb')}
    response = requests.post(server_url + "/upload", files = file, data={'json': json.dumps(json_data)})
    print(response)
    print(response.json()['output'])
    
def remote_create(args):
    print(args)