import requests
import json
import os
from ttrt.remote.util import getFlatbuffers

server_url = 'http://172.27.45.166:8080'  # Replace with your server URL

def remote_run(args):
    # print(args)
    
    fbs = getFlatbuffers(args.binary)
    command = "ttrt run"
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
    # print(args)
    
    response = requests.get(args.url)

    filepath = "download.ttnn"
    if len(args.n) > 0:
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