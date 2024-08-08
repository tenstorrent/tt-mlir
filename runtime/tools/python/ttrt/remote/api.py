import requests
import json
import os
import zipfile
import io
from ttrt.remote.util import getFlatbuffers, saveArtifacts
import re

server_url = 'http://172.27.45.166:8080'  # Replace with your server URL

def remote_set(args):
    print(args)
    print("remote_set")

def remote_get(args):
    print(args)
    print("remote_get")

def remote_run(args):
    fbs = getFlatbuffers(args.binary)
    command = "ttrt run"
    command += f" --program-index {args.program_index}"
    # for extensibility
    # for key, value in args.items():
    #     command += f" --{key} {value}"

    # add support for all
    # if int(args.program_index) != 0:
    
    # Used by client side to insert flatbuffer into the position of the placeholder
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

    if args.save_artifacts:
        saveArtifacts(response)
    else:
        print(response.json()['output'])

def remote_query(args):
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
    
    if args.save_artifacts:
        saveArtifacts(response)
    else:
        print(response.json()['output'])
    
def remote_perf(args):
    fbs = getFlatbuffers(args.binary)
    command = "ttrt perf"
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
    
    if args.save_artifacts:
        saveArtifacts(response)
    else:
        print(response.json()['output'])

def remote_download(args):
    response = requests.get(args.url)

    filepath = "download.ttnn"
    if args.n != None:
        filepath = args.n
    
    if response.status_code == 200:
        with open(filepath, 'wb') as file:
            file.write(response.content)
        print(f'File downloaded successfully as {filepath}')

def remote_upload(args):
    command = "ttrt upload"
    json_data = {"command": command}
    file = {args.binary: open(args.binary ,'rb')}
    response = requests.post(server_url + "/upload", files = file, data={'json': json.dumps(json_data)})
    print(response)
    print(response.json()['output'])
    
def remote_create(args):
    print(args)
    print("REMOTE CREATE")