import os
import sys

from time import sleep

import yaml
import subprocess

import logging
from argparse import ArgumentParser


logger = logging.getLogger("TRANSFER_FILES")
logger.setLevel(logging.DEBUG)

POD_NAME = "file-transfer-pod"
POD_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "./pod_config.yaml")
POD_CONFIG = {
    "apiVersion": "v1",
    "kind": "Pod",
    "metadata": {
        "name": POD_NAME
    },
    "spec": {
        "containers": [
            {
                "name": "alpine",
                "image": "alpine:latest",
                "command": [
                    'tail', 
                    '-f', 
                    '/dev/null'
                ],
            }
        ]
    },
}

def log_potential_error(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except:
            logger.exception("Error")
    return wrapper


def init_logging_config():
    console_format = logging.Formatter('[%(name)s][%(levelname)s] - %(message)s')
    file_format = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] - %(message)s')

    console_handler = logging.StreamHandler(); console_handler.setLevel(logging.INFO); console_handler.setFormatter(console_format)
    file_handler = logging.FileHandler(os.path.join(os.path.dirname(__file__), "./move_files_logs.log"), mode="w"); file_handler.setLevel(logging.DEBUG); file_handler.setFormatter(file_format)

    logger.addHandler(console_handler); logger.addHandler(file_handler)


@log_potential_error
def parse_args(argv):
    argv.pop(0)

    parser = ArgumentParser()
    parser.add_argument("-cf", "--config_file", action="store", dest="config_path", required=True, type=str, help="Path to the YAML config file")

    args = parser.parse_args(argv)

    return args


def generate_volume_mount(volume):
    return {
        "name": volume,
        "mountPath": "/{v}".format(v=volume)
   }


def generate_volume_entry(volume):
    return {
        "name": volume,
        "persistentVolumeClaim": {
            "claimName": volume
        }
   }


@log_potential_error
def generate_pod_config(config):
    volume_entries, volume_mounts = [], []
    for volume in config["volumes"]:
        volume_entries.append(generate_volume_entry(volume["name"]))
        volume_mounts.append(generate_volume_mount(volume["name"]))

    POD_CONFIG["spec"]["containers"][0]["volumeMounts"] = volume_mounts
    POD_CONFIG["spec"]["volumes"] = volume_entries


@log_potential_error
def run_ps_command(command):
    return subprocess.run(["powershell", "-Command", command], capture_output=True, text=True)


@log_potential_error
def launch_pod():
    output = run_ps_command("kubectl apply -f {config}".format(config=POD_CONFIG_PATH))
    logger.debug("Pod launch return {ret}".format(ret=output))

    return output.returncode


@log_potential_error
def transfer_files(config):
    command_pull = "cd {path}; kubectl cp {podname}:{vol}/{src} {dest}"
    command_push = "cd {path}; kubectl cp {src} {podname}:{vol}/{dest}"

    for volume in config["volumes"]:
        for file in volume["files"]:
            if os.path.isabs(file["source"]):
                path, src = os.path.split(file["source"])
                dest = file["destination"]
                cmd = command_push.format(path=path, podname=POD_NAME, vol=volume["name"], src=src, dest=dest)
            else:
                path, dest = os.path.split(file["destination"])
                src = file["source"]
                cmd = command_pull.format(path=path, podname=POD_NAME, vol=volume["name"], src=src, dest=dest)

            output = run_ps_command(cmd)
            
            if output.returncode == 0:
                logger.info("Kubectl cp command pass {ret}".format(ret=output))
            else:
                logger.critical("Copy files failed for command: {cmd}".format(cmd=cmd))
                logger.critical("Command output: {out}".format(out=output))


@log_potential_error
def delete_pod():
    output = run_ps_command("kubectl delete pod {podname}".format(podname=POD_NAME))
    logger.debug("Pod delete return {ret}".format(ret=output))

    return output.returncode


@log_potential_error
def main():
    init_logging_config()
    logger.info("Logging file succesfully init")
    
    args = parse_args(sys.argv)
    logger.info("Argument parsing successful!")

    with open(args.config_path, "r") as file:
        config = yaml.safe_load(file)
    logger.info("Found config file!")

    generate_pod_config(config)
    with open(POD_CONFIG_PATH, "w") as file:
        yaml.safe_dump(POD_CONFIG, file, sort_keys=False)
    logger.info("Generate pod config successful!")

    output = launch_pod()
    assert output == 0, "Could not launch pod!"
    sleep(60)
    logger.info("Pod launch attempt sucessful!")
    
    transfer_files(config)
    logger.info("File transfering finished!")
    
    delete_pod()
    assert output == 0, "Could not delete pod!"
    logger.info("Pod deletion sucessful!")


if __name__ == "__main__":
    main()