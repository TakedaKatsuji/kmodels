import os
from pathlib import Path
import click

def make_file(file_name:str):
    p = Path(f"{file_name}")
    p.touch()   


@click.command()
@click.option("--model_name", type=str)
def make_model_folder(model_name):
    os.makedirs(f"./{model_name}/jupyter")
    os.makedirs(f"./{model_name}/output")
    for file_name in [f"./{model_name}/dataset.py", f"./{model_name}/models.py", f"./{model_name}/utils.py"]:
        
        make_file(file_name)

if __name__=="__main__":
    make_model_folder()

