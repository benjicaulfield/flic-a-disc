import json
import pathlib

page = 'new_1.json'

HERE = pathlib.Path(__file__).parent
path = HERE / page

def load_page(path):
    with open(path, "r") as file:
        content = file.read()
        one = content[0]
        print(one)

if __name__ == "__main__":
    load_page(path)

