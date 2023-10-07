import os
import ast
import subprocess

def has_module_docstring(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        tree = ast.parse(file.read(), filename=file_path)
        return ast.get_docstring(tree) is not None

def get_files_with_docstrings(dir_path):
    files_with_docstrings = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                if has_module_docstring(file_path):
                    files_with_docstrings.append(file_path)
    return files_with_docstrings

def remove_undesired_rsts(rst_dir, files_with_docstrings):
    desired_rsts = {os.path.splitext(os.path.relpath(f, rst_dir))[0] + '.rst' for f in files_with_docstrings}
    for rst in os.listdir(rst_dir):
        if rst not in desired_rsts:
            os.remove(os.path.join(rst_dir, rst))

# identify files with docstrings
examples_path = "../examples"
files_with_docstrings = get_files_with_docstrings(examples_path)

# generate .rst files
output_path = "examples_docs"
result = subprocess.run(["sphinx-apidoc", "-o", output_path, examples_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
print("STDOUT:", result.stdout.decode())
print("STDERR:", result.stderr.decode())

# remove undesired .rst files
# remove_undesired_rsts(output_path, files_with_docstrings)

