import os

ALLOWED_FILES = ["Makefile", "mkdocs.yml", "pyproject.toml", "README.md"]
def find_and_replace(what_to_replace, replace_value, directory="."):
    for dname, dirs, files in os.walk(directory):
        for fname in files:
            fpath = os.path.join(dname, fname)
            if fname in ALLOWED_FILES:
                print("Replacing ", fname)
                try:
                    with open(fpath, 'r', encoding='utf-8') as f:
                        s = f.read()
                    s = s.replace(what_to_replace, replace_value)
                    with open(fpath, "w") as f:
                        f.write(s)
                except UnicodeDecodeError:
                    print("passing", fname)

if __name__ == "__main__":
    package_name = input("Package Name: ")
    python_package_name = input("Python Package Name: ")
    print(package_name, python_package_name)

    find_and_replace("PYTHON_PATH_NAME", python_package_name)
    find_and_replace("PACKAGE_NAME", package_name)
    os.rename("rename_me_pls", python_package_name)