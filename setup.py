import setuptools
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="biophysics",
    version="0.2",
    author='Robert Harkness',
    author_email='rharksci@gmail.com',
    description='A package for managing and analyzing different types of experimental and simulated biophysical data.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/robharkness/biophysics',
    project_url = {
        "Bug Tracker": "https://github.com/robharkness/biophysics/issues"
    },
    license='GNU GPL v3.0',
    packages=find_packages(),
    install_requires=['numpy','scipy','nmrglue','matplotlib','pyyaml','pandas'],
)
