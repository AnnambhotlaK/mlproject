from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(filepath: str) -> List[str]:
    '''
    Returns list of required packages from requirements.txt
    '''
    requirements = []
    with open(filepath) as file_obj:
        requirements = file_obj.readlines()
        # Replace newlines
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    
    return requirements



setup(
    name = 'mlproject',
    version = '0.0.1',
    author = 'Kishore',
    author_email = 'k4annam@gmail.com',
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')
)