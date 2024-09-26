from setuptools import find_packages,setup
from typing import List


HYPHEN_E_DOT="-e ."
def get_requirements(file_path: str) -> List[str]:
    '''
    this function is make torun the library in requirements.txt file
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements = [req.strip() for req in requirements] 

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements



setup(
    name="student_prediction",
    version="0.0.1",  # Version should be a string
    author="Sharmi",
    author_email="introvertbeauty7@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)
