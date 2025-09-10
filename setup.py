from setuptools import setup, find_packages


setup(
    name="dGbyG",
    version="0.1.0",
    author="Fan Wenchao",
    author_email="12133024@mail.sustech.edu.cn",
    description="dGbyG",
    url="https://gitub.com/f-wc/dGbyG", 

    package_dir={"": "src"},
    packages=find_packages(where="src"), 
)
