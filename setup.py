# -*- coding: utf-8 -*-

from setuptools import setup

#with open('README.rst') as readme_file:
#    readme = readme_file.read()

requirements = [
    'face_recognition_models',
    'sklearn',
    'dlib',
    'numpy',
    'scikit-image',
    'imblearn'
]

setup(
    name='expression_detector',
    version='0.1.2',
    description="Recognize face expressions from Python or from the command line",
    author="Aleksei Krikunov",
    author_email='alexey.v.krikunov@yandex.ru',
    packages=[
        'expression_recognition',
    ],
    package_dir={'expression_recognition': 'expression_recognition'},
    package_data={
        'expression_recognition': ['models/*.sav']
    },
    entry_points={
        'console_scripts': [
            'expression_detector=expression_recognition.expression_detector:main'
        ]
    },
    install_requires=requirements
)