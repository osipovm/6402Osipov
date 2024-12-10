from setuptools import setup

setup(
    name='data_analysis',
    version='0.1',
    description='Time-series analysis',
    author='Maksim Osipov',
    author_email='osipovm01@mail.ru',
    packages=['operations'],
    install_requires=[
        'numpy',
        'pandas',
    ],
)