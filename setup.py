from setuptools import find_packages, setup
setup(
    name='thunderbeard',
    packages=find_packages(include=[
        'thunderbeard'
    ]),
    version='0.1.0',
    description='Minimal python deep learning library',
    author='Ax-Time',
    license='MIT',
    install_requires=[
        'jax',
        'jaxlib',
        'numpy'
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==7.4.0'],
    test_suite='tests',
)