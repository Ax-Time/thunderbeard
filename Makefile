build:
	python setup.py bdist_wheel

test:
	python setup.py pytest

upload:
	twine upload dist/thunderbeard-0.1.0-py3-none-any.whl

upload-verbose:
	twine upload --verbose dist/thunderbeard-0.1.0-py3-none-any.whl