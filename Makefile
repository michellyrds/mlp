
.PHONY: environment
environment:
	pyenv install -s 3.8.0
	pyenv virtualenv 3.8.0 mlp
	pyenv local mlp

.PHONY: scratch-requirements
scratch-requirements:
	pip install -Ur from_scratch/requirements.txt
	pip install -Ur from_scratch/requirements.lint.txt

.PHONY: demo
demo: 
	PYTHONPATH=$(shell pwd) python from_scratch/main.py
