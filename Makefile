
.PHONY: environment
environment:
	pyenv install -s 3.8.0
	pyenv virtualenv 3.8.0 mlp
	pyenv local mlp

.PHONY: requirements
requirements:
	pip install -Ur requirements.txt

.PHONY: demo
demo: 
	PYTHONPATH=$(shell pwd) python main.py
