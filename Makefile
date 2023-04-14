.ONESHELL:
SHELL = /bin/bash

create_environment :
	source /srv/conda/etc/profile.d/conda.sh
	conda env create -f environment.yml 
	conda activate notebook
	conda install ipykernel
	python -m ipykernel install --user --name make-env --display-name "IPython - Make"

.PHONY: html
html:
	jupyter-book build .

.PHONY: html-hub
html-hub: conf.py
	sphinx-build  . _build/html -D html_baseurl=${JUPYTERHUB_SERVICE_PREFIX}proxy/absolute/8000
	@echo
	@echo "To start the Python http server, use:"
	@echo "python -m http.server --directory ${PWD}/_build/html"
	@echo "and visit this link with your browser:"
	@echo "https://stat159.datahub.berkeley.edu/user-redirect/proxy/8000/index.html"

conf.py: _config.yml _toc.yml
	jupyter-book config sphinx .


.PHONY: clean
clean:
    rm -rf figures
    rm -rf audio