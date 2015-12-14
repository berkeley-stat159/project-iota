.PHONY: test coverage convo

test:
	nosetests utils

coverage:
	nosetests utils --with-coverage --cover-package=utils

convo:
	python ./utils/conv_response/convo_response_script.py task001_run001
