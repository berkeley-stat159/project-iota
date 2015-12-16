.PHONY: all clean coverage test

all: 
	cd data && make
	cd code/utils/conv_response/ && python convo_response_script.py task001_run001 && python convo_response_script.py task003_run001 && python combine_convo_point_script.py task001_run001
	cd code/utils/linear_modeling/ && python block_linear_modeling_script.py task001_run001 && python full_linear_modeling_script.py task001_run001 && python full_dct_linear_modeling_script.py task001_run001 python full_linear_modeling_script.py task003_run001
	cd code/utils/linear_modeling/ && python ANOVA_test.py task001_run001 && python normal_assumption_script.py task001_run001 && python comparing_two_back.py	
	cd paper && make
	make clean
	
clean:
	find . -name "*.so" -o -name "*.pyc" -o -name "*.pyx.md5" | xargs rm -f

dataset:
	cd data && make

convo:
	cd code/utils/conv_response/ && python convo_response_script.py task001_run001 && python convo_response_script.py task003_run001 && python combine_convo_point_script.py task001_run001
	
modeling:
	cd code/utils/linear_modeling/ && python block_linear_modeling_script.py task001_run001 && python full_linear_modeling_script.py task001_run001 && python full_linear_modeling_script.py task003_run001 && python full_dct_linear_modeling_script.py task001_run001

testing:
	cd code/utils/linear_modeling/ && python ANOVA_test.py task001_run001 && python normal_assumption_script.py task001_run001 && python comparing_two_back.py	
	
report:
	cd paper && make

coverage:
	nosetests code/utils/tests data/tests --with-coverage --cover-package=data  --cover-package=utils

test:
	nosetests code/utils/tests data/tests

verbose:
	nosetests -v code/utils data



