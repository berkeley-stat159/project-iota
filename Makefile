.PHONY: all clean coverage test

all: clean

clean:
	find . -name "*.so" -o -name "*.pyc" -o -name "*.pyx.md5" | xargs rm -f

coverage:
	nosetests code/utils/tests data/tests --with-coverage --cover-package=data  --cover-package=utils

test:
	nosetests code/utils/tests/data/tests

verbose:
	nosetests -v code/utils data

dataset:
	cd data
	make

sample:
	python code/utils/simple_plot.py task001_run001

convo:
	python code/utils/conv_response/convo_response_script.py task001_run001
	python code/utils/conv_response/convo_response_script.py task003_run001
	python code/utils/conv_response/combine_convo_point_script.py task001_run001
	python code/utils/conv_response/combine_convo_point_script.py task003_run001

modeling:
	python code/utils/linear_modeling/block_linear_modeling_script.py task001_run001
	python code/utils/linear_modeling/block_linear_modeling_script.py task003_run001
	python code/utils/linear_modeling/full_linear_modeling_script.py task001_run001
	python code/utils/linear_modeling/full_linear_modeling_script.py task003_run001
	python code/utils/linear_modeling/full_dct_linear_modeling_script.py task001_run001
	
diagnostic:
	python code/utils/linear_modeling/ANOVA_test.py
	python code/utils/linear_modeling/normal_assumption_script.py
	python code/utils/linear_modeling/comparing_two_back.py

