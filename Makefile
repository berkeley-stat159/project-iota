.PHONY: all clean coverage test

all: clean

clean:
	find . -name "*.so" -o -name "*.pyc" -o -name "*.pyx.md5" | xargs rm -f

coverage:
	nosetests code/utils/tests data/tests --with-coverage --cover-package=data  --cover-package=utils

test:
	nosetests code/utils/tests/ data/tests

verbose:
	nosetests -v code/utils data
