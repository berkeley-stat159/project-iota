data:
	wget http://openfmri.s3.amazonaws.com/tarballs/ds115_sub001-005.tgz
	tar -xvzf ds115_sub001-005.tgz sub001
	mv sub001/model/model001/onsets/ sub001/.
	rm -rf sub001/anatomy
	rm -rf sub001/behav
	rm -rf sub001/BOLD/*
	rm -rf sub001/dti
	rm -rf sub001/model
	rm -rf sub001/task_order.txt
	wget -np -e robot=off  

validate:
	python data.py
