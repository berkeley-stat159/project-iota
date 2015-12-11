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
	mkdir -p sub001/BOLD/task001_run001
	mkdir -p sub001/BOLD/task002_run001
	mkdir -p sub001/BOLD/task003_run001
	wget -r --no-parent --reject="index.html*" -e robots=off https://nipy.bic.berkeley.edu/rcsds/ds115/sub001/ --no-check-certificate 	  mv nipy.bic.berkeley.edu/rcsds/ds115/sub001/model/model001/task001_run001.feat/filtered_func_data_mni.nii.gz sub001/BOLD/task001_run001/
	mv nipy.bic.berkeley.edu/rcsds/ds115/sub001/model/model001/task002_run001.feat/filtered_func_data_mni.nii.gz sub001/BOLD/task002_run001/
	mv nipy.bic.berkeley.edu/rcsds/ds115/sub001/model/model001/task001_run003.feat/filtered_func_data_mni.nii.gz sub001/BOLD/task003_run001/
	rm -rf nipy.bic.berkeley.edu

validate:
	python data.py
