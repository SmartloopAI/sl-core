.PHONY: clean setup build package

clean:
	rm -rf __pycache__
setup:
	pip install -r requirements.txt

test:
	python -m pytest tests/
