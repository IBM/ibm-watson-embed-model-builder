test: ## run pytest in parallel with coverage reporting. Fail under 100% code coverage
	pytest -W error \
		--cov-report term \
		--cov-report html:htmlcov \
		--cov=model_image_builder \
		--cov-fail-under=100.0 \
		--html="reports/report.html" \
		--self-contained-html \
		-n 4 \
		tests

install: ## install dev dependencies
	pip3 install -r requirements.txt

install-test: ## install test dependencies
	pip3 install -r requirements_test.txt

build: ## Build the wheel
	python3 setup.py bdist_wheel

clean: ## clean up build artifacts and test reports
	rm -fr build dist htmlcov reports *.egg-info .coverage

fmt:
	black . && isort .

fmt-check:
	black --check . && isort --check .

release:
	pip3 install twine
	twine upload --username "__token__" --password "${PYPI_TOKEN}" --repository testpypi dist/*
