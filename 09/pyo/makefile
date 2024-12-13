run:
	./process.py ../input.txt

stubcheck:
	cargo run --bin stubgen
	mypy process.py

test: test_rust test_python

test_rust:
	cargo test

test_python:
	pytest

lint: lint_rust

lint_rust:
	cargo clippy
	cargo machete
