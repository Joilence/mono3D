.PHONY: type-check test format

# Static type checking
type-check:
	mypy .

# Run tests
test:
	pytest .

# Format code
format:
	black .
	isort .