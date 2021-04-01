#!/bin/bash

command -v pre-commit >/dev/null 2>&1 || { echo >&2 "You should install pre-commit for development; exiting..."; exit 1; }
echo "Installing pre-commit hooks if they're not already present"
pre-commit install --install-hooks

