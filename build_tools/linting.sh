#!/bin/bash

set -e -x
set -o pipefail

if ! flake8 --verbose --filename=*.py torchensemble/; then
  echo 'Failure on Code Quality Check.'
  exit 1
fi