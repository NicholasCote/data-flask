# data-flask
A Flask application for exploring items of value at NCAR on k8s

## CI/CD Status

| GitHub Action | Status |
| --- | --- |
| Test Site Status |  ![Dev Build](https://github.com/NicholasCote/data-flask/actions/workflows/test-data-cicd.yaml/badge.svg) |
| Main Site Status |  ![Main Build](https://github.com/NicholasCote/data-flask/actions/workflows/data-cicd.yaml/badge.svg) |

## Blue/Green

This repository is configured to utilize the dev and main branches, couple with GitHub actions, to deploy changes made to dev in order to test functionality before updating the main branch and ultimately the production website. 

