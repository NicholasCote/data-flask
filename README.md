# data-flask
A Flask application for exploring items of value at NCAR on k8s

## CI/CD Status

| Container Build Status |  ![Dev Build](https://github.com/NicholasCote/data-flask/actions/workflows/test-data-cicd/badge.svg) | ![Main Build](https://github.com/NicholasCote/data-flask/actions/workflows/data-cicd/badge.svg) |

## Blue/Green

This repository is configured to utilize the dev and main branches, couple with GitHub actions, to deploy changes made to dev in order to test functionality before updating the main branch and ultimately the production website. 

