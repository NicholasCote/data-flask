# data-flask
A Flask application for exploring items of value at NCAR on k8s

## CI/CD Status

ML Argo CD Status: https://argocd.k8s.ucar.edu/api/badge?name=mlc1-argocd&revision=true&showAppName=true
NWSC Argo CD Status: https://argocd.k8s.ucar.edu/api/badge?name=nwc1-argocd&revision=true&showAppName=true

https://mlc1-argo.k8s.ucar.edu/api/badge?name=ncote-db-demo&revision=true&showAppName=true
https://nwc1-argo.k8s.ucar.edu/api/badge?name=ncote-celery-demo&revision=true&showAppName=true

| GitHub Action | Status |
| --- | --- |
| Test Build & Push |  ![Dev Build](https://github.com/NicholasCote/data-flask/actions/workflows/test-data-cicd.yaml/badge.svg) |
| Main Build & Push |  ![Main Build](https://github.com/NicholasCote/data-flask/actions/workflows/data-cicd.yaml/badge.svg) |

## Blue/Green

This repository is configured to utilize the dev and main branches, couple with GitHub actions, to deploy changes made to dev in order to test functionality before updating the main branch and ultimately the production website. 

