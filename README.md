# BreastCancerClassification_server
Building a Docker Image from project
```
sudo docker build -t serverml_image .
```
Pulling a Docker Image from Github Container Registry
```
sudo docker pull ghcr.io/vmmadhavareddy/serverml_image
```
Running the Container with provided Image on http://0.0.0.0/8000/ forwarded to our system locally on http://127.0.0.1/8000/
```
sudo docker run -it -p 8000:8000 --name serverml_container ghcr.io/vmmadhavareddy/serverml_image
```
