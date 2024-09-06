# server
## Building and Running the Flower AI Monitoring App

To build the Flower AI Monitoring App, follow these steps:

1. Build the monitoring image:
    ```shell
    docker build -t flower-ai-monitoring-app .
    ```

2. Run the monitoring container:
    ```shell
    docker run --name flower-ai-monitoring-container -p 8080:8080 -p 8000:8000 flower-ai-monitoring-app
    ```

## Building and Running the Flower AI Client

To build the Flower AI Client, follow these steps:

1. Build the client image:
    ```shell
    docker build -f DockerfileClient -t flower-ai-client-app .
    ```

2. Run the client container:
    ```shell
    docker run --name flower-ai-client-container flower-ai-client-app
    ```

## Removing Containers

To remove the containers, use the following commands:

```shell
docker container stop flower-ai-monitoring-container
docker container rm flower-ai-monitoring-container

docker container stop flower-ai-client-container
docker container rm flower-ai-client-container
```
