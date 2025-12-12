#not checked
docker-compose -f .devcontainer/docker-compose.yml build instant-ngp
xhost local:root
docker-compose -f .devcontainer/docker-compose.yml run instant-ngp /bin/bash

Copied from: GitHub - NVlabs/instant-ngp: Instant neural graphics primitives: lightning fast NeRF and more - <https://github.com/NVlabs/instant-ngp#compilation-windows--linux>