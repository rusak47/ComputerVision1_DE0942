$ docker image inspect deca-perfect:latest --format='{{.Size}}' | numfmt --to=iec
21G

$ docker save deca-perfect:latest | zstd -T0 -19 -o deca-perfect_latest.tar.zst
/*stdin*\            : 33.48%   (  20.4 GiB =>   6.84 GiB, deca-perfect_latest.tar.zst) 

$ zstd -dc deca-perfect_latest.tar.zst | docker load