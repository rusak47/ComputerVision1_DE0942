cd ~/DEKA2

docker build --no-cache -t deca-perfect:latest - << 'EOF'
FROM nvcr.io/nvidia/pytorch:24.10-py3

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Python dependencies + chumpy
RUN pip install --no-cache-dir \
    'numpy<2.0' chumpy==0.70 opencv-python==4.5.5.64 \
    tqdm PyYAML face-alignment==1.3.5 scikit-image yacs \
    pillow dominate flask flask-cors trimesh pyrender matplotlib \
    kornia kornia-rs \
    torch-scatter==2.1.2+pt25cu121 -f https://data.pyg.org/whl/torch-2.5.0+cu121.html

# Patch + verify with visible proof (broken, only first bool is fixed - run patch script afterwards)
RUN CHUMPY_FILE=$(find /usr/local/lib -name chumpy -type d)/__init__.py && \
    echo "Patching chumpy at: $CHUMPY_FILE" && \
    sed -i 's/from numpy import bool,/from numpy import bool_,/g'        "$CHUMPY_FILE" && \
    sed -i 's/from numpy import int,/from numpy import int_,/g'          "$CHUMPY_FILE" && \
    sed -i 's/from numpy import float,/from numpy import float_,/g'      "$CHUMPY_FILE" && \
    sed -i 's/from numpy import complex,/from numpy import complex_,/g'  "$CHUMPY_FILE" && \
    sed -i 's/from numpy import object,/from numpy import object_,/g'     "$CHUMPY_FILE" && \
    sed -i 's/from numpy import unicode,/from numpy import unicode_,/g'  "$CHUMPY_FILE" && \
    sed -i 's/from numpy import str,/from numpy import str_,/g'          "$CHUMPY_FILE" && \
    echo "Patch applied!" && \
    echo "Verification:" && \
    grep "from numpy import" "$CHUMPY_FILE"

WORKDIR /deca
EOF