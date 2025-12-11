docker run --name temp-fix \
  --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$(pwd):/deca" \
  deca-perfect:latest \
  bash -c "
    set -e
    FILE='/usr/local/lib/python3.10/dist-packages/chumpy/__init__.py'
    echo '=== BEFORE ==='
    grep 'from numpy import' \"\$FILE\"

    sed -i 's/bool,/bool_,/g'     \"\$FILE\"
    sed -i 's/int,/int_,/g'       \"\$FILE\"
    sed -i 's/float,/float_,/g'   \"\$FILE\"
    sed -i 's/complex,/complex_,/g' \"\$FILE\"
    sed -i 's/object,/object_,/g' \"\$FILE\"
    sed -i 's/unicode,/unicode_,/g' \"\$FILE\"
    sed -i 's/str,/str_,/g'       \"\$FILE\"

    echo '=== AFTER – ALL 7 FIXED ==='
    grep 'from numpy import' \"\$FILE\"

    echo '=== Pre-downloading face-alignment models once and forever ==='
    python -c \"import face_alignment; \
               fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device='cuda'); \"

    echo '=== ALL DONE – committing to image ==='
  " && \
  docker commit temp-fix deca-perfect:latest && \
  docker rm temp-fix && \
  echo "IMAGE PERMANENTLY FIXED – YOU ARE FREE FOREVER!"