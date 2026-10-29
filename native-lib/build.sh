#!/bin/sh
# MLX for the JVM (FFM) and Scala Native: mlx-c (and MLX under it) built as shared libraries, installed into
# native-lib/out — lib/libmlxc.dylib, lib/libmlx.dylib, lib/mlx.metallib (beside libmlx: MLX finds it there), include/
set -e
cd "$(dirname "$0")"
TAG=${MLX_C_TAG:-v0.8.0}
[ -d src ] || git clone -q https://github.com/ml-explore/mlx-c.git src
git -C src checkout -q "$TAG"
cmake -S src -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DMLX_C_BUILD_EXAMPLES=OFF \
  -DCMAKE_INSTALL_PREFIX="$PWD/out" > build.log
cmake --build build -j "$(sysctl -n hw.ncpu)" >> build.log
cmake --install build >> build.log
ls out/lib
# okm: the flat layer over mlx-c both platforms call
clang -O2 -dynamiclib okm.c -I out/include -L out/lib -lmlxc -Wl,-rpath,@loader_path \
  -install_name @rpath/libokm.dylib -o out/lib/libokm.dylib
ls out/lib/libokm.dylib
