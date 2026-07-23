<!--
Copyright (c) 2026 Hygon Information Technology Co., Ltd.
SPDX-License-Identifier: BSD-3-Clause
-->

编译步骤：

1 安装 mkl

```
apt-get update && apt-get install intel-mkl
```

2 转码

```
cd faiss
./faiss/gpu/hipify.sh
```

3 编译指令

```
# Ubuntu
export LIBRARY_PATH=/opt/dtk/dcc/lib/clang/17.0.0/lib/linux:$LIBRARY_PATH
CXX=hipcc CC=hipcc cmake -DFAISS_ENABLE_GPU=ON -DFAISS_ENABLE_ROCM=ON -DBUILD_SHARED_LIBS=ON -DFAISS_OPT_LEVEL=avx -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release  -DCMAKE_INSTALL_PREFIX=faiss/build/install  -DMKL_LIBRARIES=/usr/lib/x86_64-linux-gnu/libmkl_rt.so ..
#Rocky
CXX=hipcc CC=hipcc cmake -DFAISS_ENABLE_GPU=ON -DFAISS_ENABLE_ROCM=ON -DBUILD_SHARED_LIBS=ON -DFAISS_OPT_LEVEL=avx -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release  -DCMAKE_INSTALL_PREFIX=faiss/build/install  -DMKL_LIBRARIES=/opt/intel/compilers_and_libraries_2020.4.304/linux/mkl/lib/intel64_lin/libmkl_rt.so ..
```

4 whl 包编译

```
cd faiss/build/faiss/python
python setup.py bdist_wheel
```

5 测试

* faiss/build

  ```
  cd faiss/build
  ./faiss_test
  ```
* faiss/build

  ```
  ctest
  ```
