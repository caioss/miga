import os
import shutil
from subprocess import Popen

class CUDABuild:
    def __init__(self):
        home = os.environ.get("CUDA_HOME", "")

        bin_dir = os.path.join(home, "bin")
        nvcc = shutil.which("nvcc", path=bin_dir)

        lib_dir = os.path.join(home, "lib64")
        include_dir = os.path.join(home, "include")

        if nvcc is not None and os.path.exists(lib_dir) and os.path.exists(include_dir):
            self.installed = True
            self.bin = bin_dir
            self.nvcc = nvcc
            self.lib = lib_dir
            self.include = include_dir

        else:
            self.installed = False
            self.bin = None
            self.nvcc = None
            self.lib = None
            self.include = None

        self.arch = os.environ.get("CUDA_ARCH", "")
        if not self.arch:
            self.arch = ""
            # Comment out unnecessary architectures to compile faster
            self.arch += " -gencode=arch=compute_30,code=compute_30 -gencode=arch=compute_30,code=sm_30"
            self.arch += " -gencode=arch=compute_50,code=compute_50 -gencode=arch=compute_50,code=sm_50"
            self.arch += " -gencode=arch=compute_60,code=compute_60 -gencode=arch=compute_60,code=sm_60"

    def __bool__(self):
        return self.installed

    def __nonzero__(self):
        return self.__bool__()

    def compile(self, file_list, includes, dry_run=False):
        object_list = []

        if not self.installed:
            return object_list

        for file in file_list:
            obj_name = os.path.splitext(file)[0] + ".o"
            object_list.append(obj_name)

            if dry_run:
                continue

            command = "{} {} -c -o {} -Xcompiler -fPIC {} -cudart static -O3 -std c++11 {}".format(
                self.nvcc,
                file,
                obj_name,
                self.arch,
                " ".join("-I{}".format(folder) for folder in includes),
            )
            print(command)

            proc = Popen(command.split())

            out = proc.communicate()
            if out[0]:
                print(out[0].decode("utf-8"))

            if out[1]:
                print(out[1].decode("utf-8"))

            if not os.path.exists(obj_name):
                raise RuntimeError("Error compiling file {}".format(file))

        return object_list
