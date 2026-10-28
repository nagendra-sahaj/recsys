# `lightgbm` MUST be imported before `faiss` anywhere in this process. On macOS, faiss
# and lightgbm each ship their own separate physical copy of the OpenMP runtime
# (faiss bundles a private libomp.dylib inside its wheel; lightgbm links Homebrew's).
# Verified experimentally: loading them in the wrong order reliably either segfaults
# (importing faiss first, then unpickling a fitted lightgbm model -- exit code 139) or
# deadlocks (lightgbm first, then a multi-threaded FAISS search -- process hangs with
# ~0% CPU). The only combination confirmed stable end-to-end is lightgbm imported
# first, with FAISS additionally forced single-threaded (see serving/ann_index.py).
#
# Importing it here, at the top of the package `__init__.py`, guarantees the safe
# order regardless of which submodule a script or notebook happens to import first --
# do not remove this or let an import-sorter reorder/relocate it.
import lightgbm  # noqa: F401, I001
