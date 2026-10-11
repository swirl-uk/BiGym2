"""The coding-agent half of ``bigym-view``: run directories, panels, compare.

The generic viewer calls into this package for everything that knows the
run layout.

Threading: GUI callbacks only set request flags; env builds, renders and
subprocess launches belong to the viewer's main loop, whose thread owns the
EGL context. Directory scans may read cells on a worker thread.
"""
