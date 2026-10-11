"""The coding-agent half of ``bigym-view``: run directories, panels, compare.

The generic viewer calls into this package for everything that knows the
run layout.

Threading: every function here is called from the viewer's main loop. GUI
callbacks only set request flags (env builds, renders and subprocess
launches belong to the main thread, which owns the EGL context).
"""
