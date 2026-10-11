"""The coding-agent half of ``bigym-view``: run directories, panels, compare.

``bigym/vr/viewer/view_demos.py`` keeps the generic viewer (batch discovery,
env build, playback, camera panels, figure capture) and calls into this
package for everything that knows the run layout. The dependency runs one
way: nothing here imports the viewer.

Threading: every function here is called from the viewer's main loop. GUI
callbacks only set request flags (env builds, renders and subprocess
launches belong to the main thread, which owns the EGL context).
"""
