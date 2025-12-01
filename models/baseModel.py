"""
This module previously contained a BaseModel class used as a common parent
for specific dynamical models. The project now prefers concrete model
implementations without inheritance. The sampling time should be initialized
directly inside each model's constructor as `self._sampling_time = sampling_time`.

The file is kept for backward compatibility, but no BaseModel class is
exposed anymore.
"""