"""Import this script to avoid errors when '@profile' is used for performance profiling."""

try:
    profile
except NameError:
    def profile(func):
        """Directly execute the function without performance profiling."""
        return func
