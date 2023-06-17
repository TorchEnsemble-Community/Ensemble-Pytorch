

from .joblib_backend import JoblibBackend
from .ray_backend import RayBackend


class BackendFactory(object):

    @staticmethod
    def create_backend(num_workers, parallel_backend):
        """
        Create the backend for parallelization, currently support two different
        backends, Joblib and Ray.
        """
        if not num_workers > 0:
            error_msg = "`num_worker` should be greater than 0, got {} " \
                        "instead".format(num_workers)
            raise ValueError(error_msg)

        if parallel_backend not in ("Joblib", "Ray"):
            error_msg = "`parallel_backend` should be one of ('Joblib', " \
                        "'Ray'), got '{}' instead".format(parallel_backend)
            raise ValueError(error_msg)

        if parallel_backend == "Joblib":
            return JoblibBackend(num_workers)
        else:
            return RayBackend(num_workers)
