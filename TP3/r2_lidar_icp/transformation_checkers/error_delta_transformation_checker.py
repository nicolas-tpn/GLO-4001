from r2_lidar_icp.transformation_checkers.transformation_checker import TransformationChecker


class ErrorDeltaTransformationChecker(TransformationChecker):
    def __init__(self, min_error_delta: float):
        """
        Transformation checker that stops the ICP algorithm when the mean error between the point cloud
        and the reference is not decreasing anymore.
        :param min_error_delta: Minimum absolute error delta between two iterations
        """
        super().__init__()
        self.min_error_delta = min_error_delta
        self.previous_error = 0

    def begin(self):
        self.previous_error = 0

    def _is_finished_check(self, error) -> bool:
        if abs(self.previous_error - error) < self.min_error_delta:
            return True
        self.previous_error = error
        return False
