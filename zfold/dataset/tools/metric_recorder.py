"""Record evalution metrics calculated at each iteration."""

import logging

from collections import defaultdict

class MetricRecorder():
    """Record evalution metrics calculated at each iteration."""

    def __init__(self):
        """Constructor function."""

        self.metrics_list = []


    def reset(self):
        """Reset the recorder."""

        self.metrics_list = []


    def add(self, metrics):
        """Add a set of evaluation metrics into the buffer."""

        self.metrics_list.append(metrics)


    def get(self):
        """Get the averaged evaluation metrics for all the records in the buffer."""

        return self.__calc_metrics_avg()


    def display(self, header=''):
        """Display the averaged evaluation metrics for all the records in the buffer."""

        metrics = self.__calc_metrics_avg()
        metrics_str = ', '.join(['%s=%.4f' % x for x in metrics.items()])
        logging.info('%s%s', header, metrics_str)


    def __calc_metrics_avg(self):
        """Calculate the averaged evaluation metrics for all the records in the buffer."""

        if not self.metrics_list:
            return {}

        metrics_cnt = defaultdict(int)
        metrics_sum = defaultdict(float)
        for metrics in self.metrics_list:
            for key, value in metrics.items():
                metrics_cnt[key] += 1
                metrics_sum[key] += value
        metrics_avg = {k: v / metrics_cnt[k] for k, v in metrics_sum.items()}

        return metrics_avg
