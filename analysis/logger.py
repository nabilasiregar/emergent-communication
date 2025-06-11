import os
import csv
import egg.core as core

class CsvLogger(core.Callback):
    """
    A custom EGG callback to log training and validation metrics to a CSV file
    """
    def __init__(self, log_dir: str, filename: str):
        self.log_dir = log_dir
        self.log_path = os.path.join(self.log_dir, f"{filename}.csv")
        self.file = None
        self.writer = None
        self.header_written = False

    def on_train_begin(self, trainer_instance: "Trainer"):
        super().on_train_begin(trainer_instance)
        os.makedirs(self.log_dir, exist_ok=True)
        self.file = open(self.log_path, "w", newline="")
        self.writer = csv.writer(self.file)

    def _write_header(self, metrics: dict):
        if self.writer:
            header = ['epoch', 'mode', 'loss'] + sorted(metrics.keys())
            self.writer.writerow(header)
            self.header_written = True

    def _log_metrics(self, loss: float, logs: core.Interaction, epoch: int, mode: str):
        if not self.writer:
            return
        # aggregate metrics from interaction object
        aggregated_metrics = {k: v.mean().item() for k, v in logs.aux.items()}

        if not self.header_written:
            self._write_header(aggregated_metrics)

        row = [epoch, mode, f"{loss:.5f}"]
        for key in sorted(aggregated_metrics.keys()):
            row.append(f"{aggregated_metrics[key]:.5f}")
        
        self.writer.writerow(row)
        self.file.flush()

    def on_epoch_end(self, loss: float, logs: core.Interaction, epoch: int):
        self._log_metrics(loss, logs, epoch, "train")

    def on_validation_end(self, loss: float, logs: core.Interaction, epoch: int):
        self._log_metrics(loss, logs, epoch, "test")

    def on_train_end(self):
        if self.file:
            self.file.close()

