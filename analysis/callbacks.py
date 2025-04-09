import json
from egg.core.callbacks import Callback
from pathlib import Path

class DataLogger(Callback):
    def __init__(self, save_path="logs/experiment.json"):
        super().__init__()
        self.save_path = Path(save_path)
        self.epoch_logs = []

        self.save_path.parent.mkdir(parents=True, exist_ok=True)

    def _serialize_logs(self, logs, loss, epoch, mode):
        out = {
            "epoch": epoch,
            "mode": mode,
            "loss": float(loss),
        }
        out.update({k: float(v.mean()) for k, v in logs.aux.items()})
        return out

    def on_epoch_end(self, loss, logs, epoch):
        self.epoch_logs.append(self._serialize_logs(logs, loss, epoch, mode="train"))
        self._save()

    def on_validation_end(self, loss, logs, epoch):
        self.epoch_logs.append(self._serialize_logs(logs, loss, epoch, mode="test"))
        self._save()

    def _save(self):
        with open(self.save_path, "w") as f:
            json.dump(self.epoch_logs, f, indent=2)

