from egg.core.early_stopping import EarlyStopper

class EarlyStopperLoss(EarlyStopper):
    """
    Early stopping on validation loss: stop if val_loss doesn’t improve
    by at least min_delta after patience consecutive epochs.
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        validation: bool = True,
        verbose: bool = False
    ) -> None:
        """
        :param patience: how many epochs to wait after last improvement
        :param min_delta: minimum decrease in loss to count as improvement
        :param validation: whether to use validation stats (True) or training stats
        :param verbose: if True, print a message when stopping
        """
        super().__init__(validation)
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose

        self.best_loss: float = float("inf")
        self.wait: int = 0

    def should_stop(self) -> bool:
        # pick the most recent loss (validation or train)
        stats = self.validation_stats if self.validation else self.train_stats
        assert stats, "No stats collected yet for early stopping"
        current_loss, _ = stats[-1]

        # first epoch: just initialize best_loss
        if self.best_loss is None or current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1

        if self.wait >= self.patience:
            if self.verbose:
                epoch_num = len(self.validation_stats)
                print(
                    f"[EarlyStopperLoss] stopping at epoch {epoch_num}: "
                    f"val_loss did not improve by >{self.min_delta} for {self.patience} epochs "
                    f"(best={self.best_loss:.4f}, last={current_loss:.4f})"
                )
            return True

        return False
