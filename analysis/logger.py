import os
import json
from datetime import datetime
from typing import Dict, Any
import egg.core as core
import torch
import torch.nn as nn
import pdb

class ResultsCollector(core.Callback):
    """
    A class that logs epoch summaries and detailed batch interactions.
    Logs epoch-level metrics for plotting curves and summarizes graph data for later analysis.
    Logs are saved in JSON Lines format in a unique directory for each run.
    """
    def __init__(self,
                 log_dir: str = "logs",
                 experiment_name: str = "experiment",
                 log_epoch_summary: bool = True,
                 log_val_interactions: bool = True,
                 val_interaction_epoch_freq: int = 1,
                 log_train_interactions: bool = False,
                 train_interaction_epoch_freq: int = 5):
        super().__init__()

        # track current epoch for batch logging
        self.current_epoch = 0

        # user settings
        self.log_epoch_summary = log_epoch_summary
        self.log_val_interactions = log_val_interactions
        self.val_interaction_epoch_freq = val_interaction_epoch_freq
        self.log_train_interactions = log_train_interactions
        self.train_interaction_epoch_freq = train_interaction_epoch_freq

        # prepare run directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_log_dir = os.path.join(log_dir, f"{experiment_name}_{timestamp}")
        os.makedirs(self.run_log_dir, exist_ok=True)

        # open JSONL files
        self.epoch_summary_file = open(
            os.path.join(self.run_log_dir, "epoch_summary.jsonl"), "w", encoding="utf-8"
        ) if self.log_epoch_summary else None
        self.validation_interactions_file = open(
            os.path.join(self.run_log_dir, "validation_interactions.jsonl"), "w", encoding="utf-8"
        ) if self.log_val_interactions else None
        self.training_interactions_file = open(
            os.path.join(self.run_log_dir, "training_interactions.jsonl"), "w", encoding="utf-8"
        ) if self.log_train_interactions else None

    def _log_json_line(self, file_handle, data_dict: Dict[str, Any]):
        """Helper to write a dictionary as a JSON string to a file."""
        if file_handle and not file_handle.closed:
            try:
                file_handle.write(json.dumps(data_dict) + "\n")
                file_handle.flush()
            except Exception as e:
                print(f"Error writing to {getattr(file_handle, 'name', '')}: {e}")
                status = {}
                for k, v in data_dict.items():
                    try:
                        json.dumps({k: v})
                        status[k] = "OK"
                    except Exception:
                        status[k] = f"ERROR: {type(v)}"
                print("Serialization status:", status)

    def _get_epoch_metrics(self, loss: float, logs: core.Interaction, mode: str, epoch: int) -> Dict[str, Any]:
        """Aggregate loss and any aux metrics at epoch end."""
        metrics = {
            "epoch": epoch,
            "mode": mode,
            "loss": float(loss) if torch.is_tensor(loss) else loss
        }
        for k, v in (logs.aux or {}).items():
            if torch.is_tensor(v):
                try:
                    metrics[k] = v.mean().item()
                except:
                    metrics[k] = None
            else:
                metrics[k] = v
        return metrics

    def _to_list_if_tensor(self, item: Any) -> Any:
        if item is None:
            return None
        if torch.is_tensor(item):
            try:
                return item.tolist()
            except:
                return str(item)
        return item

    def _get_interaction_data(self,
                              interaction: core.Interaction,
                              epoch: int,
                              batch_idx: int,
                              mode: str) -> Dict[str, Any]:
        """Extract detailed data from one batch interaction."""
        msgs = self._to_list_if_tensor(interaction.message)
        lbls = self._to_list_if_tensor(interaction.labels)
        recv_out = None
        if interaction.receiver_output is not None:
            try:
                recv_out = interaction.receiver_output.argmax(dim=-1).tolist()
            except:
                recv_out = self._to_list_if_tensor(interaction.receiver_output)
        acc = None
        if interaction.aux and "acc" in interaction.aux:
            acc = self._to_list_if_tensor(interaction.aux["acc"])
        # summarize graph data if present
        graph_info = {}
        if interaction.aux_input and "data" in interaction.aux_input:
            data_obj = interaction.aux_input["data"]
            # total nodes and edges
            if hasattr(data_obj, 'x'):
                graph_info['total_nodes'] = int(data_obj.x.size(0))
            if hasattr(data_obj, 'edge_index'):
                graph_info['total_edges'] = int(data_obj.edge_index.size(1))
            # number of graphs
            if hasattr(data_obj, 'num_graphs'):
                graph_info['num_graphs'] = int(data_obj.num_graphs)
            elif hasattr(data_obj, 'ptr'):
                graph_info['num_graphs'] = int(data_obj.ptr.size(0) - 1)
            # nodes-per-graph stats
            try:
                counts = torch.bincount(data_obj.batch)
                graph_info['nodes_per_graph_min'] = int(counts.min())
                graph_info['nodes_per_graph_max'] = int(counts.max())
                graph_info['nodes_per_graph_mean'] = float(counts.float().mean().item())
            except:
                pass
        # sender input features
        sender_feats = None
        if interaction.sender_input is not None:
            inp = interaction.sender_input
            if isinstance(inp, tuple) and len(inp) > 0:
                sender_feats = self._to_list_if_tensor(inp[0])
            else:
                sender_feats = self._to_list_if_tensor(inp)

        return {
            "epoch": epoch,
            "batch_idx": batch_idx,
            "mode": mode,
            "messages": msgs,
            "labels": lbls,
            "receiver_output_argmax": recv_out,
            "accuracy_per_item": acc,
            "sender_input_features": sender_feats,
            "graph_info": graph_info
        }

    def on_epoch_end(self, loss: float, logs: core.Interaction, epoch: int):
        self.current_epoch = epoch
        if self.log_epoch_summary:
            metrics = self._get_epoch_metrics(loss, logs, "train", epoch)
            self._log_json_line(self.epoch_summary_file, metrics)

    def on_validation_end(self, loss: float, logs: core.Interaction, epoch: int):
        self.current_epoch = epoch
        if self.log_epoch_summary:
            metrics = self._get_epoch_metrics(loss, logs, "validation", epoch)

            self._log_json_line(self.epoch_summary_file, metrics)

    def on_batch_end(self,
                     logs: core.Interaction,
                     loss: float,
                     batch_id: int,
                     is_training: bool = True):
        if is_training:
            freq        = self.train_interaction_epoch_freq
            file_handle = self.training_interactions_file
            mode        = "train_batch_interaction"
            enabled     = self.log_train_interactions
        else:
            freq        = self.val_interaction_epoch_freq
            file_handle = self.validation_interactions_file
            mode        = "validation_batch_interaction"
            enabled     = self.log_val_interactions

        if not enabled or file_handle is None:
            return

        # only log on configured epoch frequency
        if (self.current_epoch - 1) % freq == 0:
            data = self._get_interaction_data(logs,
                                              self.current_epoch,
                                              batch_id,
                                              mode)
            self._log_json_line(file_handle, data)

    def on_train_end(self):
        """Called when training finishes—close files."""
        for fh in (self.epoch_summary_file,
                   self.validation_interactions_file,
                   self.training_interactions_file):
            if fh and not fh.closed:
                fh.close()
        print(f"ResultsCollector: Finished logging to {self.run_log_dir}")

