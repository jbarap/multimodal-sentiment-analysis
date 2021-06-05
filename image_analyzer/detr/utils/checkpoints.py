import glob

from datetime import datetime
from pathlib import Path

import torch


class CheckpointManager:
    """Class to handle model, optimizer and loss checkpoints."""
    # saves checkpoint with the format: ConfigName_Epoch_date.pt
    def __init__(self, config_name, save_every=5, current_epoch=1, save_path='data/state_dicts'):
        self.config_name = config_name
        self.save_every = save_every
        self.current_epoch = current_epoch
        self.save_path = save_path

        self.lowest_loss = 1e6  # not used atm

    def step(self, model, optimizer, loss):
        self.current_epoch += 1

        if (self.current_epoch % self.save_every == 0) and self.current_epoch != 0:
            self.save_checkpoint(model, optimizer, loss)

    def save_checkpoint(self, model, optimizer, loss):
        today = datetime.today()
        today = today.strftime("%Y-%m-%d")

        checkpoint_name = Path(self.save_path, f"{self.config_name}_{self.current_epoch}_{today}.pt")

        # NOTE: for dataparallel use: torch.save(model.module.state_dict())
        torch.save({
            'epoch': self.current_epoch,
            'state_dict': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'loss': loss
        }, checkpoint_name)

    def load_checkpoint(self, mode='latest'):
        """Return (state_dict, optimizer_state)"""

        if mode == 'latest':
            pattern = str(Path(self.save_path, self.config_name)) + "*.pt"
            checkpoints = glob.glob(pattern)
            if len(checkpoints) == 0:
                raise AttributeError("No checkpoints found.")

            checkpoints.sort(key=lambda x: datetime.strptime(x.replace('.pt', '').split('_')[-1], "%Y-%m-%d"))

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            loaded_checkpoint = torch.load(checkpoints[0], map_location=device)
        else:
            raise ValueError(f"Mode {mode} not allowed.")

        self.current_epoch = loaded_checkpoint['epoch'] + 1

        return loaded_checkpoint['state_dict'], loaded_checkpoint['optimizer_state']
            
