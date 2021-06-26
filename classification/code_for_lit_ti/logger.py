import logging
import os
from pathlib import Path
import json
from datetime import datetime
from params import args
from torch.utils.tensorboard import SummaryWriter
dt = datetime.now()
dt.replace(tzinfo=datetime.now().astimezone().tzinfo)
current_time = datetime.now().strftime('%b%d_%H-%M-%S')

_LOG_FMT = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s'
_DATE_FMT = '%m/%d/%Y %H:%M:%S'
logging.basicConfig(format=_LOG_FMT, datefmt=_DATE_FMT, level=logging.INFO)
logger = logging.getLogger('__main__')  # this is the global logger

output_dir = os.path.join('outputs', args.exp_name)
tb_log_dir = os.path.join(output_dir, 'tensorboard_logs')

Path(output_dir).mkdir(parents=True, exist_ok=True)
checkpoint_path = os.path.join(output_dir, 'last_checkpoint.pth')
if os.path.exists(checkpoint_path) and not args.resume:
    setattr(args, 'resume', checkpoint_path)

setattr(args, 'tb_log_dir', tb_log_dir)
setattr(args, 'output_dir', output_dir)

if not args.eval:
    log_path = os.path.join(output_dir, 'all_logs.txt')
    with open(os.path.join(args.output_dir, 'args.json'), 'w+') as f:
        json.dump(vars(args), f, indent=4)
else:
    log_path = os.path.join(output_dir, 'eval_logs.txt')

fh = logging.FileHandler(log_path, 'a+')
formatter = logging.Formatter(_LOG_FMT, datefmt=_DATE_FMT)
fh.setFormatter(formatter)
logger.addHandler(fh)

class TensorboardLogger(object):
    def __init__(self):
        self._logger = None
        self._global_step = 0

    def set_step(self, step):
        self._global_step = step

    def create(self, path):
        self._logger = SummaryWriter(path)

    def noop(self, *args, **kwargs):
        return

    def step(self):
        self._global_step += 1

    @property
    def global_step(self):
        return self._global_step

    def log_scaler_dict(self, log_dict, prefix=''):
        """ log a dictionary of scalar values"""
        if self._logger is None:
            return
        if prefix:
            prefix = f'{prefix}_'
        for name, value in log_dict.items():
            if isinstance(value, dict):
                self.log_scaler_dict(value, self._global_step,
                                     prefix=f'{prefix}{name}')
            else:
                self._logger.add_scalar(f'{prefix}{name}', value,
                                        self._global_step)

    def __getattr__(self, name):
        if self._logger is None:
            return self.noop
        return self._logger.__getattribute__(name)


TB_LOGGER = TensorboardLogger()