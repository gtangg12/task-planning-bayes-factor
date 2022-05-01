from typing import Optional
from dataclasses import dataclass, field


@dataclass
class TrainingArguments:
    num_train_epochs: int = field(
        default=3, metadata={'help': 'Total number of training epochs to perform.'}
    )

    per_device_train_batch_size: int = field(
        default=8, metadata={'help': 'Batch size per GPU/TPU core/CPU for training.'}
    )
    per_device_eval_batch_size: int = field(
        default=8, metadata={'help': 'Batch size per GPU/TPU core/CPU for evaluation.'}
    )

    gradient_accumulation_epochs: int = field(
        default=1,
        metadata={'help': 'Number of epochs to accumulate before performing a backward/update pass.'},
    )

    shuffle: bool = field(
        default=True, 
        metadata={'help': 'Whether to shuffle the data when training.'}
    )
    dataloader_num_workers: int = field(
        default=0,
        metadata={
            'help': 'Number of subprocesses to use for data loading. 0 means that the data will be loaded \
                in the main process.'
        },
    )

    logging_dir: Optional[str] = field(
        default=None, metadata={'help': 'logging dir for metrics.'}
    )
    logging_epochs: int = field(
        default=1, metadata={'help': 'log metrics every X epochs.'}
    )

    save_dir: str = field(
        default=None,
        metadata={'help': 'The output directory where model checkpoints will be written.'},
    )
    save_epochs: int = field(
        default=1, metadata={'help': 'Save model checkpoint every X epochs.'}
    )