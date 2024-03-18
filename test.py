import os
import sys

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from trainer import Trainer
from transposed_transformer import TT
from vanilla_transformer import VanillaTransformer
from model import GPT
from utils import ConfigNode, set_seed, setup_logging

# ---------------------------------------------------

model_config = ConfigNode(
    n_tokens=16,  # context window (fixed when transposed)
    n_layer=20,   # num layers (variable when transposed!)
    n_head=4,     # num self attention heads
    n_embd=64*3, # dimension of representation
    dropout=0.0,
    bias=False,
    is_causal=True,  # autoregressive modelling if True
    is_transposed=False  # transposed or vanilla transformer
)

# ---------------------------------------------------

def get_config():
    config = ConfigNode()

    # system
    config.system = ConfigNode()
    config.system.seed = 3407
    config.system.work_dir = './out/minimal-example-all-zeros'

    # data
    config.data = CharDataset.get_default_config()
    config.data_filepath = 'data/minimal-datasets/all-zeros.txt'

    # model
    config.model = model_config

    # trainer
    config.trainer = Trainer.get_default_config()
    config.trainer.learning_rate = 5e-4 # the model we're using is so small that we can go a bit faster
    config.trainer.max_iters = int(1e4)+1
    config.trainer.batch_size = 64

    return config

class CharDataset(Dataset):
    """
    Emits batches of characters
    """

    @staticmethod
    def get_default_config():
        C = ConfigNode()
        C.n_tokens = model_config.n_tokens
        return C

    def __init__(self, config, data):
        self.config = config

        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.vocab_size = vocab_size
        self.data = data

    def get_vocab_size(self):
        return self.vocab_size

    def get_n_tokens(self):
        return self.config.n_tokens

    def __len__(self):
        return len(self.data) - self.config.n_tokens

    def __getitem__(self, idx):
        # grab a chunk of (n_tokens + 1) characters from the data
        chunk = self.data[idx:idx + self.config.n_tokens + 1]
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        # return as tensors
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y
    
# ---------------------------------------------------

if __name__ == "__main__":
    # get default config and overrides from the command line, if any
    config = get_config()
    config.merge_from_args(sys.argv[1:])
    setup_logging(config)
    set_seed(config.system.seed)

    # construct the training dataset
    text = open(config.data_filepath, 'r').read()[:-1]  # remove final newline
    train_dataset = CharDataset(config.data, text)
    train_dataset.config.n_tokens = model_config.n_tokens

    # print config details
    model_type = "Transposed" if config.model.is_transposed else "Vanilla"
    print(f"--- {model_type} Transformer ---\n")
    print(config)

    # construct the model
    config.model.vocab_size = train_dataset.get_vocab_size()
    model_class = TT if config.model.is_transposed else VanillaTransformer # GPT # VanillaTransformer
    print(f"\n--- Instantiating {model_type} Transformer ---\n\n\t", end="")
    model = model_class(config.model)
    print("")

    # construct the trainer object
    trainer = Trainer(config.trainer, model, train_dataset)

    # iteration callback
    def batch_end_callback(trainer):

        if trainer.iter_num % 20 == 0:
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")

        if trainer.iter_num % 500 == 0:
            # evaluate both the train and test score
            model.eval()
            """
            with torch.no_grad():
                # sample from the model...
                context = "O God, O God!"
                x = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long)[None,...].to(trainer.device)
                y = model.generate(x, 50, temperature=1.0, do_sample=True, top_k=10)[0]
                completion = ''.join([train_dataset.itos[int(i)] for i in y])
                print(completion)
            """
            # save the latest model
            print("saving model")
            ckpt_path = os.path.join(config.system.work_dir, "model.pt")
            torch.save(model.state_dict(), ckpt_path)
            # revert model to training mode
            model.train()
    
    trainer.set_callback('on_batch_end', batch_end_callback)

    # run the optimization
    trainer.run()

