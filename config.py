import os
import torch
import spacy
import sys
from time import sleep
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from torchtext.data import Field, BucketIterator, Dataset, Example
from torch.utils.tensorboard import SummaryWriter

# Load dataset here
en_vi_set = load_dataset("mt_eng_vietnamese", "iwslt2015-en-vi") # en-vi

# Add more language model here
spacy_en = spacy.load('en_core_web_sm')
spacy_vi = spacy.load('vi_core_news_lg')

# Setup tokenizer
def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

def tokenize_vi(text):
    return [tok.text for tok in spacy_vi.tokenizer(text)]

# Language setup
english = Field(
    tokenize=tokenize_en, lower=True, init_token="<sos>", eos_token="<eos>"
)
vietnamese = Field(
    tokenize=tokenize_vi, lower=True, init_token="<sos>", eos_token="<eos>"
)

fields = (('src', english), ('trg', vietnamese))

# Setting

WORKINGDIR = os.getcwd() +"/"  
CHECKPOINT_FILE = "En-Vi.pth.tar"
SAVE_PATH_MODEL = WORKINGDIR + "models/"
SAVE_PATH_VOCAB = WORKINGDIR + "vocabs/"


# Training hyperparameters
num_epochs = 5
learning_rate = 2e-5
batch_size = 16 

# Model hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_model = True
save_model = True
embedding_size = 512
num_heads = 8
num_encoder_layers = 3
num_decoder_layers = 3
dropout = 0.10
max_len = 512
forward_expansion = 2048


# Tensorboard to get nice loss plot
writer_loss = SummaryWriter("runs/loss_plot/En-Vi")
writer_score = SummaryWriter("runs/score_plot/En-Vi")

# tqdm bar format
BARFORMAT = "{desc}{percentage:3.0f}%│{bar:30}│total: {n_fmt}/{total_fmt} [{elapsed} - {remaining},{rate_fmt}{postfix}]"
COLOR = "red"
COLOR_COMPLETE = "green"
ASCII = ' ▌█'
DELAYTIME = 0.5
