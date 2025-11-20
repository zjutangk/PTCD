from model.models import SetBert
from dataloader import Data
from init_parameter import init_model
import numpy as np
from utils.tools import *
parser = init_model()
args = parser.parse_args()
print(args)
num_labels = 4
LABEL_LIST = ['ISTJ', 'ISFJ', 'INFJ', 'INTJ', 'ISTP', 'ISFP', 'INFP', 'INTP', 'ESTP', 'ESFP', 'ENFP', 'ENTP', 'ESTJ', 'ESFJ', 'ENFJ', 'ENTJ']

# hyperparameters
train_csv_path = ""
test_csv_path = (
    "" 
)

dataset = Data(args.train_csv_file, args.test_csv_file, args.train_ratio, args)
test_data, test_dataloader = (
            dataset.test_examples,
            dataset.test_dataloader,
        )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for batch in tqdm(test_dataloader, desc="Iteration"):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            input_ids = input_ids.view(-1, input_ids.size(-1))
            
            print(input_ids.shape)