from utils.tools import *
from torch.utils.data import Dataset,DataLoader
import ast
from torch.nn.utils.rnn import pad_sequence



class CustomDataset(Dataset):
    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        input_ids = self.input_ids[idx]
        input_mask = self.input_mask[idx]
        segment_ids = self.segment_ids[idx]
        label_ids = self.label_ids[idx]
        return input_ids, input_mask, segment_ids, label_ids
    
def custom_collate_fn(batch):
    
    return pad_sequence(batch, batch_first=True, padding_value=0)





class Data:

    def __init__(self, train_csv, test_csv, ratio, args):
        self.max_seq_length = 50

        processor = DatasetProcessor()
        self.label_list = [
            "ISTJ",
            "ISFJ",
            "INFJ",
            "INTJ",
            "ISTP",
            "ISFP",
            "INFP",
            "INTP",
            "ESTP",
            "ESFP",
            "ENFP",
            "ENTP",
            "ESTJ",
            "ESFJ",
            "ENFJ",
            "ENTJ",
        ]

        self.num_labels = len(self.label_list)
        self.train_csv = train_csv
        self.test_csv = test_csv

        self.train_examples = self.get_examples(processor, "train")
        self.test_examples = self.get_examples(processor, "test")
        # print(len(self.train_examples), len(self.test_examples))
        self.train_dataloader = self.get_loader(self.train_examples, "train", args)
        self.test_dataloader = self.get_loader(self.test_examples, "test", args)
        #(input_ids:list[tensor], input_mask:list[tensor], segment_ids:list[tensor], label_ids:tensor)

    def get_examples(self, processor, mode="train"):
        if mode == "train":
            ori_examples = processor.get_examples(self.train_csv, mode)
            return ori_examples
        else:
            ori_examples = processor.get_examples(self.test_csv, mode)
            return ori_examples

    def get_loader(self, examples, mode, args):
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        features = convert_examples_to_features(
            examples, self.max_seq_length, tokenizer
        )
        input_ids=[]
        input_mask=[]
        segment_ids=[]
        label_ids=[]
        for p in features:
            num=0
            for f in p:
                num+=1
            print(num)
            input_ids_p = torch.tensor([f.input_ids for f in p],dtype=torch.long)
            input_mask_p = torch.tensor([f.input_mask for f in p],dtype=torch.long)
            segment_ids_p = torch.tensor([f.segment_ids for f in p],dtype=torch.long)
            input_ids.append(input_ids_p)
            input_mask.append(input_mask_p)
            segment_ids.append(segment_ids_p)
            label_ids.append(p[0].label_id)
            
        input_ids=pad_sequence(input_ids, batch_first=True, padding_value=0)
        input_mask=pad_sequence(input_mask, batch_first=True, padding_value=0)
        segment_ids=pad_sequence(segment_ids, batch_first=True, padding_value=0)
        input_ids = torch.tensor(input_ids)
        input_mask = torch.tensor(input_mask)
        segment_ids = torch.tensor(segment_ids)
       
        label_ids = torch.tensor(label_ids)
        data = TensorDataset(input_ids, input_mask, segment_ids, label_ids)

        if mode == "train":
            sampler = RandomSampler(data)
            dataloader = DataLoader(
                data, sampler=sampler, batch_size=args.pretrain_batch_size
            )
        elif mode in ["eval", "test"]:
            sampler = SequentialSampler(data)
            dataloader = DataLoader(
                data, sampler=sampler, batch_size=args.eval_batch_size
            )
        else:
            raise NotImplementedError(f"Mode {mode} not found")

        return dataloader






class InputExample(object):

    def __init__(self, guid, text_list, text_b=None, label=None):
        
        self.guid = guid
        self.text_list = text_list
        self.text_b = text_b
        self.label = label


class InputFeatures(object):

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                line = [l.lower() for l in line]
                lines.append(line)
            return lines


class DatasetProcessor(DataProcessor):

    def get_examples(self, csv_path, mode):
        if mode == "train":
            return self._create_examples(self._read_tsv(csv_path), "train")
        else:
            return self._create_examples(self._read_tsv(csv_path), "test")

    def get_labels(self, csv_path):
        """See base class."""
        import pandas as pd

        test = pd.read_csv(csv_path, sep="\t")
        labels = [str(label).lower() for label in test["choice"]]
        labels = np.unique(np.array(labels))
        return labels

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for i, line in enumerate(lines):
            # print(len(line))
            if i == 0:
                continue
            if len(line) != 6:
                continue
            guid = "%s-%s" % (set_type, i)
            text_list = line[5]
            label = line[1]
            # print(text_a,label)

            examples.append(
                InputExample(guid=guid, text_list=text_list, text_b=None, label=label)
            )
        return examples

def convert_mbti_to_labellist(mbti):
    multi_label=np.array([0,0,0,0])
    
    if "e" in mbti or "E" in mbti:
        multi_label[0]=1
    if "n" in mbti or "N" in mbti:
        multi_label[1]=1
    if "f" in mbti or "F" in mbti:
        multi_label[2]=1
    if "p" in mbti or "P" in mbti:
        multi_label[3]=1
    
    # print(multi_label)
    return multi_label

def convert_examples_to_features(examples,  max_seq_length, tokenizer):

    features = []
    for example in examples:
        features_person=[]
        num=0
        example.text_list = ast.literal_eval(example.text_list)
        for ex_index, sentence in enumerate(example.text_list):
            num+=1
            tokens_a = tokenizer.tokenize(sentence)

            tokens_b = None
            if example.text_b:
                tokens_b = tokenizer.tokenize(example.text_b)
                _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
            else:
                # Account for [CLS] and [SEP] with "- 2"
                if len(tokens_a) > max_seq_length - 2:
                    tokens_a = tokens_a[: (max_seq_length - 2)]
            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)

            if tokens_b:
                tokens += tokens_b + ["[SEP]"]
                segment_ids += [1] * (len(tokens_b) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

           
            input_mask = [1] * len(input_ids)

            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            label_id = convert_mbti_to_labellist(example.label)
            features_person.append(
                InputFeatures(
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    label_id=label_id,
                )
            )
        # print(num)
        features.append(features_person)
        
        
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop(0)  
        else:
            tokens_b.pop()
