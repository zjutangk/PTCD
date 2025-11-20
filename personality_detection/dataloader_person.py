import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import ast
from torch.utils.data import WeightedRandomSampler


def convert_mbti_to_labellist(mbti):
    multi_label = np.array([0, 0, 0, 0])
    # print(mbti)
    if "e" in mbti or "E" in mbti:
        multi_label[0] = 1
    if "n" in mbti or "N" in mbti:
        multi_label[1] = 1
    if "f" in mbti or "F" in mbti:
        multi_label[2] = 1
    if "j" in mbti or "J" in mbti:
        multi_label[3] = 1

    # print(multi_label)
    return multi_label


class PersonFeatureDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

        self.grouped_data = {}
        for person in self.df["person"].unique():
            person_data = self.df[self.df["person"] == person]
            features = [
                torch.tensor(np.array(ast.literal_eval(feat)).squeeze())
                for feat in person_data["sentence"]
            ]
            labels = convert_mbti_to_labellist(
                person_data["label"].iloc[0]
            )  
            self.grouped_data[person] = {"features": features, "label": labels}

        self.person_list = list(self.grouped_data.keys())

    def __len__(self):
        return len(self.person_list)

    def __getitem__(self, idx):
        person = self.person_list[idx]
        return {
            "features": self.grouped_data[person]["features"],
            "label": self.grouped_data[person]["label"],
            "person": person,
        }


def collate_fn(batch):
    max_length = max([len(item["features"]) for item in batch])

    padded_features = []
    labels = []
    persons = []

    for item in batch:
        features = item["features"]
        features_tensor = torch.stack(features)
        if len(features) < max_length:
            padding = torch.zeros((max_length - len(features), 768))
            features_tensor = torch.cat([features_tensor, padding], dim=0)

        padded_features.append(features_tensor)
        labels.append(item["label"])
        persons.append(item["person"])

    features_batch = torch.stack(padded_features)
    labels_batch = torch.tensor(labels)

    return {
        "features": features_batch,  # shape: [batch_size, max_length, 768]
        "labels": labels_batch,  # shape: [batch_size]
        "persons": persons,  
    }


# 使用示例
def create_dataloader(csv_path, batch_size):
    dataset = PersonFeatureDataset(csv_path)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    return dataset, dataloader


class BalancedPersonFeatureDataset(PersonFeatureDataset):
    def __init__(self, csv_path):
        super().__init__(csv_path)

        self.label_groups = {dim: {0: [], 1: []} for dim in range(4)}
        for person in self.person_list:
            labels = self.grouped_data[person]["label"]
            for dim in range(4):
                self.label_groups[dim][labels[dim]].append(person)

        max_counts = [
            max(len(groups[0]), len(groups[1])) for groups in self.label_groups.values()
        ]

        self.balanced_person_list = []
        for dim in range(4):
            for label in [0, 1]:
                persons = self.label_groups[dim][label]
                if len(persons) < max_counts[dim]:
                    additional = np.random.choice(
                        persons, size=max_counts[dim] - len(persons), replace=True
                    )
                    self.balanced_person_list.extend(additional)

        
        self.balanced_person_list.extend(self.person_list)

    def __len__(self):
        return len(self.balanced_person_list)

    def __getitem__(self, idx):
        person = self.balanced_person_list[idx]
        return {
            "features": self.grouped_data[person]["features"],
            "label": self.grouped_data[person]["label"],
            "person": person,
        }


def create_balanced_dataloader(csv_path, batch_size):
    dataset = BalancedPersonFeatureDataset(csv_path)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    return dataset, dataloader


