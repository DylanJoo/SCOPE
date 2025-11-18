from glob import glob
from datasets import load_from_disk, DatasetDict
from datasets import concatenate_datasets

dataset_dict = {}

dataset_list = []
for i, file in enumerate(glob('/users/judylan1/datasets/*-train-*')):
    dataset_list.append( load_from_disk(file) )
    print(i)
merged = concatenate_datasets(dataset_list)
print(merged)

dataset_dict['train'] = merged
dataset_dict['eval'] = load_from_disk("/users/judylan1/datasets/crux-researchy-flatten.bert-base-uncased.pretokenized-eval")

tokenized_hf_dataset_dict = DatasetDict(
    {'train': dataset_dict['train'], 'eval': dataset_dict['eval']}
)

tokenized_hf_dataset_dict.save_to_disk("/users/judylan1/datasets/crux-researchy-flatten.bert-base-uncased.pretokenized")
