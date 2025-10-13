import torch
import numpy as np
from torch.utils.data import Dataset

def clean_data(data, corre):
    cleaned_data = []
    for idx in range(0, len(data), 100):
        data_item = np.array(data[idx:idx + 100])
        if corre is not None:
            if corre > 0:
                data_cla_label = 1 if float(data_item[0][1]) > 0 else 0
                data_reg_label = float(data_item[0][1])
            else:
                data_cla_label = 0 if float(data_item[0][1]) > 0 else 1
                data_reg_label = -float(data_item[0][1])
        else:
            data_cla_label = 1 if float(data_item[0][1]) > 0 else 0
            data_reg_label = float(data_item[0][1])
        data_item = data_item[:, 3:]
        data_item = data_item.astype(np.float)
        cleaned_data.append({"data": data_item, "cla_label": data_cla_label, "reg_label": data_reg_label})
    return cleaned_data

def clean_data_with_mean(data, corre, mean_evec, test_by_region=False):
    cleaned_data = []
    for idx in range(0, len(data), 100):
        mean_id = int(idx / 100)
        mean_value = mean_evec[mean_id]
        if test_by_region:
            mean_value = float(mean_value)
        else:
            if mean_value == "nan":
                print("find nan")
                print(mean_value)
                continue
            else:
                mean_value = float(mean_value)
        data_item = np.array(data[idx:idx + 100])
        if corre is not None:
            if corre > 0:
                data_cla_label = 1 if float(data_item[0][1]) > 0 else 0
                data_reg_label = float(data_item[0][1])
            else:
                data_cla_label = 0 if float(data_item[0][1]) > 0 else 1
                data_reg_label = -float(data_item[0][1])
        else:
            data_cla_label = 1 if float(data_item[0][1]) > 0 else 0
            data_reg_label = float(data_item[0][1])
        data_item = data_item[:, 3:]
        data_item = data_item.astype(np.float)
        cleaned_data.append({"data": data_item, "cla_label": data_cla_label, "reg_label": data_reg_label, "mean_evec": mean_value})
    return cleaned_data

def combine_data(all_data):
    combined_data = []
    for key, item in all_data.items():
        combined_data += item
    return combined_data



def combine_data_avg(all_data):
    print("6 features")
    combined_x = []
    combined_y = []
    for key, item in all_data.items():
        for data in item:
            mean_data = np.mean(np.array(data["data"]), axis=0)
            flatten_x = mean_data.flatten()
            combined_x.append(flatten_x)
            combined_y.append(data["cla_label"])
    return combined_x, combined_y

def combine_data_with_mean_avg(all_data):
    print("6 features + mean")
    combined_x = []
    combined_y = []
    for key, item in all_data.items():
        for data in item:
            mean_data = np.mean(np.array(data["data"]), axis=0)
            flatten_x = mean_data.flatten()
            appended_x = np.append(flatten_x, data["mean_evec"])
            combined_x.append(appended_x)
            combined_y.append(data["cla_label"])
    return combined_x, combined_y

def combine_data_avg_std(all_data):
    print("12 features")
    combined_x = []
    combined_y = []
    for key, item in all_data.items():
        for data in item:
            mean_data = np.mean(np.array(data["data"]), axis=0)
            flatten_mean = mean_data.flatten()
            std_data = np.std(np.array(data["data"]), axis=0)
            flatten_std = std_data.flatten()
            concat_mean_std = np.concatenate((flatten_mean, flatten_std))
            combined_x.append(concat_mean_std)
            combined_y.append(data["cla_label"])
    return combined_x, combined_y

def combine_data_with_mean_avg_std(all_data):
    print("12 features + mean")
    combined_x = []
    combined_y = []
    for key, item in all_data.items():
        for data in item:
            mean_data = np.mean(np.array(data["data"]), axis=0)
            flatten_mean = mean_data.flatten()
            std_data = np.std(np.array(data["data"]), axis=0)
            flatten_std = std_data.flatten()
            concat_mean_std = np.concatenate((flatten_mean, flatten_std))
            appended_x = np.append(concat_mean_std, data["mean_evec"])
            combined_x.append(appended_x)
            combined_y.append(data["cla_label"])
    return combined_x, combined_y

def random_split_data(combined_data, train_p, valid_p):
    num_training_data = int(len(combined_data) * train_p)
    num_validation_data = int(len(combined_data) * valid_p)
    num_testing_data = int(len(combined_data) - num_training_data - num_validation_data)

    index_list = np.arange(0, len(combined_data), 1)
    np.random.shuffle(index_list)

    training_data_index = index_list[:num_training_data]
    validation_data_index = index_list[num_training_data:num_training_data + num_validation_data]
    testing_data_index = index_list[num_training_data + num_validation_data:]

    training_data = []
    validation_data = []
    testing_data = []

    for idx in training_data_index:
        training_data.append(combined_data[idx])
    for idx in validation_data_index:
        validation_data.append(combined_data[idx])
    for idx in testing_data_index:
        testing_data.append(combined_data[idx])

    print("{} training data, {} validation data, {} testing data".format(len(training_data), len(validation_data), len(testing_data)))

    return training_data, validation_data, testing_data

class abDataset(Dataset):
    def __init__(self, data_dict):
        self.data = []
        self.cla_labels = []
        self.reg_labels = []
        for item in data_dict:
            data = torch.from_numpy(item["data"]).float()
            # 确保输入形状为 [seq_len=100, input_dim=6]
            data = data.view(100, 6)  # 强制重塑为 (100, 6)
            self.data.append(data)
            self.cla_labels.append(torch.tensor(item["cla_label"]))
            self.reg_labels.append(torch.tensor(item["reg_label"]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "input": self.data[idx],  # [seq_len, input_dim]
            "cla_labels": self.cla_labels[idx],
            "reg_labels": self.reg_labels[idx],
        }

class abDataset_with_mean(abDataset):
    def __init__(self, data_dict):
        super().__init__(data_dict)
        self.mean_evec = []
        for item in data_dict:
            self.mean_evec.append(torch.tensor(item["mean_evec"]))

    def __getitem__(self, idx):
        base_item = super().__getitem__(idx)
        base_item["mean_evec"] = self.mean_evec[idx]
        return base_item

# 其他函数和类保持不变...