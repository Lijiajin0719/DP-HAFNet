import torch
import mat73
import numpy as np
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, file_paths):
        """Initialize custom dataset"""

        self.data = []
        self.target1 = []
        self.target2 = []
        self.test_data = []
        self.test_target1 = []
        self.test_target2 = []

        for file_path in file_paths:
            import_data = mat73.loadmat(file_path)

            if "breast_24" in file_path or "breast_26" in file_path or "breast_28" in file_path:
                for i in range(0, 58):
                    self.data.append(import_data['DL_data'][:, :, i])
                    self.target1.append(import_data['DL_data'][:, :, 73])
                    self.target2.append(import_data['DL_data'][:, :, 74])
                for i in range(58, 73):
                    self.test_data.append(import_data['DL_data'][:, :, i])
                    self.test_target1.append(import_data['DL_data'][:, :, 73])
                    self.test_target2.append(import_data['DL_data'][:, :, 74])
            else:
                for i in range(0, 60):
                    self.data.append(import_data['DL_data'][:, :, i])
                    self.target1.append(import_data['DL_data'][:, :, 75])
                    self.target2.append(import_data['DL_data'][:, :, 76])
                for i in range(60, 75):
                    self.test_data.append(import_data['DL_data'][:, :, i])
                    self.test_target1.append(import_data['DL_data'][:, :, 75])
                    self.test_target2.append(import_data['DL_data'][:, :, 76])

    def __len__(self):
        """Return the size of the dataset"""

        return len(self.data)

    def __getitem__(self, idx):
        """Get a single sample"""

        data = torch.from_numpy(self.data[idx]).unsqueeze(0)
        return {
            'input': data,
            'target1': torch.from_numpy(self.target1[idx]).unsqueeze(0),
            'target2': torch.from_numpy(self.target2[idx]).unsqueeze(0)
        }

    def get_test_data(self):
        """Get the test set data"""

        return self.test_data, self.test_target1, self.test_target2


if __name__ == "__main__":
    file_paths = ["../data/DLdata_breast_24.mat"]
    dataset = CustomDataset(file_paths)
    print(f"Dataset size: {len(dataset)}")
    print(f"Test data size: {len(dataset.test_data)}")

    sample = dataset[0]
    print(f"Input shape: {sample['input'].shape}")
    print(f"Target1 shape: {sample['target1'].shape}")
    print(f"Target2 shape: {sample['target2'].shape}")