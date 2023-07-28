from torch.utils.data import Dataset


class FileReader:
    """
    This class reads the original data and performs data proprocessing.
    """
    def __init__(self, root):
        pass
    
    def preprocess(self, data):
        pass


class UserDataset(Dataset):
    """
    User defined Dataset.
    args: a namespace object containing some parameters.
    file_container: an object that reads and contains some file.
    """

    def __init__(self, args, file_container):
        self.args = args
        self.file_container = file_container
        pass
    
    def __getitem__(self, index):
        """
        This methods must be implemented to convert the dataset into torch.utils.data.DataLoader.
        index: the index of the item.
        """
        pass
    
    def __len__(self):
        """
        This methods must be implemented to convert the dataset into torch.utils.data.DataLoader.
        Return the length of the dataset. Note that the __getitem__ methods will traversal from 0 ~ __len__(self) - 1
        """
        pass

    @staticmethod
    def collate_fn(batch):
        """
        This methods collate a list of items, where each item is an object returned from the __getitem__ methods.
        """
        pass
            
