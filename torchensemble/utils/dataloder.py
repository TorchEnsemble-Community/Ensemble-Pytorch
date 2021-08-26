from torch.utils.data import DataLoader


class FixedDataLoader(object):
    def __init__(self, dataloader):
        # Check input
        if not isinstance(dataloader, DataLoader):
            msg = (
                "The input used to instantiate FixedDataLoader should be a"
                " DataLoader from `torch.utils.data`."
            )
            raise ValueError(msg)

        self.elem_list = []
        for _, elem in enumerate(dataloader):
            self.elem_list.append(elem)

    def __getitem__(self, index):
        return self.elem_list[index]

    def __len__(self):
        return len(self.elem_list)
