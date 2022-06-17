import torch
import numpy as np
import collections
from torch.utils.data import Sampler, WeightedRandomSampler


def oversampler(labels):
    labels = torch.Tensor(labels)
    uni_label = torch.unique(labels) 
    uni_label_sorted, _  = torch.sort(uni_label)
    uni_label_sorted = uni_label_sorted.detach().numpy()
    label_bin = torch.bincount(labels.int()) #.detach().numpy()
    sample_weights = 1. / label_bin

    print(sample_weights, len(sample_weights))

    w_sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    return w_sampler


class OverSampler(Sampler):
    # 이상하게 1번만 돌고 자꾸 멈춤...
    def __init__(self, labels, batch_size):
        super(Sampler, self).__init__()

        self.labels = labels
        labels = torch.Tensor(labels)
        self.batch_size = batch_size
        
        self.indices = list(range(len(labels)))
        uni_label = torch.unique(labels) 
        
        uni_label_sorted, _  = torch.sort(uni_label)
        uni_label_sorted = uni_label_sorted.detach().numpy()
        label_bin = torch.bincount(labels.int()).detach().numpy()
        label_to_count = dict(zip(uni_label_sorted , label_bin))
        weights = [ len(labels) / label_to_count[float(label)] for label in labels]
        self.weights = torch.DoubleTensor(weights)

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.batch_size, replacement=True))

    def __len__(self):
        return len(self.labels)
    
    
class FocusSampler(Sampler):
    def __init__(self, labels, args):
        self.labels = labels
        self.d_len = len(self.labels) - (len(self.labels) % args.batch_size)
        self.batch_size = args.batch_size
        self.sampling_type = args.sampling_type
        self.IB_ratio = args.IB_ratio
        
    def __len__(self):
        return self.d_len
    
    def __iter__(self):
        idx_list = [0] * self.d_len
        
        label_list = np.array(self.labels)
        oob_ids = np.where(label_list == 1)[0]
        ib_ids = np.where(label_list == 0)[0]
        
        if self.sampling_type == 1:
            cover = self.batch_size // 4
            
            for i in range(self.d_len // self.batch_size):
                while True:
                    idx = np.random.randint(0, len(oob_ids), 1)[0]
                    
                    if oob_ids[idx] - cover >= 0 and oob_ids[idx] + cover < len(label_list):
                        lbs = label_list[oob_ids[idx] - cover:oob_ids[idx] + cover]
                        if sum(lbs) != len(lbs):
                            break
                
                ib_samples = np.random.choice(ib_ids, self.batch_size // 4)
                oob_samples = np.random.choice(oob_ids, self.batch_size // 4)
                lbs = np.concatenate((lbs, label_list[ib_samples], label_list[oob_samples]), -1)
                
                idx_list[i*self.batch_size:(i+1)*self.batch_size] = lbs
        elif self.sampling_type == 2:
            cover = self.batch_size // 2
            
            for i in range(self.d_len // self.batch_size):
                while True:
                    idx = np.random.randint(0, len(oob_ids), 1)[0]
                    
                    if oob_ids[idx] - cover >= 0 and oob_ids[idx] + cover < len(label_list):
                        lbs = label_list[oob_ids[idx] - cover:oob_ids[idx] + cover]
                        if sum(lbs) != len(lbs):
                            break
                        
                idx_list[i*self.batch_size:(i+1)*self.batch_size] = lbs
        elif self.sampling_type == 3:
            cover = self.batch_size // (self.IB_ratio + 1)
            
            for i in range(self.d_len // self.batch_size):
                while True:
                    idx = np.random.randint(0, len(oob_ids), 1)[0]
                    
                    if oob_ids[idx] - cover >= 0 and oob_ids[idx] + cover < len(label_list):
                        lbs = label_list[int(oob_ids[idx] - cover):int(oob_ids[idx] + cover)]
                        if sum(lbs) != len(lbs) and len(lbs)/sum(lbs) >= self.IB_ratio:
                            break
                tmp_label_list = np.concatenate((label_list[:int(oob_ids[idx]-cover)], label_list[int(oob_ids[idx]+cover):]), -1)
                
                tmp_oob_ids = np.where(tmp_label_list == 1)[0]
                tmp_ib_ids = np.where(tmp_label_list == 0)[0]
        
                n_oob = int(cover - sum(lbs))
                n_ib = int(self.IB_ratio * cover - n_oob)
                
                ib_samples = np.random.choice(tmp_ib_ids, n_ib)
                
                if n_oob > 0:
                    oob_samples = np.random.choice(tmp_oob_ids, n_oob)
                    lbs = np.concatenate((lbs, tmp_label_list[ib_samples], tmp_label_list[oob_samples]), -1)
                else:
                    lbs = np.concatenate((lbs, tmp_label_list[ib_samples]), -1)
                
                idx_list[i*self.batch_size:(i+1)*self.batch_size] = lbs
        elif self.sampling_type == 4:
            cover = self.batch_size // (self.IB_ratio + 1)
            hf_cover = self.batch_size // 2
            
            for i in range(self.d_len // self.batch_size):
                while True:
                    idx = np.random.randint(0, len(oob_ids), 1)[0]
                    
                    if oob_ids[idx] - cover >= 0 and oob_ids[idx] + cover < len(label_list):
                        lbs = label_list[int(oob_ids[idx] - cover):int(oob_ids[idx] + cover)]
                        if sum(lbs) != len(lbs) and len(lbs)/sum(lbs) >= self.IB_ratio:
                            break
                tmp_label_list = np.concatenate((label_list[:int(oob_ids[idx]-cover)], label_list[int(oob_ids[idx]+cover):]), -1)
                
                tmp_oob_ids = np.where(tmp_label_list == 1)[0]
                tmp_ib_ids = np.where(tmp_label_list == 0)[0]
        
                n_oob = int(hf_cover - sum(lbs))
                n_ib = int(hf_cover - np.count_nonzero(lbs))
                
                ib_samples = np.random.choice(tmp_ib_ids, n_ib)
                
                if n_oob > 0:
                    oob_samples = np.random.choice(tmp_oob_ids, n_oob)
                    lbs = np.concatenate((lbs, tmp_label_list[ib_samples], tmp_label_list[oob_samples]), -1)
                else:
                    lbs = np.concatenate((lbs, tmp_label_list[ib_samples]), -1)
                
                idx_list[i*self.batch_size:(i+1)*self.batch_size] = lbs
            
        return iter(idx_list)


class MPerClassSampler(Sampler):
    """
    At every iteration, this will return m samples per class. For example,
    if dataloader's batchsize is 100, and m = 5, then 20 classes with 5 samples
    each will be returned
    """

    def __init__(self, labels, m, batch_size=None, length_before_new_iter=100000):
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
        self.m_per_class = int(m)
        self.batch_size = int(batch_size) if batch_size is not None else batch_size
        self.labels_to_indices = self.get_labels_to_indices(labels)
        self.labels = list(self.labels_to_indices.keys())
        self.length_of_single_pass = self.m_per_class * len(self.labels)
        self.list_size = length_before_new_iter
        if self.batch_size is None:
            if self.length_of_single_pass < self.list_size:
                self.list_size -= (self.list_size) % (self.length_of_single_pass)
        else:
            assert self.list_size >= self.batch_size
            assert (
                self.length_of_single_pass >= self.batch_size
            ), "m * (number of unique labels) must be >= batch_size"
            assert (
                self.batch_size % self.m_per_class
            ) == 0, "m_per_class must divide batch_size without any remainder"
            self.list_size -= self.list_size % self.batch_size

    def __len__(self):
        return self.list_size

    def __iter__(self):
        idx_list = [0] * self.list_size
        i = 0
        num_iters = self.calculate_num_iters()
        for _ in range(num_iters):
            np.random.shuffle(self.labels)

            if self.batch_size is None:
                curr_label_set = self.labels
            else:
                curr_label_set = self.labels[: self.batch_size // self.m_per_class]
            for label in curr_label_set:
                t = self.labels_to_indices[label]
                idx_list[i : i + self.m_per_class] = self.safe_random_choice(
                    t, size=self.m_per_class
                )
                i += self.m_per_class
        return iter(idx_list)

    def calculate_num_iters(self):
        divisor = (
            self.length_of_single_pass if self.batch_size is None else self.batch_size
        )
        return self.list_size // divisor if divisor < self.list_size else 1

    def safe_random_choice(self, input_data, size):
        """
        Randomly samples without replacement from a sequence. It is "safe" because
        if len(input_data) < size, it will randomly sample WITH replacement
        Args:
            input_data is a sequence, like a torch tensor, numpy array,
                            python list, tuple etc
            size is the number of elements to randomly sample from input_data
        Returns:
            An array of size "size", randomly sampled from input_data
        """
        replace = len(input_data) < size
        return np.random.choice(input_data, size=size, replace=replace)

    def get_labels_to_indices(self, labels):
        """
        Creates labels_to_indices, which is a dictionary mapping each label
        to a numpy array of indices that will be used to index into self.dataset
        """
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()
        labels_to_indices = collections.defaultdict(list)
        for i, label in enumerate(labels):
            labels_to_indices[label].append(i)
        for k, v in labels_to_indices.items():
            labels_to_indices[k] = np.array(v, dtype=np.int)
        return labels_to_indices