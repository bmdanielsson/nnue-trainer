import numpy as np
import ctypes
import torch
import os
import sys
import glob
from torch.utils.data import Dataset


local_dllpath = [n for n in glob.glob('./*dataloader.*') if n.endswith('.so') or n.endswith('.dll') or n.endswith('.dylib')]
if not local_dllpath:
    print('Cannot find data_loader shared library.')
    sys.exit(1)
dllpath = os.path.abspath(local_dllpath[0])
dll = ctypes.cdll.LoadLibrary(dllpath)


class SparseBatch(ctypes.Structure):
    _fields_ = [
        ('num_inputs', ctypes.c_int),
        ('size', ctypes.c_int),
        ('is_white', ctypes.POINTER(ctypes.c_float)),
        ('outcome', ctypes.POINTER(ctypes.c_float)),
        ('score', ctypes.POINTER(ctypes.c_float)),
        ('num_active_white_features', ctypes.c_int),
        ('num_active_black_features', ctypes.c_int),
        ('white', ctypes.POINTER(ctypes.c_int)),
        ('black', ctypes.POINTER(ctypes.c_int)),
        ('white_values', ctypes.POINTER(ctypes.c_float)),
        ('black_values', ctypes.POINTER(ctypes.c_float))
    ]


    def get_tensors_cpu(self):
        us = torch.from_numpy(np.ctypeslib.as_array(self.is_white, shape=(self.size, 1))).clone()
        them = 1.0 - us
        outcome = torch.from_numpy(np.ctypeslib.as_array(self.outcome, shape=(self.size, 1))).clone()
        score = torch.from_numpy(np.ctypeslib.as_array(self.score, shape=(self.size, 1))).clone()
        iw = torch.from_numpy(np.ctypeslib.as_array(self.white, shape=(self.num_active_white_features, 2)).transpose()).clone()
        ib = torch.from_numpy(np.ctypeslib.as_array(self.black, shape=(self.num_active_white_features, 2)).transpose()).clone()
        white_values = torch.from_numpy(np.ctypeslib.as_array(self.white_values, shape=(self.num_active_white_features,))).clone()
        black_values = torch.from_numpy(np.ctypeslib.as_array(self.black_values, shape=(self.num_active_black_features,))).clone()
        white = torch.sparse.FloatTensor(iw.long(), white_values, (self.size, self.num_inputs))
        black = torch.sparse.FloatTensor(ib.long(), black_values, (self.size, self.num_inputs))
        return us, them, white, black, outcome, score


    def get_tensors(self, device):
        white_values = torch.from_numpy(np.ctypeslib.as_array(self.white_values, shape=(self.num_active_white_features,))).pin_memory().to(device=device, non_blocking=True)
        black_values = torch.from_numpy(np.ctypeslib.as_array(self.black_values, shape=(self.num_active_black_features,))).pin_memory().to(device=device, non_blocking=True)
        iw = torch.transpose(torch.from_numpy(np.ctypeslib.as_array(self.white, shape=(self.num_active_white_features, 2))).pin_memory().to(device=device, non_blocking=True), 0, 1).long()
        ib = torch.transpose(torch.from_numpy(np.ctypeslib.as_array(self.black, shape=(self.num_active_white_features, 2))).pin_memory().to(device=device, non_blocking=True), 0, 1).long()
        us = torch.from_numpy(np.ctypeslib.as_array(self.is_white, shape=(self.size, 1))).pin_memory().to(device=device, non_blocking=True)
        them = 1.0 - us
        outcome = torch.from_numpy(np.ctypeslib.as_array(self.outcome, shape=(self.size, 1))).pin_memory().to(device=device, non_blocking=True)
        score = torch.from_numpy(np.ctypeslib.as_array(self.score, shape=(self.size, 1))).pin_memory().to(device=device, non_blocking=True)
        white = torch._sparse_coo_tensor_unsafe(iw, white_values, (self.size, self.num_inputs))
        black = torch._sparse_coo_tensor_unsafe(ib, black_values, (self.size, self.num_inputs))
        white._coalesced_(True)
        black._coalesced_(True)
        return us, them, white, black, outcome, score


SparseBatchPtr = ctypes.POINTER(SparseBatch)

create_sparse_batch_stream = dll.create_sparse_batch_stream
create_sparse_batch_stream.restype = ctypes.c_void_p
create_sparse_batch_stream.argtypes = [ctypes.c_char_p, ctypes.c_ulonglong, ctypes.c_int, ctypes.c_int]
destroy_sparse_batch_stream = dll.destroy_sparse_batch_stream
destroy_sparse_batch_stream.argtypes = [ctypes.c_void_p]
fetch_next_sparse_batch = dll.fetch_next_sparse_batch
fetch_next_sparse_batch.restype = SparseBatchPtr
fetch_next_sparse_batch.argtypes = [ctypes.c_void_p]
destroy_sparse_batch = dll.destroy_sparse_batch


class SparseBatchProvider:
    def __init__(self, filename, nsamples, batch_size, use_factorizer,
                 device='cpu'):
        self.create_stream = create_sparse_batch_stream
        self.destroy_stream = destroy_sparse_batch_stream
        self.fetch_next = fetch_next_sparse_batch
        self.destroy_batch = destroy_sparse_batch
        self.filename = filename.encode('utf-8')
        self.nsamples = nsamples
        self.batch_size = batch_size
        self.use_factorizer = use_factorizer
        self.device = device

        self.stream = self.create_stream(self.filename, nsamples, batch_size,
                                         use_factorizer)


    def __iter__(self):
        return self


    def __next__(self):
        v = self.fetch_next(self.stream)

        if v:
            if self.device == 'cpu':
                tensors = v.contents.get_tensors_cpu()
            else:
                tensors = v.contents.get_tensors(self.device)
            self.destroy_batch(v)
            return tensors
        else:
            raise StopIteration


    def __del__(self):
        self.destroy_stream(self.stream)


class SparseBatchDataset(torch.utils.data.IterableDataset):
    def __init__(self, filename, nsamples, batch_size, use_factorizer,
                 num_batches, device='cpu'):
        super(SparseBatchDataset).__init__()
        self.filename = filename
        self.nsamples = nsamples
        self.batch_size = batch_size
        self.use_factorizer = use_factorizer
        self.device = device
        self.num_batches = num_batches


    def __len__(self):
        return self.num_batches


    def __iter__(self):
        return SparseBatchProvider(self.filename, self.nsamples,
                                   self.batch_size, self.use_factorizer,
                                   device=self.device)
