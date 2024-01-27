import torch
import mmap
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_random_chunk(split,batch_size,block_size,encode):
    filename = "output_train.txt" if split == 'train' else 'output_val.txt'
    with open(filename,'rb') as f:
        with mmap.mmap(f.fileno(), 0,access=mmap.ACCESS_READ) as mm:
            file_size = len(mm)
            start_pos = random.randint(0,(file_size) - block_size * batch_size )
            mm.seek(start_pos)
            block = mm.read(block_size *batch_size-1)

            decoded_block = block.decode('utf-8',errors='ignore').replace('\r','')

            data = torch.tensor(encode(decoded_block),dtype=torch.long)
    
    return data


def get_batch(split,batch_size,block_size,encode):
    data = get_random_chunk(split=split,batch_size=batch_size,block_size=block_size,encode=encode)
    ix = torch.randint(len(data) - block_size,(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x,y = x.to(device),y.to(device)

    return x,y