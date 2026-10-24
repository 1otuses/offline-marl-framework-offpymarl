import os
import h5py
import torch as th
import numpy as np

############## DataBatch ##############
class OfflineDataBatch():
    def __init__(self, data, batch_size, max_seq_length, device='cpu') -> None:
        self.data = data
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length # 若使用完整长度则为 None
        self.device = device
        for k, v in self.data.items():
            # (batch_size, T, n_agents, *shape)
            # 在此处截断，直接与 OfflineBuffer 对接
            self.data[k] = v[:, :max_seq_length].to(self.device)
    
    def __getitem__(self, item):
        if isinstance(item, str):
            if item in self.data:
                return self.data[item]
            elif hasattr(self, item):
                return getattr(self, item)
            else:
                raise ValueError('Cannot index OfflineDataBatch with key "{}"'.format(item))
        else:
            raise ValueError('Cannot index OfflineDataBatch with key "{}"'.format(item))

    def to(self, device=None):
        if device is None:
            device = self.device
        for k, v in self.data.items():
            self.data[k] = v.to(device)
        self.device = device # 更新 self.device
    
    def keys(self):
        return list(self.data.keys())
    
    def assign(self, key, value):
        if key in self.data:
            assert 0, "Cannot assign to existing key"
        self.data[key] = value


############## OfflineBuffer ##############
class OfflineBufferH5(): # 单任务
    def __init__(self, args, map_name, quality,
                 data_path='', # 最深层目录
                 max_buffer_size=2000,
                 device='cpu',
                 shuffle=True) -> None:
        self.args = args
        self.base_data_folder = args.offline_data_folder
        self.map_name = map_name 
        # 地图名称是一个抽象名称，可以是sc2中的map_name / mpe中的scenario_name
        self.quality = quality
        # 设置数据路径
        if not isinstance(data_path, list) and os.path.exists(data_path):
            self.data_path_list = [data_path] 
        else:
            self.data_path_list = []
            for i, quality_i in enumerate(quality.split("_")): # 例如 "medium_expert"
                data_path_i = data_path[i] if isinstance(data_path, list) else data_path
                self.data_path_list.append(self.get_data_path(map_name, quality_i, data_path_i))

        self.h5_paths = []
        for final_data_path in self.data_path_list:
            self.h5_paths.extend([os.path.join(final_data_path, f) for f in sorted(os.listdir(final_data_path)) if f.endswith(".h5")])
        print(self.h5_paths)
        if not self.h5_paths:
            raise ValueError("No HDF5 datasets found in {}".format(self.data_path_list))
        #self.h5_paths = [os.path.join(self.data_path, f) for f in os.listdir(self.data_path)]

        self.max_buffer_size = max_buffer_size
        self.device = device # 设备实际上无法工作。
        self.shuffle = shuffle
        dataset = [self._read_data(path) for path in self.h5_paths]
        self.data = {
            k: np.concatenate([v[k] for v in dataset], axis=0) for k in dataset[0].keys()
        }
        self.keys = list(self.data.keys())
        self.buffer_size = self.data[self.keys[0]].shape[0]

        if shuffle:
            # 再次洗牌
            shuffled_idx = np.random.choice(self.buffer_size, self.buffer_size, replace=False)
            self.data = {k: v[shuffled_idx] for k, v in self.data.items()}
        if max_buffer_size > 0 and self.buffer_size > max_buffer_size:
            self.data = {k: v[:max_buffer_size] if shuffle else v[-max_buffer_size:]
                         for k, v in self.data.items()}
            self.buffer_size = max_buffer_size
        
    
    def get_data_path(self, map_name, quality, data_path):
        data_path = os.path.join(self.base_data_folder, self.args.env, map_name, quality, data_path)
        if all([".h5" not in f for f in os.listdir(data_path)]):
            # 自动查找文件夹
            existing_folders = [f for f in sorted(os.listdir(data_path))
                                if os.path.isdir(os.path.join(data_path, f))
                                and any(name.endswith(".h5") for name in os.listdir(os.path.join(data_path, f)))]
            assert len(existing_folders) > 0
            return os.path.join(data_path, existing_folders[-1])
        else:
            return data_path   

    def _read_data(self, h5_path):
        data = {}
        with h5py.File(h5_path, 'r') as f:
            for k in f.keys():
                added_data = f[k][:]
                if k not in data:
                    data[k] = added_data
                else:
                    data[k] = np.concatenate((data[k], added_data), axis=0)
        return data

    @staticmethod
    def max_t_filled(filled):
        return th.sum(filled, 1).max(0)[0]
    
    def can_sample(self, batch_size):
        return self.buffer_size >= batch_size

    def sample(self, batch_size):
        sampled_ep_idx = np.random.choice(self.buffer_size, batch_size, replace=False)
        
        sampled_data = {k: th.tensor(v[sampled_ep_idx]) for k, v in self.data.items()}
        if self.args.use_corrected_terminated and "corrected_terminated" in sampled_data:
            sampled_data["terminated"] = sampled_data["corrected_terminated"]
        else:
            # 末行是 terminal transition 的数据缺少 bootstrap 帧；补零占位即可，
            # terminated 会关闭该帧的 bootstrap，最后一次奖励仍参与训练。
            lengths = sampled_data["filled"].sum(dim=1).reshape(-1).long()
            rows = th.arange(batch_size)
            needs_frame = sampled_data["terminated"][rows, lengths - 1, 0].bool()
            if needs_frame.any():
                if (lengths[needs_frame] == sampled_data["filled"].shape[1]).any():
                    sampled_data = {k: th.cat((v, th.zeros_like(v[:, :1])), dim=1)
                                    for k, v in sampled_data.items()}
                for value in sampled_data.values():
                    value[rows[needs_frame], lengths[needs_frame]] = 0
                sampled_data["filled"][rows[needs_frame], lengths[needs_frame], 0] = 1
                # 占位帧提供合法 no-op，避免策略/目标网络在全非法动作上归一化。
                sampled_data["avail_actions"][rows[needs_frame], lengths[needs_frame], :, 0] = 1
        """sampled_data = {}
        for k, v in self.data.items():
            dtype = self.scheme[k].get("dtype", th.float32) if self.scheme is not None and k in self.scheme else th.float32
            sampled_data[k] = th.tensor(v[sampled_ep_idx], dtype=dtype)"""
            
        max_ep_t = self.max_t_filled(filled=sampled_data['filled']).item()
        offline_data_batch = OfflineDataBatch(data=sampled_data, 
                                              batch_size=batch_size, 
                                              max_seq_length=max_ep_t, 
                                              device=self.device)
        return offline_data_batch


class OfflineBuffer():
    def __init__(self, args, map_name, quality,
                data_path='', # 最深层目录
                max_buffer_size=2000,
                device='cpu',
                shuffle=True) -> None:

        if args.offline_data_type=="h5":
            self.buffer = OfflineBufferH5(args, map_name, quality,
                                          data_path, max_buffer_size,
                                          device, shuffle)
            self.buffer_size = self.buffer.buffer_size
        else:
            raise NotImplementedError("Do not support offline data type: {}".format(args.offline_data_type))
    
    def can_sample(self, batch_size):
        return self.buffer.can_sample(batch_size)

    def sample(self, batch_size):
        return self.buffer.sample(batch_size)

    def sequential_iter(self, batch_size):
        return self.buffer.sequential_iter(batch_size)

    def reset_sequential_iter(self):
        self.buffer.reset_sequential_iter()

class DataSaver():
    def __init__(self, save_path, logger=None, max_size=2000) -> None:
        self.save_path = save_path
        self.max_size = max_size
        #self.episode_batch = []
        self.data_batch = []
        self.cur_size = 0
        self.part_cnt = 0
        self.logger = logger
        os.makedirs(save_path, exist_ok=True)
    
    def append(self, data):
        self.data_batch.append(data) # 数据 \OfflineDataBatch/EpisodeBatch
        self.cur_size += data[list(data.keys())[0]].shape[0]
        #if len(self.episode_batch) >= self.max_size:
        if self.cur_size >= self.max_size:
            self.save_batch()
    
    def save_batch(self):
        #if len(self.data_batch) == 0:
        if self.cur_size == 0:
            return
        keys = list(self.data_batch[0].keys())
        data_dict = {k: [] for k in keys}
        for data in self.data_batch:
            for k in keys:
                if isinstance(data[k], th.Tensor):
                    data_dict[k].append(data[k].numpy())
                else:
                    data_dict[k].append(data[k])
                    
        # 拼接示例：[(x, T, n_agents, *shape), ...] -> [max_size, T, n_agents, *shape]
        data_dict = {k: np.concatenate(v) for k, v in data_dict.items()}
        save_file = os.path.join(self.save_path, "part_{}.h5".format(self.part_cnt))
        with h5py.File(save_file, 'w') as file:
            for k, v in data_dict.items():
                file.create_dataset(k, data=v, compression='gzip', compression_opts=9)
        if self.logger is not None:
            self.logger.console_logger.info("Save offline buffer to {} with {} episodes".format(save_file, self.cur_size))
        else:
            print("Save offline buffer to {} with {} episodes".format(save_file, self.cur_size))
        self.data_batch.clear()
        self.cur_size = 0
        self.part_cnt += 1
    
    def close(self):
        self.save_batch()
