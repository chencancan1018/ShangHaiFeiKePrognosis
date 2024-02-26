
import sys
from os.path import abspath, dirname
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import zoom
from torch.cuda.amp import autocast
import albumentations
from albumentations.pytorch import ToTensorV2


class ClsConfig:

    def __init__(self,  network_f):
        # TODO: 模型配置文件
        self.network_f = network_f
        if self.network_f is not None:
            from mmcv import Config

            if isinstance(self.network_f, str):
                self.network_cfg = Config.fromfile(self.network_f)
            else:
                import tempfile

                with tempfile.TemporaryDirectory() as temp_config_dir:
                    with tempfile.NamedTemporaryFile(dir=temp_config_dir, suffix='.py') as temp_config_file:
                        with open(temp_config_file.name, 'wb') as f:
                            f.write(self.network_f.read())

                        self.network_cfg = Config.fromfile(temp_config_file.name)

    def __repr__(self) -> str:
        return str(self.__dict__)


class ClsModel:

    def __init__(self, model_f, network_f):
        # TODO: 模型文件定制
        self.model_f = model_f
        self.network_f = network_f


class ClsPredictor:

    def __init__(self, gpu: int, model: ClsModel):
        self.gpu = gpu
        self.model = model
        self.config = ClsConfig(self.model.network_f)
        self.load_model()

    def load_model(self):
        self.net = self._load_model(self.model.model_f, self.config.network_cfg, half=False)

    def _load_model(self, model_f, network_f, half=False) -> None:
        if isinstance(model_f, str):
            # 根据后缀判断类型
            if model_f.endswith(".pth"):
                net = self.load_model_pth(model_f, network_f, half)
            else:
                net = self.load_model_jit(model_f, half)
        else:
            # 根据模型文件前两个字节判断文件类型(hexdump filename | head -n 10)
            model_f.seek(0)
            headers = model_f.peek(2)
            if headers[0] == 0x80 and headers[1] == 0x02:
                # pth文件类型
                net = self.load_model_pth(model_f, network_f, half)
            else:
                # pt文件类型
                net = self.load_model_jit(model_f, half)
        return net

    def load_model_jit(self, model_f, half) -> None:
        # 加载静态图
        from torch import jit

        if not isinstance(model_f, str):
            model_f.seek(0)
        net = jit.load(model_f, map_location=f"cpu")
        net = net.eval()
        if half:
            net.half()
        net.cuda(self.gpu)
        net = net.forward_test
        return net
    
    def load_model_pth(self, model_f, network_cfg, half) -> None:
        # 加载动态图
        from starship.umtf.common import build_network

        sys.path.append(dirname(dirname(abspath(__file__))))
        from train import custom  # noqa: F401

        config = network_cfg

        net = build_network(config.model, test_cfg=config.test_cfg)

        if not isinstance(model_f, str):
            model_f.seek(0)
        checkpoint = torch.load(model_f, map_location=f"cpu")
        net.load_state_dict(checkpoint["state_dict"], strict=False)
        net.eval()
        if half:
            net.half()
        net.cuda(self.gpu)
        net = net.forward_test
        return net

    def _get_input(self, vol, seg):

        config = self.config.network_cfg

        def window_array(vol, win_level, win_width):
            win = [
                win_level - win_width / 2,
                win_level + win_width / 2,
            ]
            vol = np.clip(vol, win[0], win[1])
            vol -= win[0]
            vol /= win_width
            return vol

        # 加窗
        vol = [window_array(vol, wl, wd)[None] for wl, wd in zip(config.win_level, config.win_width)]
        vol = np.concatenate(vol, axis=0) # channel first
        seg = seg[None] # channel first
        vol_shape = np.array(vol.shape[1:])
        seg_shape = np.array(seg.shape[1:])

        # resize
        patch_size = np.array(config.patch_size)
        if np.any(vol_shape != patch_size):
            scale = np.array(np.array([vol.shape[0]] + list(patch_size)) / np.array(vol.shape))
            vol = zoom(vol, scale, order=1)
        
        if np.any(patch_size != seg_shape):
            scale = np.array(np.array([seg.shape[0]] + list(patch_size)) / np.array(seg.shape))
            seg = zoom(seg, scale, order=0)

        assert (np.array(vol.shape[1:]) == np.array(seg.shape[1:])).all()

        # numpy2tensor
        vol = torch.from_numpy(vol).float()
        seg = torch.from_numpy(seg).float()
        vol = torch.cat([vol, seg], dim=0)[None]
        return vol

    def forward(self, vol, spacing=None):
        lung_mask = vol[1]
        vol = vol[0]

        data = self._get_input(vol, lung_mask)
        with autocast():
            data = data.cuda(self.gpu).detach()
            pred = self.net(data)
            del data
            prob = F.softmax(pred, dim=1) # prob size: [1, 3]
            pred = torch.argmax(prob, dim=1, keepdim=False) # pred size: [1]

        prob = prob.cpu().detach().numpy() # prob size: [1, 3]
        pred = pred.cpu().detach().numpy().astype(np.int8) # pred size: [1]
        return prob, pred

class ClsPredictor25D(ClsPredictor):
    
    def _get_input(self, vol):
        config = self.config.network_cfg
        data = []
        for slice in vol:
            slice = ToTensorV2()(image=slice)['image']
            # normalize 
            slice = slice / 255 # to [0,1]
            slice = (slice - 0.5) / 0.5 # to [-1, 1]
            data.append(slice[None])
        data = torch.cat(data, dim=0)
        assert data.shape[0] == config.patch_size[0]
        assert data.shape[1] == 3
        assert data.shape[2] == config.patch_size[1]
        assert data.shape[3] == config.patch_size[2]
        return data[None]

    def forward(self, vol, spacing=None):
        data = self._get_input(vol)
        with autocast():
            data = data.cuda(self.gpu).detach()
            out0, out1 = self.net(data)
            del data
            # out0 size: [32, 3]
            # out1 size: [1, 3]
            prob0 = F.softmax(out0, dim=1) # prob0 size: [32, 3]
            pred0 = torch.argmax(prob0, dim=1, keepdim=False) # pred0 size: [32]
            prob1 = F.softmax(out1, dim=1) # prob0 size: [1, 3]
            pred1 = torch.argmax(prob1, dim=1, keepdim=False) # pred0 size: [1]
        
        prob0 = prob0.cpu().detach().numpy()
        prob1 = prob1.cpu().detach().numpy()

        pred0 = pred0.cpu().detach().numpy().astype(np.int8)      
        pred1 = pred1.cpu().detach().numpy().astype(np.int8) 

        probs = [prob0, prob1]
        preds = [pred0, pred1]
        return probs, preds


class ClsPredictor2D(ClsPredictor):
    
    def _get_input(self, vol):
        config = self.config.network_cfg
        data = []
        for slice in vol:
            slice = ToTensorV2()(image=slice)['image']
            # normalize 
            slice = slice / 255 # to [0,1]
            slice = (slice - 0.5) / 0.5 # to [-1, 1]
            data.append(slice[None])
        data = torch.cat(data, dim=0)
        assert data.shape[0] == config.batch_size
        assert data.shape[1] == 3
        assert data.shape[2] == config.patch_size[0]
        assert data.shape[3] == config.patch_size[1]
        return data

    def forward(self, vol, spacing=None):
        data = self._get_input(vol)
        with autocast():
            data = data.cuda(self.gpu).detach()
            out= self.net(data)
            del data
            # out0 size: [b, 3]
            prob = F.softmax(out, dim=1) # prob0 size: [b, 3]
            pred = torch.argmax(prob, dim=1, keepdim=False) # pred0 size: [b]

        prob = prob.cpu().detach().numpy()
        pred = pred.cpu().detach().numpy().astype(np.int8)      
        return prob, pred
    
class ClsPredictor2DSigHead(ClsPredictor):
    
    def _get_input(self, vol):
        config = self.config.network_cfg
        data = []
        for slice in vol:
            slice = ToTensorV2()(image=slice)['image']
            # normalize 
            slice = slice / 255 # to [0,1]
            slice = (slice - 0.5) / 0.5 # to [-1, 1]
            data.append(slice[None])
        data = torch.cat(data, dim=0)
        # assert data.shape[0] == config.batch_size
        assert data.shape[1] == 3
        assert data.shape[2] == config.patch_size[0]
        assert data.shape[3] == config.patch_size[1]
        return data

    def forward(self, vol, spacing=None):
        data = self._get_input(vol)
        with autocast():
            data = data.cuda(self.gpu).detach()
            out= self.net(data)
            del data
            # out size: [b, 1]
            prob = F.sigmoid(out).squeeze() # prob size: [b]
        prob = prob.cpu().detach().numpy()  
        return prob
