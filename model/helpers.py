import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torchvision.transforms.functional import resize, to_pil_image

import utils.functions as fn


class SamResize:
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        h, w, _ = image.shape
        long_side = max(h, w)
        if long_side != self.size:
            return self.apply_image(image)
        else:
            return image

    def apply_image(self, image):
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.size)
        return np.array(resize(to_pil_image(image), target_size))

    @staticmethod
    def get_preprocess_shape(oldh, oldw, long_side_length):
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(size={self.size})"


class SamPad:
    def __init__(self, size, fill=0, pad_mode="corner"):
        self.size = size
        self.fill = fill
        self.pad_mode = pad_mode

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        h, w = image.shape[-2:]
        th, tw = self.size, self.size
        assert th >= h and tw >= w
        if self.pad_mode == "corner":
            image = F.pad(image, (0, tw - w, 0, th - h), value=self.fill)
        else:
            raise NotImplementedError
        return image

    def __repr__(self) -> str:
        return f"{type(self).__name__}(size={self.size},mode={self.pad_mode},fill={self.fill})"


class OpSequential(nn.Module):
    def __init__(self, op_list: list[nn.Module or None]):
        super(OpSequential, self).__init__()
        valid_op_list = []
        for op in op_list:
            if op is not None:
                valid_op_list.append(op)
        self.op_list = nn.ModuleList(valid_op_list)

    def forward(self, x):
        for op in self.op_list:
            x = op(x)
        return x


class Conv(nn.Module):
    def __init__(self, inp, oup, k=3, s=1, d=1, gr=1, use_bias=False, drop=0, norm="bn2d", act="relu"):
        super(Conv, self).__init__()
        padding = fn.get_same_padding(k)
        padding *= d

        self.drop = nn.Dropout2d(drop, inplace=False) if drop > 0 else None
        self.conv = nn.Conv2d(inp, oup, k, s, padding=padding, dilation=d, groups=gr, bias=use_bias)
        self.norm = fn.build_norm(norm, num_features=oup)
        self.act = fn.build_act(act)

    def forward(self, x):
        if self.drop is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class DSConv(nn.Module):
    def __init__(self, inp, oup, k=3, s=1, use_bias=False, norm=("bn2d", "bn2d"), act=("relu6", None)):
        super(DSConv, self).__init__()

        use_bias = fn.val2tuple(use_bias, 2)
        norm = fn.val2tuple(norm, 2)
        act = fn.val2tuple(act, 2)

        self.depth_conv = Conv(inp, oup, k, s, gr=inp, norm=norm[0], act=act[0], use_bias=use_bias[0])
        self.point_conv = Conv(inp, oup, 1, norm=norm[1], act=act[1], use_bias=use_bias[1])

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class MBConv(nn.Module):
    def __init__(self, inp, oup, k=3, s=1, mid_ch=None, exp_ratio=6, use_bias=False,
                 norm=("bn2d", "bn2d", "bn2d"), act=("relu6", "relu6", None)):
        super(MBConv, self).__init__()

        use_bias = fn.val2tuple(use_bias, 3)
        norm = fn.val2tuple(norm, 3)
        act = fn.val2tuple(act, 3)
        mid_ch = mid_ch or round(inp * exp_ratio)

        self.inverted_conv = Conv(inp, mid_ch, 1, s=1, norm=norm[0], act=act[0], use_bias=use_bias[0])
        self.depth_conv = Conv(mid_ch, mid_ch, k, s=s, gr=mid_ch, norm=norm[1], act=act[1], use_bias=use_bias[1])
        self.point_conv = Conv(mid_ch, oup, 1, norm=norm[2], act=act[2], use_bias=use_bias[2])

    def forward(self, x):
        x = self.inverted_conv(x)
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class FusedMBConv(nn.Module):
    def __init__(self, inp, oup, k=3, s=1, mid_ch=None, exp_ratio=6, gr=1, use_bias=False, norm=("bn2d", "bn2d"),
                 act=("relu6", None)):
        super().__init__()
        use_bias = fn.val2tuple(use_bias, 2)
        norm = fn.val2tuple(norm, 2)
        act = fn.val2tuple(act, 2)

        mid_ch = mid_ch or round(inp * exp_ratio)

        self.spatial_conv = Conv(inp, mid_ch, k, s, gr=gr, use_bias=use_bias[0], norm=norm[0], act=act[0])
        self.point_conv = Conv(mid_ch, oup, 1, use_bias=use_bias[1], norm=norm[1], act=act[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spatial_conv(x)
        x = self.point_conv(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, main, shortcut, post_act=None, pre_norm: nn.Module or None = None):
        super(ResidualBlock, self).__init__()

        self.pre_norm = pre_norm
        self.main = main
        self.shortcut = shortcut
        self.post_act = fn.build_act(post_act)

    def forward_main(self, x):
        if self.pre_norm is None:
            return self.main(x)
        else:
            return self.main(self.pre_norm(x))

    def forward(self, x):
        if self.main is None:
            res = x
        elif self.shortcut is None:
            res = self.forward_main(x)
        else:
            res = self.forward_main(x) + self.shortcut(x)
            if self.post_act:
                res = self.post_act(res)
        return res


class IdentityLayer(nn.Module):
    def forward(self, x):
        return x


class ResBlock(nn.Module):
    def __init__(self, inp, oup, k=3, s=1, mid_ch=None, exp_ratio=1, use_bias=False, norm=("bn2d", "bn2d"),
                 act=("relu6", None)):
        super().__init__()
        use_bias = fn.val2tuple(use_bias, 2)
        norm = fn.val2tuple(norm, 2)
        act = fn.val2tuple(act, 2)

        mid_ch = mid_ch or round(inp * exp_ratio)

        self.conv1 = Conv(inp, mid_ch, k, s, use_bias=use_bias[0], norm=norm[0], act=act[0])
        self.conv2 = Conv(mid_ch, oup, k, 1, use_bias=use_bias[1], norm=norm[1], act=act[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class LiteMLA(nn.Module):
    def __init__(self, inp, oup, heads=None, heads_ratio=1.0, dim=8, use_bias=False, norm=(None, "bn2d"),
                 act=(None, None), kernel_func="relu", scales=(5,), eps=1.0e-15):
        super(LiteMLA, self).__init__()
        self.eps = eps
        heads = heads or int(inp // dim * heads_ratio)

        total_dim = heads * dim

        use_bias = fn.val2tuple(use_bias, 2)
        norm = fn.val2tuple(norm, 2)
        act = fn.val2tuple(act, 2)

        self.dim = dim
        self.qkv = Conv(inp, 3 * total_dim, 1, use_bias=use_bias[0], norm=norm[0], act=act[0])
        self.aggreg = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(3 * total_dim, 3 * total_dim, scale, padding=fn.get_same_padding(scale),
                              groups=3 * total_dim, bias=use_bias[0]),
                    nn.Conv2d(3 * total_dim, 3 * total_dim, 1, groups=3 * heads, bias=use_bias[0]),
                )

                for scale in scales

            ]
        )
        self.kernel_func = fn.build_act(kernel_func, inplace=False)

        self.proj = Conv(total_dim * (1 + len(scales)), oup, 1, use_bias=use_bias[1], norm=norm[1], act=act[1])

    @autocast(enabled=False)
    def relu_linear_att(self, qkv):
        B, _, H, W = list(qkv.size())

        if qkv.dtype == torch.float16:
            qkv = qkv.float()

        qkv = torch.reshape(qkv, (B, -1, 3 * self.dim, H * W,), )
        qkv = torch.transpose(qkv, -1, -2)
        q, k, v = (qkv[..., 0: self.dim], qkv[..., self.dim: 2 * self.dim], qkv[..., 2 * self.dim:],)

        # lightweight linear attention
        q = self.kernel_func(q)
        k = self.kernel_func(k)

        # linear matmul
        trans_k = k.transpose(-1, -2)

        v = F.pad(v, (0, 1), mode="constant", value=1)
        kv = torch.matmul(trans_k, v)
        out = torch.matmul(q, kv)
        out = out[..., :-1] / (out[..., -1:] + self.eps)

        out = torch.transpose(out, -1, -2)
        out = torch.reshape(out, (B, -1, H, W))
        return out

    def forward(self, x):
        # generate multi-scale q, k, v
        qkv = self.qkv(x)
        multi_scale_qkv = [qkv]
        for op in self.aggreg:
            multi_scale_qkv.append(op(qkv))
        multi_scale_qkv = torch.cat(multi_scale_qkv, dim=1)

        out = self.relu_linear_att(multi_scale_qkv)
        out = self.proj(out)

        return out


class DAGBlock(nn.Module):
    def __init__(self, inputs, merge, post_input, middle, outputs):
        super(DAGBlock, self).__init__()

        self.input_keys = list(inputs.keys())
        self.input_ops = nn.ModuleList(list(inputs.values()))
        self.merge = merge
        self.post_input = post_input

        self.middle = middle

        self.output_keys = list(outputs.keys())
        self.output_ops = nn.ModuleList(list(outputs.values()))

    def forward(self, feature_dict):
        feat = [op(feature_dict[key]) for key, op in zip(self.input_keys, self.input_ops)]
        if self.merge == "add":
            feat = fn.list_sum(feat)
        elif self.merge == "cat":
            feat = torch.concat(feat, dim=1)
        else:
            raise NotImplementedError
        if self.post_input is not None:
            feat = self.post_input(feat)
        feat = self.middle(feat)
        for key, op in zip(self.output_keys, self.output_ops):
            feature_dict[key] = op(feat)
        return feature_dict


class SamNeck(DAGBlock):
    def __init__(self, fid_list, inp_list, head_width, head_depth, exp_ratio, mid_op, out_dim=256, norm="bn2d",
                 act="gelu"):
        inputs = {}
        for fid, in_channel in zip(fid_list, inp_list):
            inputs[fid] = OpSequential(
                [Conv(in_channel, head_width, 1, norm=norm, act=None), fn.UpSampleLayer(size=(64, 64)), ])

        middle = []
        for _ in range(head_depth):
            if mid_op == "mb":
                block = MBConv(head_width, head_width, exp_ratio=exp_ratio, norm=norm, act=(act, act, None))
            elif mid_op == "fmb":
                block = FusedMBConv(head_width, head_width, exp_ratio=exp_ratio, norm=norm, act=(act, None))
            elif mid_op == "res":
                block = ResBlock(head_width, head_width, exp_ratio=exp_ratio, norm=norm, act=(act, None))
            else:
                raise NotImplementedError
            middle.append(ResidualBlock(block, IdentityLayer()))
        middle = OpSequential(middle)

        outputs = {"sam_encoder": OpSequential([Conv(head_width, out_dim, 1, use_bias=True, norm=None, act=None, ), ])}

        super(SamNeck, self).__init__(inputs, "add", None, middle=middle, outputs=outputs)


class Encoder(nn.Module):
    def __init__(self, backbone, neck):
        super().__init__()
        self.backbone = backbone
        self.neck = neck

        self.norm = fn.build_norm("ln2d", 256)

    def forward(self, x):
        feed_dict = self.backbone(x)
        feed_dict = self.neck(feed_dict)

        output = feed_dict["sam_encoder"]
        output = self.norm(output)
        return output


class EfficientViTBlock(nn.Module):
    def __init__(self, inp, heads_ratio=1.0, dim=32, exp_ratio=4, scales=(5,), norm="bn2d", act="hswish"):
        super(EfficientViTBlock, self).__init__()
        self.context_module = ResidualBlock(
            LiteMLA(inp=inp, oup=inp, heads_ratio=heads_ratio, dim=dim, norm=(None, norm), scales=scales, ),
            IdentityLayer(), )
        local_module = MBConv(inp=inp, oup=inp, exp_ratio=exp_ratio, use_bias=(True, True, False),
                              norm=(None, None, norm), act=(act, act, None), )
        self.local_module = ResidualBlock(local_module, IdentityLayer())

    def forward(self, x):
        x = self.context_module(x)
        x = self.local_module(x)
        return x
