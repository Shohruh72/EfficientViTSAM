from model.helpers import *
import utils.functions as fn


class EfficientViTBackbone(nn.Module):
    def __init__(self, width, depth, inp=3, dim=32, exp_ratio=4, norm="bn2d", act="hswish"):
        super().__init__()
        self.width = []
        self.input_stem = [Conv(inp=3, oup=width[0], s=2, norm=norm, act=act)]
        for _ in range(depth[0]):
            block = self.build_local_block(inp=width[0], oup=width[0], s=1, exp_ratio=1, norm=norm, act=act)
            self.input_stem.append(ResidualBlock(block, IdentityLayer()))
        inp = width[0]
        self.input_stem = OpSequential(self.input_stem)
        self.width.append(inp)

        # stages
        self.stages = []
        for w, d in zip(width[1:3], depth[1:3]):
            stage = []
            for i in range(d):
                s = 2 if i == 0 else 1
                block = self.build_local_block(inp=inp, oup=w, s=s, exp_ratio=exp_ratio, norm=norm, act=act)
                block = ResidualBlock(block, IdentityLayer() if s == 1 else None)
                stage.append(block)
                inp = w
            self.stages.append(OpSequential(stage))
            self.width.append(inp)

        for w, d in zip(width[3:], depth[3:]):
            stage = []
            block = self.build_local_block(inp=inp, oup=w, s=2, exp_ratio=exp_ratio, norm=norm, act=act,
                                           fewer_norm=True)
            stage.append(ResidualBlock(block, None))
            inp = w

            for _ in range(d):
                stage.append(EfficientViTBlock(inp=inp, dim=dim, exp_ratio=exp_ratio, norm=norm, act=act))
            self.stages.append(OpSequential(stage))
            self.width.append(inp)
        self.stages = nn.ModuleList(self.stages)

    @staticmethod
    def build_local_block(inp, oup, s, exp_ratio, norm, act, fewer_norm=False, ):
        if exp_ratio == 1:
            block = DSConv(inp, oup, s=s, use_bias=(True, False) if fewer_norm else False,
                           norm=(None, norm) if fewer_norm else norm, act=(act, None))
        else:
            block = MBConv(inp, oup, s=s, exp_ratio=exp_ratio, use_bias=(True, True, False) if fewer_norm else False,
                           norm=(None, None, norm) if fewer_norm else norm, act=(act, act, None), )
        return block

    def forward(self, x):
        output_dict = {"input": x}
        output_dict["stage0"] = x = self.input_stem(x)
        for stage_id, stage in enumerate(self.stages, 1):
            output_dict["stage%d" % stage_id] = x = stage(x)
        output_dict["stage_final"] = x
        return output_dict


class EfficientViTLargeBackbone(nn.Module):
    def __init__(self, width, depth, block_list=None, expand=None, few_norm=None, inp=3, qkv_dim=32, norm="bn2d",
                 act="gelu"):
        super().__init__()
        block_list = block_list or ["res", "fmb", "fmb", "mb", "att"]
        expand = expand or [1, 4, 4, 4, 6]
        few_norm = few_norm or [False, False, False, True, True]

        self.width_list = []
        self.stages = []
        # stage 0
        stage0 = [Conv(inp=3, oup=width[0], s=2, norm=norm, act=act)]
        for _ in range(depth[0]):
            block = self.build_local_block(block=block_list[0], inp=width[0], oup=width[0], s=1,
                                           exp_ratio=expand[0], norm=norm, act=act, few_norm=few_norm[0])
            stage0.append(ResidualBlock(block, IdentityLayer()))
        inp = width[0]
        self.stages.append(OpSequential(stage0))
        self.width_list.append(inp)

        for stage_id, (w, d) in enumerate(zip(width[1:], depth[1:]), start=1):
            stage = []
            block = self.build_local_block(
                block="mb" if block_list[stage_id] not in ["mb", "fmb"] else block_list[stage_id],
                inp=inp, oup=w, s=2, exp_ratio=expand[stage_id] * 4, norm=norm, act=act,
                few_norm=few_norm[stage_id])
            stage.append(ResidualBlock(block, None))
            inp = w

            for _ in range(d):
                if block_list[stage_id].startswith("att"):
                    stage.append(EfficientViTBlock(inp=inp, dim=qkv_dim, exp_ratio=expand[stage_id],
                                                   scales=(3,) if block_list[stage_id] == "att@3" else (5,), norm=norm,
                                                   act=act, ))
                else:
                    block = self.build_local_block(block=block_list[stage_id], inp=inp, oup=inp, s=1,
                                                   exp_ratio=expand[stage_id], norm=norm, act=act,
                                                   few_norm=few_norm[stage_id], )
                    block = ResidualBlock(block, IdentityLayer())
                    stage.append(block)
            self.stages.append(OpSequential(stage))
            self.width_list.append(inp)
        self.stages = nn.ModuleList(self.stages)

    @staticmethod
    def build_local_block(block, inp, oup, s, exp_ratio, norm, act, few_norm=False):
        if block == "res":
            block = ResBlock(inp=inp, oup=oup, s=s, use_bias=(True, False) if few_norm else False,
                             norm=(None, norm) if few_norm else norm, act=(act, None))
        elif block == "fmb":
            block = FusedMBConv(inp=inp, oup=oup, s=s, exp_ratio=exp_ratio,
                                use_bias=(True, False) if few_norm else False, norm=(None, norm) if few_norm else norm,
                                act=(act, None))
        elif block == "mb":
            block = MBConv(inp=inp, oup=oup, s=s, exp_ratio=exp_ratio,
                           use_bias=(True, True, False) if few_norm else False,
                           norm=(None, None, norm) if few_norm else norm, act=(act, act, None))
        else:
            raise ValueError(block)
        return block

    def forward(self, x):
        output_dict = {"input": x}
        for stage_id, stage in enumerate(self.stages):
            output_dict["stage%d" % stage_id] = x = stage(x)
        output_dict["stage_final"] = x
        return output_dict

