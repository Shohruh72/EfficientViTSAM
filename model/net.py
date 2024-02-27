import copy
import os

from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything import SamAutomaticMaskGenerator
from torch.nn.modules.batchnorm import _BatchNorm
from torchvision import transforms

from model.backbone import *


def vit_backbone_l0(**kwargs):
    backbone = EfficientViTLargeBackbone(width=[32, 64, 128, 256, 512], depth=[1, 1, 1, 4, 4],
                                         **fn.build_kwargs_from_config(kwargs, EfficientViTLargeBackbone), )
    return backbone


def vit_backbone_l1(**kwargs):
    backbone = EfficientViTLargeBackbone(width_list=[32, 64, 128, 256, 512], depth_list=[1, 1, 1, 6, 6],
                                         **fn.build_kwargs_from_config(kwargs, EfficientViTLargeBackbone), )
    return backbone


def vit_backbone_l2(**kwargs):
    backbone = EfficientViTLargeBackbone(width_list=[32, 64, 128, 256, 512], depth_list=[1, 2, 2, 8, 8],
                                         **fn.build_kwargs_from_config(kwargs, EfficientViTLargeBackbone), )
    return backbone


def vit_backbone_l3(**kwargs):
    backbone = EfficientViTLargeBackbone(width_list=[64, 128, 256, 512, 1024], depth_list=[1, 2, 2, 8, 8],
                                         **fn.build_kwargs_from_config(kwargs, EfficientViTLargeBackbone), )
    return backbone


def vit_sam_l0(image_size=512, **kwargs):
    backbone = vit_backbone_l0(**kwargs)

    neck = SamNeck(fid_list=["stage4", "stage3", "stage2"], inp_list=[512, 256, 128], head_width=256, head_depth=4,
                   exp_ratio=1, mid_op="fmb", )

    image_encoder = Encoder(backbone, neck)
    return build_efficientvit_sam(image_encoder, image_size)


def vit_sam_l1(image_size=512, **kwargs):
    backbone = vit_backbone_l1(**kwargs)

    neck = SamNeck(fid_list=["stage4", "stage3", "stage2"], inp_list=[512, 256, 128], head_width=256, head_depth=8,
                   exp_ratio=1, mid_op="fmb")

    image_encoder = Encoder(backbone, neck)
    return build_efficientvit_sam(image_encoder, image_size)


def vit_sam_l2(image_size=512, **kwargs):
    backbone = vit_backbone_l2(**kwargs)

    neck = SamNeck(fid_list=["stage4", "stage3", "stage2"], inp_list=[512, 256, 128], head_width=256,
                   head_depth=12, exp_ratio=1, mid_op="fmb")

    image_encoder = Encoder(backbone, neck)
    return build_efficientvit_sam(image_encoder, image_size)


def vit_sam_xl0(image_size=1024, **kwargs):
    backbone = EfficientViTLargeBackbone(width=[32, 64, 128, 256, 512, 1024], depth=[0, 1, 1, 2, 3, 3],
                                         block=["res", "fmb", "fmb", "fmb", "att@3", "att@3"],
                                         expand=[1, 4, 4, 4, 4, 6],
                                         few_norm=[False, False, False, False, True, True],
                                         **fn.build_kwargs_from_config(kwargs, EfficientViTLargeBackbone), )

    neck = SamNeck(fid_list=["stage5", "stage4", "stage3"], inp_list=[1024, 512, 256], head_width=256, head_depth=6,
                   exp_ratio=4, mid_op="fmb", )

    image_encoder = Encoder(backbone, neck)
    return build_efficientvit_sam(image_encoder, image_size)


def vit_sam_xl1(image_size, **kwargs):
    backbone = EfficientViTLargeBackbone(width=[32, 64, 128, 256, 512, 1024], depth=[1, 2, 2, 4, 6, 6],
                                         block=["res", "fmb", "fmb", "fmb", "att@3", "att@3"],
                                         expand=[1, 4, 4, 4, 4, 6], few_norm=[False, False, False, False, True, True],
                                         **fn.build_kwargs_from_config(kwargs, EfficientViTLargeBackbone), )

    neck = SamNeck(fid_list=["stage5", "stage4", "stage3"], inp_list=[1024, 512, 256], head_width=256, head_depth=12,
                   exp_ratio=4, mid_op="fmb", )

    image_encoder = Encoder(backbone, neck)
    return build_efficientvit_sam(image_encoder, image_size)


class EfficientViTSam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(self, image_encoder, prompt_encoder, mask_decoder, image_size=(1024, 512)):
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder

        self.image_size = image_size

        self.transform = transforms.Compose(
            [
                SamResize(self.image_size[1]),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[123.675 / 255, 116.28 / 255, 103.53 / 255],
                    std=[58.395 / 255, 57.12 / 255, 57.375 / 255],
                ),
                SamPad(self.image_size[1]),
            ]
        )

    def postprocess_masks(
            self,
            masks: torch.Tensor,
            input_size: tuple[int, ...],
            original_size: tuple[int, ...],
    ) -> torch.Tensor:
        masks = F.interpolate(
            masks,
            (self.image_size[0], self.image_size[0]),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks


def build_efficientvit_sam(image_encoder, image_size):
    return EfficientViTSam(
        image_encoder=image_encoder,
        prompt_encoder=PromptEncoder(
            embed_dim=256,
            image_embedding_size=(64, 64),
            input_image_size=(1024, 1024),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=256,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        image_size=(1024, image_size),
    )


def set_norm_eps(model, eps=None):
    for m in model.modules():
        if isinstance(m, (nn.GroupNorm, nn.LayerNorm, _BatchNorm)):
            if eps is not None:
                m.eps = eps


def load_state_dict_from_file(file, only_state_dict=True):
    file = os.path.realpath(os.path.expanduser(file))
    checkpoint = torch.load(file, map_location="cpu")
    if only_state_dict and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    return checkpoint


def create_sam_model(name, pretrained=True, weight_url=None, **kwargs):
    model_dict = {
        "l0": vit_sam_l0,
        "l1": vit_sam_l1,
        "l2": vit_sam_l2,
        "xl0": vit_sam_xl0,
        "xl1": vit_sam_xl1,
    }

    weight_list = {
        "l0": f"args.model_dir/l0.pt",
        "l1": f"args.model_dir/l1.pt",
        "l2": f"args.model_dir/l2.pt",
        "xl0": f"args.model_dir/xl0.pt",
        "xl1": f"args.model_dir/xl1.pt",
    }

    model_id = name.split("-")[0]
    if model_id not in model_dict:
        raise ValueError(f"Do not find {name} in the model zoo. List of models: {list(model_dict.keys())}")
    else:
        model = model_dict[model_id](**kwargs)
    set_norm_eps(model, 1e-6)

    if pretrained:
        weight_url = weight_url or weight_list.get(name, None)
        if weight_url is None:
            raise ValueError(f"Do not find the pretrained weight of {name}.")
        else:
            weight = load_state_dict_from_file(weight_url)
            model.load_state_dict(weight)
    return model


# ----------------------------------------- Predictor -------------------------------

def build_point_grid(n_per_side):
    offset = 1 / (2 * n_per_side)
    points_one_side = np.linspace(offset, 1 - offset, n_per_side)
    points_x = np.tile(points_one_side[None, :], (n_per_side, 1))
    points_y = np.tile(points_one_side[:, None], (1, n_per_side))
    points = np.stack([points_x, points_y], axis=-1).reshape(-1, 2)
    return points


def build_all_layer_point_grids(n_per_side, n_layers, scale_per_layer):
    points_by_layer = []
    for i in range(n_layers + 1):
        n_points = int(n_per_side / (scale_per_layer ** i))
        points_by_layer.append(build_point_grid(n_points))
    return points_by_layer


def get_device(model):
    return model.parameters().__next__().device


class ViTSamPredictor:
    def __init__(self, sam_model):
        self.model = sam_model
        self.reset_image()

    @property
    def transform(self):
        return self

    @property
    def device(self):
        return get_device(self.model)

    def reset_image(self):
        self.is_image_set = False
        self.features = None
        self.original_size = None
        self.input_size = None

    def apply_coords(self, coords, im_size=None):
        old_h, old_w = self.original_size
        new_h, new_w = self.input_size
        coords = copy.deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(self, boxes, im_size=None):
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2))
        return boxes.reshape(-1, 4)

    @torch.inference_mode()
    def set_image(self, image, image_format="RGB"):
        assert image_format in [
            "RGB",
            "BGR",
        ], f"image_format must be in ['RGB', 'BGR'], is {image_format}."
        if image_format != self.model.image_format:
            image = image[..., ::-1]

        self.reset_image()

        self.original_size = image.shape[:2]
        self.input_size = ResizeLongestSide.get_preprocess_shape(
            *self.original_size, long_side_length=self.model.image_size[0]
        )

        torch_data = self.model.transform(image).unsqueeze(dim=0).to(get_device(self.model))
        self.features = self.model.image_encoder(torch_data)
        self.is_image_set = True

    def predict(self, point_coords=None, point_labels=None, box=None, mask_input=None, multimask_output=True,
                return_logits=False):

        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        device = get_device(self.model)
        # Transform input prompts
        coords_torch, labels_torch, box_torch, mask_input_torch = None, None, None, None
        if point_coords is not None:
            assert point_labels is not None, "point_labels must be supplied if point_coords is supplied."
            point_coords = self.apply_coords(point_coords)
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=device)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=device)
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
        if box is not None:
            box = self.apply_boxes(box)
            box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
            box_torch = box_torch[None, :]
        if mask_input is not None:
            mask_input_torch = torch.as_tensor(mask_input, dtype=torch.float, device=device)
            mask_input_torch = mask_input_torch[None, :, :, :]

        masks, iou_predictions, low_res_masks = self.predict_torch(
            coords_torch,
            labels_torch,
            box_torch,
            mask_input_torch,
            multimask_output,
            return_logits=return_logits,
        )

        masks = masks[0].detach().cpu().numpy()
        iou_predictions = iou_predictions[0].detach().cpu().numpy()
        low_res_masks = low_res_masks[0].detach().cpu().numpy()
        return masks, iou_predictions, low_res_masks

    @torch.inference_mode()
    def predict_torch(self, point_coords=None, point_labels=None, boxes=None, mask_input=None, multimask_output=True,
                      return_logits=False, ):
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        if point_coords is not None:
            points = (point_coords, point_labels)
        else:
            points = None

        # Embed prompts
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=mask_input,
        )

        # Predict masks
        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )

        masks = self.model.postprocess_masks(low_res_masks, self.input_size, self.original_size)

        if not return_logits:
            masks = masks > self.model.mask_threshold

        return masks, iou_predictions, low_res_masks


class ViTSamMaskGen(SamAutomaticMaskGenerator):
    def __init__(self, model, points_per_side=32, points_per_batch=64, pred_iou_thresh=0.88,
                 stability_score_thresh=0.95, stability_score_offset=1.0, box_nms_thresh=0.7, crop_n_layers=0,
                 crop_nms_thresh=0.7, crop_overlap_ratio=512 / 1500, crop_n_points_downscale_factor=1, point_grids=None,
                 min_mask_region_area=0, output_mode="binary_mask"):

        assert (points_per_side is None) != (
                point_grids is None
        ), "Exactly one of points_per_side or point_grid must be provided."
        if points_per_side is not None:
            self.point_grids = build_all_layer_point_grids(
                points_per_side,
                crop_n_layers,
                crop_n_points_downscale_factor,
            )
        elif point_grids is not None:
            self.point_grids = point_grids
        else:
            raise ValueError("Can't have both points_per_side and point_grid be None.")

        assert output_mode in [
            "binary_mask",
            "uncompressed_rle",
            "coco_rle",
        ], f"Unknown output_mode {output_mode}."
        if output_mode == "coco_rle":
            from pycocotools import mask as mask_utils  # type: ignore # noqa: F401

        if min_mask_region_area > 0:
            import cv2  # type: ignore # noqa: F401

        self.predictor = ViTSamPredictor(model)
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.box_nms_thresh = box_nms_thresh
        self.crop_n_layers = crop_n_layers
        self.crop_nms_thresh = crop_nms_thresh
        self.crop_overlap_ratio = crop_overlap_ratio
        self.crop_n_points_downscale_factor = crop_n_points_downscale_factor
        self.min_mask_region_area = min_mask_region_area
        self.output_mode = output_mode
