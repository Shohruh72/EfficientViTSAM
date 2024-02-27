import argparse
import time

import cv2
import matplotlib.pyplot as plt
import yaml
from PIL import Image
from matplotlib.patches import Rectangle

from model.net import *


def draw_scatter(image, points, color="g", marker="*", s=10, ew=0.25, tmp_name=".tmp.png"):
    dpi = 300
    oh, ow, _ = image.shape
    plt.close()
    plt.figure(1, figsize=(oh / dpi, ow / dpi))
    plt.imshow(image)
    if isinstance(color, str):
        color = [color for _ in points]
    for (x, y), c in zip(points, color):
        plt.scatter(x, y, color=c, marker=marker, s=s, edgecolors="white", linewidths=ew)
    plt.axis("off")
    plt.savefig(tmp_name, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.0)
    img = cv2.imread(tmp_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = cv2.resize(img, dsize=(ow, oh))
    os.remove(tmp_name)
    plt.close()
    return image


def draw_binary_mask(raw_image, binary_mask, mask_color=(0, 0, 255)):
    color_mask = np.zeros_like(raw_image, dtype=np.uint8)
    color_mask[binary_mask == 1] = mask_color
    mix = color_mask * 0.5 + raw_image * (1 - 0.5)
    binary_mask = np.expand_dims(binary_mask, axis=2)
    canvas = binary_mask * mix + (1 - binary_mask) * raw_image
    canvas = np.asarray(canvas, dtype=np.uint8)
    return canvas


def cat_images(image_list, axis=1, pad=20):
    shape_list = [image.shape for image in image_list]
    max_h = max([shape[0] for shape in shape_list]) + pad * 2
    max_w = max([shape[1] for shape in shape_list]) + pad * 2

    for i, image in enumerate(image_list):
        canvas = np.zeros((max_h, max_w, 3), dtype=np.uint8)
        h, w, _ = image.shape
        crop_y = (max_h - h) // 2
        crop_x = (max_w - w) // 2
        canvas[crop_y: crop_y + h, crop_x: crop_x + w] = image
        image_list[i] = canvas

    image = np.concatenate(image_list, axis=axis)
    return image


def draw_bbox(image, bbox, color="g", linewidth=1, tmp_name=".tmp.png", ):
    dpi = 300
    oh, ow, _ = image.shape
    plt.close()
    plt.figure(1, figsize=(oh / dpi, ow / dpi))
    plt.imshow(image)
    if isinstance(color, str):
        color = [color for _ in bbox]
    for (x0, y0, x1, y1), c in zip(bbox, color):
        plt.gca().add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, lw=linewidth, edgecolor=c, facecolor=(0, 0, 0, 0)))
    plt.axis("off")
    plt.savefig(tmp_name, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.0)
    img = cv2.imread(tmp_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = cv2.resize(img, dsize=(ow, oh))
    os.remove(tmp_name)
    plt.close()
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="l0")
    parser.add_argument("--weight_url", type=str, default='./outputs/weights/l0.pt')
    parser.add_argument("--model_dir", type=str, default='./outputs/weights/')
    parser.add_argument("--multimask", action="store_true")
    parser.add_argument("--image_path", type=str, default="outputs/img.jpg")
    parser.add_argument("--output_path", type=str, default="outputs/sam_demo.png")

    parser.add_argument("--mode", type=str, default="point", choices=["point", "box"])
    parser.add_argument("--point", type=str, default=None)
    parser.add_argument("--box", type=str, default='[150,70,640,400]')

    # EfficientViTSamAutomaticMaskGenerator args
    parser.add_argument("--pred_iou_thresh", type=float, default=0.8)
    parser.add_argument("--stability_score_thresh", type=float, default=0.85)
    parser.add_argument("--min_mask_region_area", type=float, default=100)

    args, opt = parser.parse_known_args()

    vit_sam = create_sam_model(args.model, True, args.weight_url).cuda().eval()
    vit_sam_pred = ViTSamPredictor(vit_sam)

    raw_image = np.array(Image.open(args.image_path).convert("RGB"))

    tmp_file = f".tmp_{time.time()}.png"
    shape = raw_image.shape[:2]

    if args.mode == "point":
        args.point = yaml.safe_load(args.point or f"[[{shape[1] // 2},{shape[0] // 2},{1}]]")
        point_coords = [(x, y) for x, y, _ in args.point]
        point_labels = [l for _, _, l in args.point]

        vit_sam_pred.set_image(raw_image)
        masks, _, _ = vit_sam_pred.predict(point_coords=np.array(point_coords), point_labels=np.array(point_labels),
                                           multimask_output=args.multimask, )
        plots = [
            draw_scatter(draw_binary_mask(raw_image, binary_mask, (0, 0, 255)), point_coords,
                         color=["g" if l == 1 else "r" for l in point_labels], s=10, ew=0.25, tmp_name=tmp_file, )
            for binary_mask in masks
        ]
        plots = cat_images(plots, axis=1)
        Image.fromarray(plots).save(args.output_path)
    elif args.mode == "box":
        args.box = yaml.safe_load(args.box)
        vit_sam_pred.set_image(raw_image)
        masks, _, _ = vit_sam_pred.predict(point_coords=None, point_labels=None, box=np.array(args.box),
                                           multimask_output=args.multimask, )
        plots = [
            draw_bbox(
                draw_binary_mask(raw_image, binary_mask, (0, 0, 255)),
                [args.box],
                color="g",
                tmp_name=tmp_file,
            )
            for binary_mask in masks
        ]
        plots = cat_images(plots, axis=1)
        Image.fromarray(plots).save(args.output_path)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
