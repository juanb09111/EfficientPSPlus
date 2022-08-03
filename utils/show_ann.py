import torch
import numpy as np
import cv2

BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White


def visualize_bboxes(img, bboxes, color=BOX_COLOR, thickness=2, **kwargs):
    # x_min, y_min, w, h = bbox
    # x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    return img

def visualize_titles(img, bbox, titles, color=BOX_COLOR, thickness=2, font_thickness = 2, font_scale=0.35, **kwargs):
    # x_min, y_min, w, h = bbox
    # x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
    for title in titles:
        x_min, y_min, x_max, y_max = bbox
        x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)
        ((text_width, text_height), _) = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
        cv2.putText(img, title, (x_min, y_min - int(0.3 * text_height)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, TEXT_COLOR,
                    font_thickness, lineType=cv2.LINE_AA)
    return img

def visualize_masks(masks, color=BOX_COLOR, thickness=2):
    label_image = np.zeros([1, *masks.shape[-2:]])
    for mask in masks:
        mask = mask.int().cpu().numpy()
        label_image += (mask > 0)
    label_image = cv2.cvtColor(np.float32(label_image[0])*255, cv2.COLOR_GRAY2RGB)
    alpha = 0
    beta = (1.0 - alpha)
    return cv2.addWeighted(label_image, alpha, label_image, beta, 0.0)
    