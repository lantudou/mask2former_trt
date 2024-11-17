# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
from detectron2.utils.logger import setup_logger
setup_logger()

import argparse
import glob
import multiprocessing as mp
import os

# fmt: off
import sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import tempfile
import time
import warnings

import cv2
import numpy as np
import tqdm
from torch.nn import functional as F
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
from detectron2.structures import Boxes, ImageList, Instances, BitMasks


from mask2former import add_maskformer2_config
from predictor import VisualizationDemo

from torch2trt.torch2trt import *
from mask2former.modeling.transformer_decoder.position_encoding import PositionEmbeddingSine
from mask2former.modeling.pixel_decoder.msdeformattn import MSDeformAttnPixelDecoder
from mask2former.modeling.pixel_decoder.ops.modules.ms_deform_attn import MSDeformAttnFunctionModule

from mask2former.modeling.transformer_decoder.mask2former_transformer_decoder import MultiScaleMaskedTransformerDecoder
import ctypes

import  logging


@tensorrt_converter(MultiScaleMaskedTransformerDecoder.convert_attn_mask_float_value)
def convert_convert_attn_mask_float_value(ctx):


    module = get_arg(ctx, 'self', pos=0, default=None)
    input_tensor = get_arg(ctx, 'repeated_mask', pos=1, default=None)

    output = ctx.method_return
    # 获取输入张量

    input_trt, threshold_constant_trt = add_missing_trt_tensors(ctx.network, [input_tensor, 0.5])
    input_trt, threshold_constant_trt = broadcast_trt_tensors(ctx.network, [input_trt, threshold_constant_trt],
                                                     max(len(input_trt.shape), len(threshold_constant_trt.shape)))

    _, zero_constant_trt = add_missing_trt_tensors(ctx.network, [input_tensor, 0])
    _, zero_constant_trt = broadcast_trt_tensors(ctx.network, [input_trt, zero_constant_trt],
                                                     max(len(input_trt.shape), len(zero_constant_trt.shape)))
    _, neg_constant_trt = add_missing_trt_tensors(ctx.network, [input_tensor, float("-inf")])
    _, neg_constant_trt = broadcast_trt_tensors(ctx.network, [input_trt, neg_constant_trt],
                                                 max(len(input_trt.shape), len(neg_constant_trt.shape)))

    _, one_constant_trt = add_missing_trt_tensors(ctx.network, [input_tensor, 1.0])
    _, one_constant_trt = broadcast_trt_tensors(ctx.network, [input_trt, one_constant_trt],
                                                max(len(input_trt.shape), len(one_constant_trt.shape)))

    # Create shape constant with proper shape for broadcasting
    shape_length = input_trt.shape[-1]
    shape_dims = [1] * len(input_trt.shape)
    shape_dims[-1] = 1  # Match the reduced dimension
    _, shape_constant_trt = add_missing_trt_tensors(ctx.network,
                                                    [input_tensor, float(shape_length)])
    _, shape_constant_trt = broadcast_trt_tensors(ctx.network, [input_trt, shape_constant_trt],
                                                max(len(input_trt.shape), len(shape_constant_trt.shape)))

    # bool compare_less = input < 0.5 ?
    compare_less = ctx.network.add_elementwise(
        input_trt,
        threshold_constant_trt,
        trt.ElementWiseOperation.LESS
    )
    # if compare_less  bool_to_float = 1 else bool_to_float = 0
    bool_to_float = ctx.network.add_select(
        compare_less.get_output(0),
        one_constant_trt,
        zero_constant_trt
    )

    #  reduce_layer = ADD compare_less
    reduce_layer = ctx.network.add_reduce(
        bool_to_float.get_output(0),
        trt.ReduceOperation.SUM,
        (1 << (len(input_trt.shape) - 1)),  # Reduce last dimension
        keep_dims=True  # Keep dimensions to make broadcasting work
    )

    # bool all_true_compare = ( reduce_layer == shape)
    all_true_compare = ctx.network.add_elementwise(
        reduce_layer.get_output(0),
        shape_constant_trt,
        trt.ElementWiseOperation.EQUAL
    )

    # Broadcast the comparison result back to original shape if needed
    if len(all_true_compare.get_output(0).shape) != len(input_trt.shape):
        shuffle = ctx.network.add_shuffle(all_true_compare.get_output(0))
        shuffle.reshape_dims = input_trt.shape
        broadcast_compare = shuffle.get_output(0)
    else:
        broadcast_compare = all_true_compare.get_output(0)

    # if all_true_compare updated_compare = 0  else updated_compare = bool_to_float
    updated_compare = ctx.network.add_select(
        broadcast_compare,
        zero_constant_trt,
        bool_to_float.get_output(0)
    )

    # if bool_to_float > 0  bool_to_float = TRUE else bool_to_float = FLASE
    float_to_bool = ctx.network.add_elementwise(
        updated_compare.get_output(0),
        zero_constant_trt,
        trt.ElementWiseOperation.GREATER
    )

    final_mask = ctx.network.add_select(
        float_to_bool.get_output(0),
        neg_constant_trt,
        zero_constant_trt
    )

    output._trt = final_mask.get_output(0)



@tensorrt_converter(MSDeformAttnFunctionModule.forward)
def convert_multiscale_deformableAttn(ctx):
    ctypes.CDLL('libtorch2trt_plugins.so')

    module = get_arg(ctx, 'self', pos=0, default=None)
    value = get_arg(ctx, 'value', pos=1, default=None)
    shapes = get_arg(ctx, 'shapes', pos=2, default=None)
    level_start_index = get_arg(ctx, 'level_start_index', pos=3, default=None)
    sampling_locations = get_arg(ctx, 'sampling_locations', pos=4, default=None)
    attention_weights = get_arg(ctx, 'attention_weights', pos=5, default=None)

    output = ctx.method_return
    size = make_size_wrapper(output.shape)

    # registry = trt.get_plugin_registry()
    # creator = registry.get_plugin_creator('MultiscaleDeformableAttnPlugin_TRT', '1', '')
    registry = trt.get_plugin_registry()

    plugin = None
    for plugin_creator in registry.plugin_creator_list:

        if plugin_creator.name == 'MultiscaleDeformableAttnPlugin_TRT':  # 替换为你的插件名称
            field_collection = trt.PluginFieldCollection([])
            plugin = plugin_creator.create_plugin(name=plugin_creator.name, field_collection=field_collection)
            #break

    value_trt = value._trt

    shapes_trt = add_missing_trt_tensors(ctx.network, [shapes])[0]
    level_start_index_trt = add_missing_trt_tensors(ctx.network, [level_start_index])[0]

    sampling_locations_trt = sampling_locations._trt
    attention_weights_trt = attention_weights._trt
    layer = ctx.network.add_plugin_v2(
        [value_trt, shapes_trt, level_start_index_trt, sampling_locations_trt, attention_weights_trt], plugin)

    # output._trt = layer.get_output(0)

    layer = ctx.network.add_shuffle(layer.get_output(0))
    layer.set_input(1, size._trt)
    output._trt = layer.get_output(0)


@tensorrt_converter(PositionEmbeddingSine.forward)
def convert_PositionEmbeddingSine(ctx):
    output = ctx.method_return

    output_np = output.detach().cpu().numpy()
    output._trt = ctx.network.add_constant(output.shape, np.ascontiguousarray(output_np)).get_output(0)

@tensorrt_converter(MSDeformAttnPixelDecoder.y_split)
def convert_MSDeformAttnPixelDecoder_y_split(ctx):
    # module  = get_arg(ctx, 'self', pos=0, default=None)
    y = get_arg(ctx, 'y', pos=1, default=None)

    y_trt = add_missing_trt_tensors(ctx.network, [y])[0]
    outputs = ctx.method_return

    dim = 1
    start = [0] * len(y.shape)
    stride = [1] * len(start)
    offset = 0
    for i, output in enumerate(outputs):
        shape = list(output.shape)

        start[dim] = offset
        layer = ctx.network.add_slice(y_trt, start=start, shape=shape, stride=stride)
        output._trt = layer.get_output(0)
        offset = offset + shape[dim]


# constants
WINDOW_NAME = "mask2former demo"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="maskformer2 demo for builtin configs")
    parser.add_argument(
        "--config-file",

        #default="../configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml",
        default="../configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        default=["../test/test_dog.jpg"],
        help="A list of space separated input images; "
             "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        default="../",
        help="A file or directory to save output visualizations. "
             "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",

        #default=["MODEL.WEIGHTS", "../model_final_3c8ec9.pkl"],  # 添加默认值
        default=["MODEL.WEIGHTS", "../model_final_94dc52.pkl"],  # 添加默认值
        nargs=argparse.REMAINDER,
    )
    return parser





class MaskFormer_TRT_model(torch.nn.Module):
    def __init__(self, backbone, sem_seg_head, cfg, input_size):
        super().__init__()
        self.sem_seg_head = sem_seg_head
        self.backbone = backbone

        self.num_queries = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        self.test_topk_per_image = cfg.TEST.DETECTIONS_PER_IMAGE

        self.semantic_on = cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
        self.instance_on = cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
        self.panoptic_on = cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,

        self.semantic_on = self.semantic_on[0]
        self.instance_on = self.instance_on[0]
        self.panoptic_on = self.panoptic_on[0]
        # Ensure that exactly one of them is True

        self.device = "cuda"

        self.batch_size = 1

        self.input_size = input_size
        sem_seg_head.predictor.set_export()

    def postprccess_img(self, model_output, aug_image, saved_name):

        processed_results = []
        if self.panoptic_on:
            labels, scores, maskpreds = model_output

            processed_results.append({})
            prediction = model.panoptic_inference_after_trt(labels, scores, maskpreds)
            processed_results[-1]["panoptic_seg"] = prediction

        elif self.instance_on:
            final_scores, pred_classes, pred_masks = model_output
            result = Instances(self.input_size)
            result.pred_masks =  pred_masks[0]
            result.pred_boxes = Boxes(torch.zeros(pred_masks[0].size(0), 4))
            result.pred_classes = pred_classes[0].int()
            result.scores = final_scores[0]

            processed_results.append({})
            processed_results[-1]["instances"] = result

        elif self.semantic_on:
            semseg = model_output
            processed_results.append({})
            processed_results[-1]["sem_seg"] = semseg[0]

        predictions, visualized_output = demo.run_on_prediction(processed_results[0], aug_image)

        if os.path.isdir(args.output):
            assert os.path.isdir(args.output), args.output
            out_filename = os.path.join(args.output, os.path.basename(saved_name))
        else:
            assert len(args.input) == 1, "Please specify a directory with args.output"
            out_filename = args.output
        visualized_output.save(out_filename)

    def larger(self, selected_masks, num):
        pred_masks = (selected_masks > num).float()
        return pred_masks

    def get_topk_predictions(self, scores, mask_pred):
        """
        获取topk的预测结果
        Args:
            scores: shape [B, Q, K]
            labels: shape [B, Q*K]
            mask_pred: shape [B, Q, H, W]
            batch_size: int
        Returns:
            scores_per_image: shape [B, topk]
            labels_per_image: shape [B, topk]
            selected_masks: shape [B, topk, H, W]
        """
        # 构建标签张量 [B, Q, K] -> [B, Q*K]
        labels = torch.arange(self.sem_seg_head.num_classes, device=self.device) \
            .unsqueeze(0).unsqueeze(0) \
            .repeat(self.batch_size, self.num_queries, 1) \
            .reshape(self.batch_size, -1)

        # [B, topk]
        scores_per_image, topk_indices = scores.topk(self.test_topk_per_image, dim=1, sorted=True)

        # 获取预测类别 [B, topk]
        labels_per_image = labels[torch.arange(self.batch_size, device=self.device).unsqueeze(1), topk_indices]

        labels_per_image = labels_per_image.float()
        # 获取对应的mask索引 [B, topk]
        mask_indices = topk_indices // self.sem_seg_head.num_classes

        # 获取对应的mask预测 [B, topk, H, W]
        batch_indices = torch.arange(self. batch_size, device=self.device)[:, None]
        selected_masks = mask_pred[batch_indices, mask_indices]

        return scores_per_image, labels_per_image, selected_masks
    def panoptic_inference_max_float(self, softmax_output):
        scores, labels = softmax_output.max(2)
        return scores, labels.float()

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def panoptic_inference(self, mask_cls, mask_pred):
        softmax_output = F.softmax(mask_cls, dim=-1)
        scores, labels = self.panoptic_inference_max_float(softmax_output)
        mask_pred = mask_pred.sigmoid()
        bsz = scores.shape[0]
        #cur_prob_masks = scores.view(bsz, -1, 1, 1) * mask_pred
        return labels, scores, mask_pred

    def instance_inference(self, mask_cls, mask_pred):
        """
        批量推理，只返回scores和预测类别
        Args:
            mask_cls: shape [B, Q, K+1]
            mask_pred: shape [B, Q, H, W]
        Returns:
            scores: shape [B, topk]
            pred_classes: shape [B, topk]
        """
        # [B, Q, K]
        scores = F.softmax(mask_cls, dim=-1)[:, :, :-1]


        # 获取topk预测结果
        scores_per_image, pred_classes, selected_masks = self.get_topk_predictions(
            scores.reshape(self. batch_size, -1), mask_pred)

        # # 计算mask scores
        pred_masks = self.larger(selected_masks, 0 )
        #
        # # Step 1: Compute the sigmoid of selected_masks
        sigmoid_selected_masks = selected_masks.sigmoid()
        #
        # # Step 2: Flatten the sigmoid_selected_masks and pred_masks
        flattened_sigmoid_selected_masks = sigmoid_selected_masks.flatten(2)
        flattened_pred_masks = pred_masks.flatten(2)
        #
        # # Step 3: Compute the numerator
        numerator = (flattened_sigmoid_selected_masks * flattened_pred_masks).sum(2)
        #
        # # Step 4: Compute the denominator
        denominator = flattened_pred_masks.sum(2) + 1e-6
        #
        # # Step 5: Compute the final mask_scores
        #
        mask_scores = numerator / denominator
        #
        # # 最终scores [B, topk]
        final_scores = scores_per_image * mask_scores

        return final_scores, pred_classes, pred_masks


    def forward(self, input):
        features = self.backbone(input)

        outputs = self.sem_seg_head(features)
        mask_cls_results = outputs["pred_logits"]
        mask_pred_results = outputs["pred_masks"]

        # return mask_cls_results, mask_pred_results
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size= self.input_size,
            mode="bilinear",
            align_corners=False,
        )

        if self.panoptic_on:
            outputs = self.panoptic_inference(mask_cls_results, mask_pred_results)
        elif self.instance_on:
            outputs = self.instance_inference(mask_cls_results, mask_pred_results)
        elif self.semantic_on:
            outputs = self.semantic_inference(mask_cls_results, mask_pred_results)

        return outputs

@tensorrt_converter(MaskFormer_TRT_model.panoptic_inference_max_float)
def convert_panoptic_inference_max_float(ctx):
    self = ctx.method_args[0]
    input = ctx.method_args[1]
    output = ctx.method_return
    dim = 2
    keepdim = False

    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]

    k = 1  # We only want the maximum value
    topk_layer = ctx.network.add_topk(input_trt, trt.TopKOperation.MAX, k, torch_dim_to_trt_axes(dim))
    shuffle_layer_values = ctx.network.add_shuffle(topk_layer.get_output(0))
    shuffle_layer_indices = ctx.network.add_shuffle(topk_layer.get_output(1))

    # 使用输入的shape来设置reshape维度
    input_shape = input_trt.shape
    shuffle_layer_values.reshape_dims = (input_shape[0], input_shape[1])  # 保持前两维
    shuffle_layer_indices.reshape_dims = (input_shape[0], input_shape[1])


    cast_layer = ctx.network.add_cast(shuffle_layer_indices.get_output(0), trt.DataType.FLOAT)

    output[0]._trt = shuffle_layer_values.get_output(0)
    output[1]._trt = cast_layer.get_output(0)



@tensorrt_converter(MaskFormer_TRT_model.larger)
def convert_larger(ctx):
    self = ctx.method_args[0]
    selected_masks = ctx.method_args[1]
    num = ctx.method_args[2]

    output = ctx.method_return
    input_a_trt, input_b_trt = add_missing_trt_tensors(ctx.network, [selected_masks, num])
    input_a_trt, input_b_trt = broadcast_trt_tensors(ctx.network, [input_a_trt, input_b_trt],
                                                     max(len(input_a_trt.shape), len(input_b_trt.shape)))
    layer = ctx.network.add_elementwise(input_a_trt, input_b_trt, trt.ElementWiseOperation.GREATER)

    cast_layer = ctx.network.add_cast(layer.get_output(0), trt.DataType.FLOAT)

    output._trt = cast_layer.get_output(0)


@tensorrt_converter(MaskFormer_TRT_model.get_topk_predictions)
def convert_get_topk_predictions(ctx):
    # Get input arguments
    self = ctx.method_args[0]
    scores = ctx.method_args[1]
    mask_pred = ctx.method_args[2]

    output = ctx.method_return

    scores_trt = add_missing_trt_tensors(ctx.network, [scores])[0]
    mask_pred_trt = add_missing_trt_tensors(ctx.network, [mask_pred])[0]

    labels_np = np.tile(
        np.arange(self.sem_seg_head.num_classes, dtype=np.float32)[None, None, :],
        (self.batch_size, self.num_queries, 1)
    ).reshape(self.batch_size, -1)

    labels_trt = ctx.network.add_constant(labels_np.shape, np.ascontiguousarray(labels_np)).get_output(0)

    # Get topk scores and indices [B, topk]
    topk_layer = ctx.network.add_topk(input = scores_trt,
                                      op =trt.TopKOperation.MAX,
                                      k = self.test_topk_per_image,
                                      axes= 1 << 1)

    scores_output = topk_layer.get_output(0)  # Values
    topk_indices = topk_layer.get_output(1)  # Indices


    # Calculate mask_indices (topk_indices // num_classes)
    num_classes_constant = trt.Weights(np.array([self.sem_seg_head.num_classes], dtype=np.int32))
    num_classes_layer = ctx.network.add_constant(trt.Dims([1]), num_classes_constant)

    topk_indices, num_classes = broadcast_trt_tensors(ctx.network, [topk_indices, num_classes_layer.get_output(0)], len(topk_indices.shape))

    mask_indices = ctx.network.add_elementwise(topk_indices,
                                               num_classes,
                                               trt.ElementWiseOperation.FLOOR_DIV)

    batch_size = scores.shape[0]
    batch_indices = np.arange(batch_size, dtype=np.int32)
    batch_indices = np.repeat(batch_indices[:, np.newaxis], self.test_topk_per_image, axis=1)
    batch_indices_constant = ctx.network.add_constant(batch_indices.shape,
                                                      np.ascontiguousarray(batch_indices)).get_output(0)

    # Combine batch indices and topk indices for labels
    # Reshape tensors to [B*topk]
    batch_size_topk = batch_size * self.test_topk_per_image
    batch_indices_reshape = ctx.network.add_shuffle(batch_indices_constant)
    batch_indices_reshape.reshape_dims = trt.Dims([batch_size_topk])

    topk_indices_reshape = ctx.network.add_shuffle(topk_indices)
    topk_indices_reshape.reshape_dims = trt.Dims([batch_size_topk])

    # Gather operation for labels using flattened indices
    gather_labels = ctx.network.add_gather(labels_trt,
                                           topk_indices_reshape.get_output(0),
                                           1)  # gather along dim 1

    # Reshape labels back to [B, topk]
    labels_reshape = ctx.network.add_shuffle(gather_labels.get_output(0))
    labels_reshape.reshape_dims = trt.Dims([batch_size, self.test_topk_per_image])

    # Similar process for masks using mask_indices
    mask_indices_reshape = ctx.network.add_shuffle(mask_indices.get_output(0))
    mask_indices_reshape.reshape_dims = trt.Dims([batch_size_topk])

    gather_masks = ctx.network.add_gather(mask_pred_trt,
                                          mask_indices_reshape.get_output(0),
                                          1)  # gather along dim 1

    masks_reshape = ctx.network.add_shuffle(gather_masks.get_output(0))
    masks_reshape.reshape_dims = trt.Dims([batch_size, self.test_topk_per_image,
                                           mask_pred.shape[2], mask_pred.shape[3]])

    output[0]._trt = scores_output
    output[1]._trt = labels_reshape.get_output(0)
    output[2]._trt = masks_reshape.get_output(0)



def preprocess_img(img, cfg):
    import detectron2.data.transforms as T
    from detectron2.structures import Boxes, ImageList, Instances, BitMasks
    aug = T.ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
    )
    img = img[:, :, ::-1]
    aug_image = aug.get_transform(img).apply_image(img)

    image = torch.as_tensor(aug_image.astype("float32").transpose(2, 0, 1))

    pixel_mean = cfg.MODEL.PIXEL_MEAN
    pixel_mean = torch.tensor(pixel_mean).view(3, 1, 1)
    pixel_std = cfg.MODEL.PIXEL_STD
    pixel_std = torch.tensor(pixel_std).view(3, 1, 1)



    image = (image - pixel_mean) / pixel_std
    images = ImageList.from_tensors([image], 32)

    return images[0],aug_image



if __name__ == "__main__":

    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)


    demo = VisualizationDemo(cfg)
    model = demo.predictor.model
    model.eval()

    #print(model.sem_seg_head.predictor.transformer_self_attention_layers[0].self_attn.in_proj_weight.data)
    # print(model)
    backbone = model.backbone
    sem_seg_head = model.sem_seg_head
    #print(model.sem_seg_head.predictor.transformer_self_attention_layers[0].self_attn.in_proj_weight.data)
    test_image = read_image(args.input[0], format="BGR")
    image, aug_image= preprocess_img(test_image,cfg)

    image = image.unsqueeze(0).cuda()

    input_size = (image.shape[-2], image.shape[-1])

    trt_model = MaskFormer_TRT_model(backbone, sem_seg_head, cfg, input_size)
    x = torch.zeros((1, 3, image.shape[-2], image.shape[-1])).cuda()

    out1 = trt_model(image)

    model_trt = torch2trt(trt_model, [x], max_workspace_size=1 << 30)
    out = model_trt(image)

    trt_model.postprccess_img(out, aug_image, "test_dog_trt_result.jpg")
    trt_model.postprccess_img(out1, aug_image, "test_dog_result.jpg")



    # error_logits = torch.abs(out["pred_logits"] - out1["pred_logits"])
    # error_masks = torch.abs(out["pred_masks"] - out1["pred_masks"])
    #
    # print("绝对误差最大值:")
    # print(f"logits: {torch.max(error_logits)}")
    # print(f"masks: {torch.max(error_masks)}")



    #print((torch.max(torch.abs(out[2] - out1[2]))))
    #
    #
    # print((torch.max(torch.abs(out['res2'] - out1['res2'])) ))
    # print((torch.max(torch.abs(out['res3'] - out1['res3']))))
    # print((torch.max(torch.abs(out['res4'] - out1['res4'])) ))
    # print((torch.max(torch.abs(out['res5'] - out1['res5']))))
    # mask_cls_results, mask_pred_results = trt_model(image)
    #
    # postprccess_img(mask_cls_results, mask_pred_results)
    #
    #
    # model_trt = torch2trt(trt_model, [x], max_workspace_size=1 << 30)
    # out = model_trt(image)

