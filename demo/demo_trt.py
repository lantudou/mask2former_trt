# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
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

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from mask2former import add_maskformer2_config
from predictor import VisualizationDemo



from torch2trt.torch2trt import *
from mask2former.modeling.transformer_decoder.position_encoding import PositionEmbeddingSine
from mask2former.modeling.pixel_decoder.msdeformattn import MSDeformAttnPixelDecoder
from mask2former.modeling.pixel_decoder.ops.modules.ms_deform_attn import MSDeformAttnFunctionModule
import ctypes

@tensorrt_converter(MSDeformAttnFunctionModule.forward)
def convert_multiscale_deformableAttn(ctx):
    ctypes.CDLL('libtorch2trt_plugins.so')

    module  = get_arg(ctx, 'self', pos=0, default=None)
    value  = get_arg(ctx, 'value', pos=1, default=None)
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
            plugin = plugin_creator.create_plugin(name=plugin_creator.name, field_collection = field_collection)
            break

    value_trt = value._trt

    shapes_trt = add_missing_trt_tensors(ctx.network, [shapes])[0]
    level_start_index_trt = add_missing_trt_tensors(ctx.network, [level_start_index])[0]


    sampling_locations_trt = sampling_locations._trt
    attention_weights_trt = attention_weights._trt
    layer = ctx.network.add_plugin_v2([value_trt, shapes_trt, level_start_index_trt,sampling_locations_trt, attention_weights_trt ], plugin)
    
    #output._trt = layer.get_output(0)

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
	
    #module  = get_arg(ctx, 'self', pos=0, default=None)
    y  = get_arg(ctx, 'y', pos=1, default=None)
    print(y.shape)
    y_trt = add_missing_trt_tensors(ctx.network, [y])[0]
    outputs = ctx.method_return
    
    dim = 1
    start = [0] * len(y.shape) 
    stride = [1] * len(start)
    offset = 0	
    for i, output in enumerate(outputs):
        shape = list(output.shape) 
        print(shape)
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
        default="/share_data/YuhaoSun/Mask2Former-main/configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        default="/share_data/YuhaoSun/Mask2Former-main/test/n02087046_6166.jpg",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        default="/share_data/YuhaoSun/Mask2Former-main/",
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
        default=["MODEL.WEIGHTS", "/share_data/YuhaoSun/Mask2Former-main/model_final_3c8ec9.pkl"],  # 添加默认值
        nargs=argparse.REMAINDER,
    )
    return parser

class MaskFormer_TRT_model(torch.nn.Module):
		def __init__(self, backbone, sem_seg_head):
			super().__init__()
			self.sem_seg_head = sem_seg_head
			self.backbone = backbone
		def forward(self, input):
			features = self.backbone(input)
			outputs = self.sem_seg_head(features)
			return outputs

def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False


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
    # print(model)
    backbone = model.backbone
    sem_seg_head = model.sem_seg_head


    #out1 = backbone(x)

    model = MaskFormer_TRT_model(backbone, sem_seg_head)
    x = torch.ones((1, 3, 480, 560)).cuda()
    out1 = model(x)
    model_trt = torch2trt(model, [x], max_workspace_size=1<< 29)
    out = model_trt(x)
    #pos = sem_seg_head(out1)
    #test_trt = torch2trt(sem_seg_head, [out1], max_workspace_size=1<< 29)
    #pos_trt = test_trt(out1)
    print(out)

    for x, y in zip(out['aux_outputs'], out1['aux_outputs']):
        print(torch.max(torch.abs(x['pred_logits']- y['pred_logits'])))
        print(torch.max(torch.abs(x['pred_masks'] - y['pred_masks'])))

    # print(torch.max(torch.abs(out['res2'] - out1['res2'] )))
    # print(torch.max(torch.abs(out['res3'] - out1['res3'] )))
    # print(torch.max(torch.abs(out['res4'] - out1['res4'] )))
    # print(torch.max(torch.abs(out['res5'] - out1['res5'] )))
    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )

            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                visualized_output.save(out_filename)
            else:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit
    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        assert args.output is None, "output not yet supported with --webcam!"
        cam = cv2.VideoCapture(0)
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cam.release()
        cv2.destroyAllWindows()
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)
        codec, file_ext = (
            ("x264", ".mkv") if test_opencv_video_format("x264", ".mkv") else ("mp4v", ".mp4")
        )
        if codec == ".mp4v":
            warnings.warn("x264 codec not available, switching to mp4v")
        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + file_ext
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*codec),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            if args.output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()
