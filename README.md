# mask2former_trt: Use the torch2trt library to convert the official Mask2former model to TensorRT
![中文README](README_CN.md)
### Key Highlights
* The PyTorch model is directly converted to a native model built using the TensorRT API, rather than using torch_tensorrt. The converted model can run completely independently of PyTorch.
* Pure Python implementation for easy one-click conversion.
* A good example of using torch2trt to convert complex PyTorch models.
* Future updates will support more features.

## Challenges in Model Conversion
* In the Encoder section, the msdeformattn operator is implemented as a separate custom CUDA operation, which severely impacts conventional model conversion methods, such as direct conversion to ONNX or TorchScript.
* In the Decoder section, the attn_mask parameter of the native PyTorch nn.multiheadattention() operator does not support 4D tensors that include batches, leading to inconsistencies in input sizes compared to TensorRT.

## Optimization Process
* Added MSDeformableAttnPlugin as a custom plugin to torch2trt.
* Implemented a multiheadattention in PyTorch that supports batch attn_mask parameters.
* Modified a series of implementations in the model and added numerous custom converter functions for torch2trt to ensure smooth conversion.
* Integrated some post-processing steps without branching if statements into the model to further enhance inference speed.

## Notes
* The tested version of TensorRT used in this repository is 8.6.1.6.
* The inference image sizes commonly used in the original Mask2former repository are 800 and 1200. On machines with insufficient memory, conversion may lead to out-of-memory errors. It is recommended to adjust the MIN_SIZE_TEST and MAX_SIZE_TEST parameters in cfg.INPUT to modify the model's input size.
* Due to differences in operator implementation, there may be discrepancies in inference results compared to native PyTorch. If you encounter unacceptable discrepancies during use, please raise an issue for specific analysis.
* Do not use this repository for model training.

## Usage Guide
1. Follow the official Mask2former library instructions to complete the installation of the native Mask2former.
*  See [installation instructions](INSTALL.md).
2. Clone my maintained torch2trt library and compile the newly added MSDeformableAttnPlugin.
```bash
git submoudle init
git submoudle update
cd torch2trt
```
Then change the paths of the TensorRT library and header files in the CMakeLists.txt of torch2trt to your own paths, and then compile and install torch2trt.
```bash
python setup.py install
cmake -B build . && cmake --build build --target install && sudo ldconfig
```

3. Download the weights and test images, then run the script as shown below.
```bash
cd demo/
python demo_trt.py --config-file ../configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml \
  --input input1.jpg \
  [--other-options]
  --opts MODEL.WEIGHTS /path/to/checkpoint_file
```
## Results Showcase
*  The configuration file is panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml 

*  Original Image
![Example Image](test/test_dog.jpg)
* Input image size is 427, 640
* PyTorch test results
![Example Image](test/test_dog_result.jpg)
* TensorRT test results
![Example Image](test/test_dog_trt_result.jpg)

## TO DO
* ~~Support Swin backbone~~  Completed
* Support semantic-segmentation models
* Complete testing and debugging for batch_size > 1
* fp16 int8 quantization
* Convert the mask2former_video model