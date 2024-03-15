# Painter-easy
<img src="https://github.com/liushh39/Painter/blob/main/framework.jpg" width="600">
- This is the simplified implementation of [Images Speak in Images: A Generalist Painter for In-Context Visual Learning](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_Images_Speak_in_Images_A_Generalist_Painter_for_In-Context_Visual_CVPR_2023_paper.html) (CVPR 2023).
- Since the original [repository](https://github.com/baaivision/Painter) had a lot of files and different ways of loading different datasets, it was easy to get confused. So this repository simplifies and comments the code so that beginners and interested researchers can quickly test it with their own tasks or images.

## 1. Environment
- Linux (Recommend)
- PyTorch >= 1.8.1
- Other requirements
   ```
    pip install -r requirements.txt
   ```
## 2. Dataset
You can make your own datasets based on `test_img`
## 3. Testing
1. Download the pre-trained Painter from [here](https://huggingface.co/BAAI/Painter/blob/main/painter_vit_large.pth). Place it in the same directory with `test.py`.
2. Run the following command:
```
python test.py --task denoise
```
Tasks include denoise, derain, image_enhancement, instance_segmentation, keypoint_detection, and semantic_segmentation.

3. If there is an error reported by the torch version of _six.py, try the following solution:
```
import collections.abc as container_abcs → from torch._six import container_abcs 
```
## 4. Citation
```
@article{Painter,
  title={Images Speak in Images: A Generalist Painter for In-Context Visual Learning},
  author={Wang, Xinlong and Wang, Wen and Cao, Yue and Shen, Chunhua and Huang, Tiejun},
  journal={arXiv preprint arXiv:2212.02499},
  year={2022}
}
```
