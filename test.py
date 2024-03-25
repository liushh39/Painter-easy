import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import glob
import tqdm
from PIL import Image
import models_painter

# ImageNet 的标准化参数
imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


def get_args_parser():
    parser = argparse.ArgumentParser('demo', add_help=False)
    # 模型路径
    parser.add_argument('--ckpt_path', type=str, help='path to ckpt', default='./painter_vit_large.pth')
    # 选择视觉任务(实际上是选择prompt)
    parser.add_argument('--task', type=str, help='denoise, derain, image_enhancement, instance_segmentation,'
                                                 'keypoint_detection, semantic_segmentation',
                        default='denoise')
    parser.add_argument('--model', type=str, help='dir to ckpt',
                        default='painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1')
    parser.add_argument('--prompt', type=str, help='prompt image in train set',
                        default='100')
    parser.add_argument('--input_size', type=int, default=448)
    return parser.parse_args()


def prepare_model(chkpt_dir, arch='painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1'):
    model = getattr(models_painter, arch)()
    # 加载模型
    checkpoint = torch.load(chkpt_dir, map_location='cuda:0')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model


def run_one_image(img, tgt, size, model, out_path, device):
    # 把输入转换为张量
    x = torch.tensor(img)
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)

    tgt = torch.tensor(tgt)
    tgt = tgt.unsqueeze(dim=0)
    tgt = torch.einsum('nhwc->nchw', tgt)

    bool_masked_pos = torch.zeros(model.patch_embed.num_patches)
    bool_masked_pos[model.patch_embed.num_patches // 2:] = 1
    bool_masked_pos = bool_masked_pos.unsqueeze(dim=0)  # (1,1568)

    valid = torch.ones_like(tgt)
    y, mask = model(x.float().to(device), tgt.float().to(device), bool_masked_pos.to(device), valid.float().to(device))
    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    output = y[0, y.shape[1] // 2:, :, :]
    output = output * imagenet_std + imagenet_mean
    output = F.interpolate(
        output[None, ...].permute(0, 3, 1, 2), size=[size[1], size[0]], mode='bicubic').permute(0, 2, 3, 1)[0]

    return output.numpy()


if __name__ == '__main__':
    args = get_args_parser()

    ckpt_path = args.ckpt_path
    task = args.task
    model = args.model
    prompt = args.prompt
    input_size = args.input_size

    model_painter = prepare_model(ckpt_path, model)
    print('Model loaded.')

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  # 设置是否使用 cuda
    model_painter.to(device)

    # prompt所在路径 输入和目标
    img2_path = "./test_img/" + task + "/prompt/input.png".format(prompt)
    tgt2_path = "./test_img/" + task + "/prompt/gt.png".format(prompt)
    print('prompt: {}'.format(tgt2_path))

    # 读取图片
    img2 = Image.open(img2_path).convert("RGB")
    img2 = img2.resize((input_size, input_size))
    img2 = np.array(img2) / 255.

    tgt2 = Image.open(tgt2_path)
    tgt2 = tgt2.resize((input_size, input_size))
    tgt2 = np.array(tgt2) / 255.

    model_painter.eval()

    # 测试图像的输入和保存路径
    real_src_dir = "./test_img/" + task + "/img"
    real_dst_dir = "./result"
    if not os.path.exists(real_dst_dir):
        os.makedirs(real_dst_dir)
    img_path_list = glob.glob(os.path.join(real_src_dir, "*.png")) + glob.glob(os.path.join(real_src_dir, "*.jpg"))
    for img_path in tqdm.tqdm(img_path_list):
        # 读取图片
        img_name = os.path.basename(img_path)
        out_path = os.path.join(real_dst_dir, img_name)
        img_org = Image.open(img_path).convert("RGB")
        size = img_org.size
        img = img_org.resize((input_size, input_size))
        img = np.array(img) / 255.

        # 对两对图片进行concat
        img = np.concatenate((img2, img), axis=0)
        assert img.shape == (input_size * 2, input_size, 3)
        # 标准化
        img = img - imagenet_mean
        img = img / imagenet_std

        tgt = tgt2  # 测试对象对应的 target 实际是没有的
        tgt = np.concatenate((tgt2, tgt), axis=0)

        assert tgt.shape == (input_size * 2, input_size, 3)
        # 标准化
        tgt = tgt - imagenet_mean
        tgt = tgt / imagenet_std

        # 随机掩码
        torch.manual_seed(2)

        output = run_one_image(img, tgt, size, model_painter, out_path, device)
        rgb_restored = output
        rgb_restored = np.clip(rgb_restored, 0, 1)

        # 保存图像
        output = rgb_restored * 255
        output = Image.fromarray(output.astype(np.uint8))
        output.save(out_path)
