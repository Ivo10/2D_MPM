import numpy as np
import os


def crop_stack_save_images(image_paths, mask_path, crop_size, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    mask = np.load(mask_path)
    half_size = crop_size // 2

    index = 0
    height, width = mask.shape
    for i in range(height):
        for j in range(width):
            if mask[i, j] == 1:
                cropped_stack = []
                for idx, path in enumerate(image_paths):
                    img_array = np.load(path)

                    top = i - half_size
                    bottom = i + half_size
                    left = j - half_size
                    right = j + half_size

                    # 默认使用zero_padding(mirror padding, copy padding)
                    cropped_img = np.zeros((crop_size, crop_size), dtype=img_array.dtype)
                    # 原图中的有效边界(都是闭区间)
                    valid_top = max(0, top)
                    valid_bottom = min(height - 1, bottom)
                    valid_left = max(0, left)
                    valid_right = min(width - 1, right)

                    # 子图中需要填充原图的边界
                    crop_top = valid_top - top
                    crop_bottom = crop_size - (bottom - valid_bottom) - 1
                    crop_left = valid_left - left
                    crop_right = crop_size - (right - valid_right) - 1

                    cropped_img[crop_top:crop_bottom + 1, crop_left:crop_right + 1] = img_array[
                                                                                      valid_top:valid_bottom + 1,
                                                                                      valid_left:valid_right + 1]
                    cropped_stack.append(cropped_img)

                final_stack = np.stack(cropped_stack, axis=0)
                save_path = os.path.join(output_dir, f'pos_{index}.npy')
                np.save(save_path, final_stack)
                index += 1


if __name__ == '__main__':
    image_paths = ['../datasets/npy/As.npy',
                   '../datasets/npy/Cu.npy',
                   '../datasets/npy/Pb.npy',
                   '../datasets/npy/Zn.npy']
    mask_path = '../datasets/npy/Mask.npy'
    output_dir = '../datasets/npy/cropped_images'
    crop_size = 11
    crop_stack_save_images(image_paths, mask_path, crop_size, output_dir)
