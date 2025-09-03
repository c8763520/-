import nibabel as nib
import numpy as np

input_nii = '/home/algo/Downloads/nifti_out/M33360_T1.nii.gz'
output_nii = '/home/algo/Downloads/nifti_out/M33360_T1_gray16.nii.gz'

img = nib.load(input_nii)
data_structured = img.dataobj.get_unscaled()

def structured_rgb_to_array(struct_arr):
    r = struct_arr['R'].astype(np.float32)
    g = struct_arr['G'].astype(np.float32)
    b = struct_arr['B'].astype(np.float32)
    return np.stack([r, g, b], axis=-1)

if data_structured.dtype.names is not None and set(data_structured.dtype.names) >= {'R','G','B'}:
    data_rgb = structured_rgb_to_array(data_structured)
    data_gray = (0.2989 * data_rgb[...,0] + 0.5870 * data_rgb[...,1] + 0.1140 * data_rgb[...,2]).astype(np.int16)
else:
    data_gray = np.array(data_structured).astype(np.int16)

# 新建header並更新資料型態與 scaling 欄位
new_header = img.header.copy()
new_header.set_data_dtype(np.int16)  # 設定資料型態為 int16
new_header['scl_slope'] = 1
new_header['scl_inter'] = 0

new_img = nib.Nifti1Image(data_gray, img.affine, new_header)
nib.save(new_img, output_nii)

print(f'轉換完成並儲存：{output_nii}')
