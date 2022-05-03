import time

import torch

import ipdb

# coords = torch.tensor(
#     [
#         [1,2,3],
#         [4,5,6],
#         [7,8,9],
#         [3,2,1],
#         [6,5,4]
#     ]

# ).float()

coords = torch.rand((int(10000), 3))

rot_mat_pred = torch.tensor(
    [
        [12,1,34],
        [-1,1,1],
        [5,1,-3],
    ]

).float()

rot_mat = torch.tensor(
    [
        [1,33,1],
        [1,22,2],
        [1,11,55],
    ]

).float()

if torch.cuda.is_available():
    coords = coords.cuda()
    rot_mat = rot_mat.cuda()
    rot_mat_pred = rot_mat_pred.cuda()

y_translated = torch.matmul(rot_mat, torch.transpose(coords.float(), 0, 1))
y_pred_translated = torch.matmul(rot_mat_pred, torch.transpose(coords.float(), 0, 1))

t1 = time.time()
loss_rot_instance = 0.0
for j in range(len(coords)):
    y_pred_translated_j_diff = y_pred_translated[:, j:j+1] - y_translated
    norms = torch.linalg.norm(y_pred_translated_j_diff, dim=0)
    loss_rot_instance += torch.pow(norms, 2).min()
t2 = time.time()

print('loss_rot_instance', loss_rot_instance, 'time:', t2-t1)

y_pred_translated_reshaped = y_pred_translated.view(3, 1, -1)
y_pred_translated_permuted = y_pred_translated_reshaped.permute((2, 0, 1))
y_pred_translated_diff = y_pred_translated_permuted - y_translated
norms = torch.linalg.norm(y_pred_translated_diff, dim=1)
loss_rot_instance_2 = torch.pow(norms, 2).min(dim=1).values.sum()

t3 = time.time()

print('loss_rot_instance_2', loss_rot_instance_2, 'time:', t3-t2)

assert torch.isclose(loss_rot_instance, loss_rot_instance_2)

ipdb.set_trace()