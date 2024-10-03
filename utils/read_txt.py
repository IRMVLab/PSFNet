# -*- coding:UTF-8 -*-

import os


def read_pose_eval(txt_path: str):
    with open(txt_path, 'r') as f:
        data = f.readlines()
        length = len(data) // 3
        epoch_list = length * [-1]
        RMSE_list = length * [-1]
        error_list = length * [-1]
        for row in range(length):
            index1 = data[row * 3].index(':') + 2
            index2 = data[row * 3 + 1].index(':') + 2
            index3 = data[row * 3 + 2].index(':') + 2
            ep = data[row * 3][index1:].strip()
            rmse = data[row * 3 + 1][index2:].strip()
            error = data[row * 3 + 2][index3:].strip()
            epoch_list[row] = int(ep)
            RMSE_list[row] = float(rmse)
            error_list[row] = float(error) * 100

        min_RMSE_index = RMSE_list.index(min(RMSE_list))
        min_error_index = error_list.index(min(error_list))
        min_epoch = set()
        min_epoch.add(epoch_list[min_RMSE_index])
        min_epoch.add(epoch_list[min_error_index])
        min_infor = ['{}:{:.4f}'.format(epoch_list[min_RMSE_index], RMSE_list[min_RMSE_index]),
                     '{}:{:.4f}'.format(epoch_list[min_error_index], error_list[min_error_index])]
        return epoch_list, RMSE_list, error_list, min_infor, min_epoch  # [int] [float] [float] [str,str] {int ,int}


def get_poss_data(eval_list: list, file_path: str):
    if len(eval_list) == 0:
        return
    eval_count = len(eval_list)
    pose_loss_data = []
    Rmse_data = []
    Error_data = []
    min_data = []
    save_epoch_set = set()
    for i, num in enumerate(eval_list):
        output_path = os.path.join(file_path, 'pwclonet_{:02d}/output.txt'.format(num))
        epoch_list, RMSE_list, error_list, min_infor, min_epoch = read_pose_eval(output_path)
        if i == 0:
            pose_loss_data.append(['Epoch', epoch_list])
        Rmse_data.append(['{:02d} translational'.format(num), RMSE_list])
        Error_data.append(['{:02d} rotational'.format(num), error_list])
        min_data.append(['{:02d} T/R'.format(num), min_infor])
        save_epoch_set = save_epoch_set | min_epoch  # {int}

    pose_loss_data += Rmse_data
    pose_loss_data += Error_data
    pose_loss_data += min_data

    T_epoch_data = [data[1] for data in Rmse_data]
    R_epoch_data = [data[1] for data in Error_data]
    epoch_count = len(T_epoch_data[0])
    T_mean = [0.0] * epoch_count
    R_mean = [0.0] * epoch_count
    for i in range(epoch_count):
        t_mean = 0
        q_mean = 0
        for t, q in zip(T_epoch_data, R_epoch_data):
            t_mean = t_mean + t[i]
            q_mean = q_mean + q[i]
        t_mean = t_mean / eval_count
        q_mean = q_mean / eval_count
        T_mean[i] = t_mean
        R_mean[i] = q_mean
    T_mean_min_index = T_mean.index(min(T_mean))
    R_mean_min_index = R_mean.index(min(R_mean))
    T_mean_epoch = pose_loss_data[0][1][T_mean_min_index]
    R_mean_epoch = pose_loss_data[0][1][R_mean_min_index]
    save_epoch_set.add(T_mean_epoch)
    save_epoch_set.add(R_mean_epoch)

    mean_list = ['mean min T/R',
                 ['{}:{:.4f}'.format(T_mean_epoch, T_mean[T_mean_min_index]),
                  '{}:{:.4f}'.format(R_mean_epoch, R_mean[R_mean_min_index])]]
    pose_loss_data.append(mean_list)

    return pose_loss_data, save_epoch_set
