# 读取xlsx文件内容，得到数据
import numpy as np
from plot.line import plot_line


# step值越过多少个点再记录
def read_origin_PINN_log(PINN_log_path, step=1):
    loss_Omega = []
    min_loss = 1e8
    with open(PINN_log_path, 'r') as fs:
        line_num = 0
        while True:
            line = fs.readline()  # 整行读取数据
            if not line:
                break
            # 读取loss
            if 'Iter ' in line:
                line_num = line_num + 1
                if line_num % step != 0:
                    continue

                loss = line.split(' loss_region ')[-1].split(
                    ' loss_Gamma ')[0]
                if loss is not None:
                    loss = float(loss)
                    if min_loss > loss:
                        min_loss = loss
                    loss_Omega.append(min_loss)

    return loss_Omega


def read_fictitious_PINN_log(PINN_log_path, step=1):
    loss_Omega = []
    min_loss = 1e8
    with open(PINN_log_path, 'r') as fs:
        line_num = 0
        while True:
            line = fs.readline()  # 整行读取数据
            if not line:
                break
            # 读取loss
            if 'Iter ' in line:
                line_num = line_num + 1
                if line_num % step != 0:
                    continue

                loss = line.split(' loss_Omega ')[-1].split(
                    ' loss_omega ')[0]
                if loss is not None:
                    loss = float(loss)
                    if min_loss > loss:
                        min_loss = loss
                    loss_Omega.append(min_loss)

    return loss_Omega


if __name__ == "__main__":
    root_path = 'c:/softwarefiles/VSCodeProjects/FictitiousDomain/data/'
    datas = []
    ORIGIN_TIME_STR = '20220509_165759'
    ORIGIN_TASK_NAME = 'task_pinn_origin3'

    FICTITIOUS_TIME_STR = '20220509_182504'
    FICTITIOUS_TASK_NAME = 'task_pinn_fictitious_domain7'

    ORIGIN_PINN_log_path = root_path + '/' + \
        ORIGIN_TASK_NAME + '/' + ORIGIN_TIME_STR + '/log.txt'

    origin_loss = read_origin_PINN_log(
        PINN_log_path=ORIGIN_PINN_log_path, step=1)

    FICTITOUS_PINN_log_path = root_path + '/' + \
        FICTITIOUS_TASK_NAME + '/' + FICTITIOUS_TIME_STR + '/log.txt'

    fictitious_loss = read_fictitious_PINN_log(
        PINN_log_path=FICTITOUS_PINN_log_path, step=1)

    Num_Epoch = len(origin_loss)
    epochs = [i for i in range(1, Num_Epoch + 1)]

    data = np.stack((epochs, origin_loss), 1)
    datas.append(data)
    data = np.stack((epochs, fictitious_loss), 1)
    datas.append(data)

    data_labels = ['origin', 'fictitious domain']
    # data_labels = None

    xy_labels = ['Epoch/10', 'Loss']
    plot_line(datas=datas,
              data_labels=data_labels,
              xy_labels=xy_labels,
              title=None,
              file_name=root_path + '/loss_origin3_fictitious7',
              ylog=True)
    print('done')
