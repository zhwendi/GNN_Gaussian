import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import hurst
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# 数据归一化函数
def normalize_data(data):
    if isinstance(data, np.ndarray):
        data_min = data.min(axis=0, keepdims=True)
        data_max = data.max(axis=0, keepdims=True)
    elif isinstance(data, torch.Tensor):
        data_min = data.min(dim=0, keepdim=True).values
        data_max = data.max(dim=0, keepdim=True).values
    else:
        raise TypeError("Unsupported data type. Only NumPy arrays and PyTorch tensors are supported.")

    # 检查数据范围是否为零
    if np.all(data_max == data_min):
        # 如果数据范围为零，则在分母上加上1e-8
        return (data - data_min) / (data_max - data_min + 1e-8)
    else:
        # 正常归一化
        return (data - data_min) / (data_max - data_min)

# def normalize_data(data):
#     data_min = np.min(data)
#     data_max = np.max(data)
#
#     # 检查 data_max 和 data_min 是否相等
#     if data_max == data_min:
#         # 处理数据范围为零的情况
#         # 例如，将所有值设置为 0.5 或返回原始数据
#         return np.zeros_like(data)
#     else:
#         # 正常归一化
#         return (data - data_min) / (data_max - data_min)

# 赫斯特指数生成函数
def Seasonal_DFA_alpha(x):
    DF_x = x.detach().numpy().flatten()
    if len(DF_x) < 100:
        raise ValueError("Series length must be greater or equal to 100")
    H, _, _ = hurst.compute_Hc(DF_x, kind='change')
    print(H)
    if H <= 0.5:
        alpha, beta = 0.999, 0.999
    else:
        alpha = np.clip(8 * (1 - H), 0.0, 0.999)
        beta = np.clip(2 * np.power(10, 0.5 - H), 0.0, 0.999)

    print("α=", alpha, "β=", beta)
    return alpha, beta

# 定义GNN模型
class GNNModel(nn.Module):
    def __init__(self, node_features, edge_features):
        super(GNNModel, self).__init__()
        self.node_features = node_features
        self.edge_features = edge_features

    def forward(self, x, edge_attr, edge_index):
        updated_x = x.clone()
        for t in range(1, x.size(0)):
            temp_x = x[t - 1].clone()
            for i in range(edge_index.shape[1]):
                start_node = edge_index[0, i]
                end_node = edge_index[1, i]
                temp_x[end_node] = temp_x[start_node] - edge_attr[t - 1, i]
            updated_x[t] = temp_x.clone()
        return updated_x

# 高斯卷积核
class GaussianConv(nn.Module):
    def __init__(self, node_features, initial_alpha, initial_beta):
        super(GaussianConv, self).__init__()
        self.node_features = node_features
        self.alpha = nn.Parameter(torch.tensor(initial_alpha, requires_grad=True))
        self.beta = nn.Parameter(torch.tensor(initial_beta, requires_grad=True))

    def forward(self, x, alpha=None, beta=None):
        if alpha is None:
            alpha = self.alpha
        if beta is None:
            beta = self.beta
        gauss_kernel = torch.exp(-((torch.arange(self.node_features, dtype=torch.float32) - alpha) ** 2) / (2 * beta ** 2))
        gauss_kernel = gauss_kernel / torch.sum(gauss_kernel)
        gauss_kernel = gauss_kernel.view(1, 1, -1)
        x = x.unsqueeze(1)
        kernel_size = gauss_kernel.size(2)
        padding = (kernel_size - 1) // 2
        x_conv = F.conv1d(x, gauss_kernel, padding=padding)
        return x_conv.squeeze(1)

# 计算RMSE
# def rmse_loss(output, target):
#     return torch.sqrt(F.mse_loss(output, target))
#
# # 计算MAE
# def mae_loss(output, target):
#     return F.l1_loss(output, target)
#
# # 计算RMSE
# def final_rmse_loss(output, target):
#     return torch.sqrt(F.mse_loss(output, target, reduction='none'))
#
# # 计算MAE
# def final_mae_loss(output, target):
#     return F.l1_loss(output, target, reduction='none')


# 计算RMSE
def rmse_loss(output, target):
    differences = (output - target) ** 2
    mean_square_error = torch.mean(differences)
    return torch.sqrt(mean_square_error)

# 计算MAE
def mae_loss(output, target):
    differences = torch.abs(output - target)
    return torch.mean(differences)

# 计算MAPE
def mape_loss(output, target):
    # 避免除以零的情况，可以添加一个非常小的值epsilon
    # epsilon = 1e-8
    differences = torch.abs((output - target) / (target))
    return torch.mean(differences)


# 计算RMSE（每个样本的损失，不取平均）
def final_rmse_loss(output, target):
    differences = (output - target) ** 2
    mse_per_column = torch.mean(differences, dim=0)
    # 计算每一列的均方根误差（RMSE）
    rmse_per_column = torch.sqrt(mse_per_column)
    # print(rmse_per_column.shape)
    return rmse_per_column

# 计算MAE（每个样本的损失，不取平均）
def final_mae_loss(output, target):
    differences = torch.abs(output - target)
    mae_per_column = torch.mean(differences, dim=0)
    # print(mae_per_column.shape)
    return mae_per_column

# 计算MAPE（每个样本的损失，不取平均）
def final_mape_loss(output, target):
    # 避免除以零的情况，可以添加一个非常小的值epsilon
    # epsilon = 1e-8
    # differences = torch.abs((output - target) / (target + epsilon))
    differences = torch.abs((output - target) / target)
    mape_per_column = torch.mean(differences, dim=0)*100
    # print(mape_per_column.shape)
    return mape_per_column


# 训练模型并返回总RMSE损失
def train_model(data_path):
    df = pd.read_csv(data_path)

    data_np = np.nan_to_num(df.apply(pd.to_numeric, errors='coerce').to_numpy())
    x = torch.tensor(normalize_data(data_np[0:len(data_np)-576, :]), dtype=torch.float32)

    edge_index = torch.tensor([[73,5,154,263,56,96,42,58,95,72,271,134,107,130,227,167,298,209,146,170,173,117,0,92,243,203,80,97,28,57,55,223,143,269,290,110,
                                121,299,293,148,150,98,70,128,131,132,242,18,43,118,207,169,127,208,297,168,166,13,26,94,219,217,31,215,111,116,36,301,273,138,284,
                                114,245,48,206,144,237,304,35,115,86,214,27,216,218,76,238,91,52,75,44,256,1,212,270,32,3,247,249,261,260,259,103,302,104,71,88,268,
                                240,9,239,23,22,276,155,157,158,286,102,285,15,8,300,34,161,125,235,163,236,250,122,252,69,39,234,82,274,175,177,213,179,33,181,183,
                                184,185,254,188,141,278,289,190,192,194,196,198,266,135,54,231,66,59,112,14,145,228,205,244,100,303,136,305,153,151,149,295,291,294,
                                77,109,292,147,29,222,79,90,2,81,224,229,4,246,93,87,74,165,241,108,137,123,37,84,101,221,220,201,211,210,262,6,200,199,197,195,193,
                                191,189,280,279,140,187,186,296,126,182,248,25,178,142,176,174,113,124,253,30,67,164,119,120,61,19,162,51,160,159,64,287,267,288,283,
                                281,282,60,38,65,230,47,265,99,53,45,12,41,272,106,17,63,202,16,40,21,233,257,235,264,19,266,122,70,164,70,122,1,252,164,1,119,179,
                                128,246,65,11,3,65,246,231,3,231,247,47,35,265,135,35,47,116,135,116,54,240,10,240,108,91,208,104,209,104,207,251,118,85,57,242,57,
                                243,109,276,143,133,97,68,79,27,27,46,217,98,217,46],
                               [5,154,263,56,96,42,58,95,72,271,68,107,130,129,167,298,209,146,170,173,117,0,92,243,62,80,97,28,57,55,223,143,269,290,110,121,299,
                                293,148,150,152,70,255,131,132,133,18,43,118,207,169,127,208,297,168,166,226,26,94,219,217,31,215,111,116,36,78,20,138,284,114,245,
                                48,206,144,172,24,35,115,86,214,27,216,218,76,238,50,52,75,171,7,1,46,270,32,10,247,249,225,260,259,103,302,104,71,88,268,240,9,239,
                                23,22,49,258,157,158,286,102,285,15,8,300,34,161,125,235,163,236,250,122,252,69,39,234,82,274,175,177,213,179,33,181,183,184,185,254,
                                188,141,278,289,190,192,194,196,198,180,135,54,231,66,59,232,14,89,228,205,244,100,303,136,305,139,151,149,295,291,294,77,109,292,147,
                                29,222,79,90,2,81,204,229,4,246,11,87,74,165,241,108,137,123,37,84,101,221,220,201,211,210,262,306,83,199,197,195,193,191,189,280,279,
                                140,187,186,296,126,182,248,25,178,142,176,174,113,124,253,30,67,164,119,120,61,19,162,51,160,159,64,287,267,288,283,281,282,156,38,
                                65,230,47,265,264,53,45,12,41,272,106,17,63,202,16,40,105,233,277,275,264,163,266,162,70,252,70,119,1,252,46,1,119,46,128,33,65,11,
                                230,230,247,59,11,231,247,59,35,265,115,115,54,36,265,116,54,36,10,9,91,91,137,104,127,104,146,251,118,85,207,242,28,243,28,276,292,
                                133,269,68,57,99,98,46,216,98,31,46,31]],
                              dtype=torch.long)

    edge_attr_pd = pd.read_csv('F:/AAA/GNN_Gaussian/data/PEMS04/edge_attr.csv')
    edge_attr_np = np.nan_to_num(edge_attr_pd.to_numpy(dtype=np.float32))
    edge_attr = torch.tensor(normalize_data(edge_attr_np[0:len(data_np)-576, :]), dtype=torch.float32)

    data = TensorDataset(x, edge_attr)
    loader = DataLoader(data, batch_size=864, shuffle=False)

    total_final_rmse = torch.zeros(x.size(1))
    total_final_mae = torch.zeros(x.size(1))
    total_final_mape = torch.zeros(x.size(1))
    rmse_losses = []
    mae_losses = []
    mape_losses = []
    best_alpha_values = []
    best_beta_values = []

    num_batches = len(loader)

    for batch_idx, (batch_data, batch_edge_attr) in enumerate(loader):
        print(f"Processing Batch {batch_idx + 1}")

        data_chunk = batch_data[:576]
        initial_alpha, initial_beta = Seasonal_DFA_alpha(data_chunk)

        gnn = GNNModel(node_features=x.size(1), edge_features=edge_attr.size(1))
        updated_x = gnn(data_chunk, batch_edge_attr[:576], edge_index)

        gauss_conv = GaussianConv(node_features=x.size(1), initial_alpha=initial_alpha, initial_beta=initial_beta)
        optimizer = torch.optim.Adam([gauss_conv.alpha, gauss_conv.beta], lr=0.0001)

        best_rmse = float('inf')
        best_alpha = None
        best_beta = None

        for iteration in range(30):
            x_conv = gauss_conv(updated_x)

            rmse = final_rmse_loss(x_conv, updated_x)
            mae = final_mae_loss(x_conv, updated_x)
            mape = final_mape_loss(x_conv, updated_x)

            optimizer.zero_grad()
            rmse.mean().backward(retain_graph=True)
            optimizer.step()

            for i in range(rmse.size(0)):
                print(
                    f"Iteration {iteration + 1} - RMSE: {rmse[i].item()}, MAE: {mae[i].item()}, MAPE: {mape[i].item()}, Alpha: {gauss_conv.alpha.item()}, Beta: {gauss_conv.beta.item()}")

            if rmse.mean().item() < best_rmse:
                best_rmse = rmse.mean().item()
                best_alpha = gauss_conv.alpha.item()
                best_beta = gauss_conv.beta.item()

        saved_alpha = best_alpha
        saved_beta = best_beta
        print(f"Best Loss for first part: RMSE:{best_rmse}, Best Alpha: {saved_alpha}, Best Beta: {saved_beta}")

        remaining_data = batch_data[576:]
        updated_x_remaining = gnn(remaining_data, batch_edge_attr[576:], edge_index)

        final_x_conv = gauss_conv(updated_x_remaining, saved_alpha, saved_beta)
        final_rmse = final_rmse_loss(final_x_conv, updated_x_remaining)
        final_mae = final_mae_loss(final_x_conv, updated_x_remaining)
        final_mape = final_mape_loss(final_x_conv, updated_x_remaining)

        for i in range(final_rmse.size(0)):
            print(
                f"Final Loss for Batch {batch_idx + 1}, Column {i + 1}: RMSE: {final_rmse[i].item()}, MAE: {final_mae[i].item()}, MAPE: {final_mape[i].item()}")

        total_final_rmse += final_rmse
        total_final_mae += final_mae
        total_final_mape += final_mape
        rmse_losses.append(final_rmse.tolist())
        mae_losses.append(final_mae.tolist())
        mape_losses.append(final_mape.tolist())
        best_alpha_values.append(saved_alpha)
        best_beta_values.append(saved_beta)

    avg_final_rmse = total_final_rmse / num_batches
    avg_final_mae = total_final_mae / num_batches
    avg_final_mape = total_final_mape / num_batches

    print(
        f"Average Final RMSE: {avg_final_rmse.tolist()}, Average Final MAE: {avg_final_mae.tolist()}, Average Final MAPE: {avg_final_mape.tolist()}")

    print(
        f"Total Final RMSE: {total_final_rmse.tolist()}, Total Final MSE: {total_final_mae.tolist()}, Total Final MAPE: {total_final_mape.tolist()}")

    # 保存最终的RMSE，best_alpha，best_beta到Excel文件
    df_results = pd.DataFrame({
        'Final RMSE': rmse_losses,
        'Final MAE': mae_losses,
        'Final MAPE': mape_losses,
        'Best Alpha': best_alpha_values,
        'Best Beta': best_beta_values
    })

    df_results.to_excel(f'F:/AAA/GNN_Gaussian/data/PEMS04/{data_path.split("/")[-1].split(".")[0]}_results_2016.xlsx', index=False)

    return avg_final_rmse.tolist(), avg_final_mae.tolist(), avg_final_mape.tolist()

# 训练两个数据集并绘制对比图
data_path1 = 'F:/AAA/GNN_Gaussian/data/PEMS04/output222.csv'
data_path2 = 'F:/AAA/GNN_Gaussian/data/PEMS04/STL/1000/final_data_1000.csv'

avg_rmse1,avg_mae1,avg_mape1 = train_model(data_path1)
avg_rmse2,avg_mae2,avg_mape2 = train_model(data_path2)

# 创建一个DataFrame来保存RMSE损失值
rmse_df = pd.DataFrame({
    'PEMS04 RMSE': avg_rmse1,
    'PEMS04 MAE': avg_mae1,
    'PEMS04 MAPE': avg_mape1,
    'PEMS04_STL RMSE': avg_rmse2,
    'PEMS04_STL MAE': avg_mae2,
    'PEMS04_STL MAPE':avg_mape2
})

# 将DataFrame写入到Excel文件中
rmse_df.to_excel('F:/AAA/GNN_Gaussian/data/PEMS04/total_loss_comparison_2016.xlsx', index=False)

print("RMSE values have been saved to 'total_loss_comparison.xlsx'")

# 绘制对比图
num_columns = 307
num_bars_per_fig = 30  # 每个图显示的条形数
num_figures = (num_columns + num_bars_per_fig - 1) // num_bars_per_fig

for i in range(num_figures):
    start = i * num_bars_per_fig
    end = min(start + num_bars_per_fig, num_columns)
    plt.figure(figsize=(15, 6))
    x = range(start, end)
    width = 0.25
    plt.bar(x, avg_rmse1[start:end], width, label='PEMS04', color='#B8D4E9')
    plt.bar([p + width + 0.05 for p in x],avg_rmse2[start:end], width, label='PEMS04_STL', color='#2F7DBB')
    plt.xlabel('Column')
    plt.ylabel('Total RMSE Loss')
    plt.title(f'Total RMSE Loss per Column for Two Data Sets (Columns {start+1} to {end})')
    plt.legend()
    plt.xticks([p + width / 2 + 0.025 for p in x], x)
    plt.tight_layout()
    plt.savefig(f'F:/AAA/GNN_Gaussian/data/PEMS04/PEMS04_test_rmse_comparison_{i}_2016.png')
    plt.show()