import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import hurst
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import  math

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
        # 避免数值下溢，给指数运算的结果加上一个非常小的正数
        gauss_kernel = gauss_kernel.clamp(min=1e-10)

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
# def mape_loss(output, target):
#     # 避免除以零的情况，可以添加一个非常小的值epsilon
#     epsilon = 1e-8
#     # 计算绝对误差，然后除以target，最后取平均值
#     return torch.mean(torch.abs((output - target) / (target + epsilon)))

# # 计算RMSE
# def final_rmse_loss(output, target):
#     return torch.sqrt(F.mse_loss(output, target, reduction='none'))
#
# # 计算MAE
# def final_mae_loss(output, target):
#     return F.l1_loss(output, target, reduction='none')
#
# # 计算MAPE
# def final_mape_loss(output, target):
#     return torch.abs((output - target) / target)


# 计算RMSE
# def rmse_loss(output, target):
#     differences = (output - target) ** 2
#     mean_square_error = torch.mean(differences)
#     return torch.sqrt(mean_square_error)
#
# # 计算MAE
# def mae_loss(output, target):
#     differences = torch.abs(output - target)
#     return torch.mean(differences)
#
# # 计算MAPE
# def mape_loss(output, target):
#     # 避免除以零的情况，可以添加一个非常小的值epsilon
#     # epsilon = 1e-8
#     differences = torch.abs((output - target) / target)
#     return torch.mean(differences)


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
    df = pd.read_excel(data_path)

    data_np = np.nan_to_num(df.apply(pd.to_numeric, errors='coerce').to_numpy())
    x = torch.tensor(normalize_data(data_np[0:len(data_np)-48, :]), dtype=torch.float32)

    edge_index = torch.tensor([[0,0,0,1,1,1,2,2,3,4,4,4,4,6,6,6,6,6,7,8,8,14,11,11],
                               [4,2,8,6,11,10,6,8,4,11,1,6,1,1,2,10,8,5,9,2,0,12,4,1]],
                              dtype=torch.long)


    edge_attr_pd = pd.read_excel('F:/AAA/GNN_Gaussian/data/ENTSO/edge_attr.xlsx')
    edge_attr_np = np.nan_to_num(edge_attr_pd.to_numpy(dtype=np.float32))
    edge_attr = torch.tensor(normalize_data(edge_attr_np[0:len(data_np)-48, :]), dtype=torch.float32)

    data = TensorDataset(x, edge_attr)
    loader = DataLoader(data, batch_size=168, shuffle=False)

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

        data_chunk = batch_data[:144]
        initial_alpha, initial_beta = Seasonal_DFA_alpha(data_chunk)

        gnn = GNNModel(node_features=x.size(1), edge_features=edge_attr.size(1))
        updated_x = gnn(data_chunk, batch_edge_attr[:144], edge_index)

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

        remaining_data = batch_data[144:]
        updated_x_remaining = gnn(remaining_data, batch_edge_attr[144:], edge_index)

        final_x_conv = gauss_conv(updated_x_remaining, saved_alpha, saved_beta)


        final_rmse = final_rmse_loss(final_x_conv, updated_x_remaining)
        final_mae= final_mae_loss(final_x_conv, updated_x_remaining)
        final_mape= final_mape_loss(final_x_conv, updated_x_remaining)


        for i in range(final_rmse.size(0)):
            print(f"Final Loss for Batch {batch_idx + 1}, Column {i+1}: RMSE: {final_rmse[i].item()}, MAE: {final_mae[i].item()}, MAPE: {final_mape[i].item()}")

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

    print(f"Total Final RMSE: {total_final_rmse.tolist()}, Total Final MSE: {total_final_mae.tolist()}, Total Final MAPE: {total_final_mape.tolist()}")

    # 保存最终的RMSE，best_alpha，best_beta到Excel文件
    df_results = pd.DataFrame({
        'Final RMSE': rmse_losses,
        'Final MAE': mae_losses,
        'Final MAPE': mape_losses,
        'Best Alpha': best_alpha_values,
        'Best Beta': best_beta_values
    })

    df_results.to_excel(f'F:/AAA/GNN_Gaussian/data/ENTSO/{data_path.split("/")[-1].split(".")[0]}_results_168.xlsx', index=False)

    return avg_final_rmse.tolist(),avg_final_mae.tolist(),avg_final_mape.tolist()

# 训练两个数据集并绘制对比图
data_path1 = 'F:/AAA/GNN_Gaussian/data/ENTSO/ENTSO_filled.xlsx'
data_path2 = 'F:/AAA/GNN_Gaussian/data/ENTSO/ENTSO_filled.xlsx'
# data_path2 = 'F:/AAA/GNN_Gaussian/data/ENTSO/STL/15000/final_data_15000.xlsx'


avg_rmse1,avg_mae1,avg_mape1 = train_model(data_path1)
avg_rmse2,avg_mae2,avg_mape2 = train_model(data_path2)

# 创建一个DataFrame来保存RMSE损失值
rmse_df = pd.DataFrame({
    'EIA24 RMSE': avg_rmse1,
    'EIA24 MAE': avg_mae1,
    'EIA24 MAPE': avg_mape1,
    'EIA24_STL RMSE': avg_rmse2,
    'EIA24_STL MAE': avg_mae2,
    'EIA24_STL MAPE': avg_mape2
})

# 将DataFrame写入到Excel文件中
rmse_df.to_excel('F:/AAA/GNN_Gaussian/data/ENTSO/total_loss_comparison_168.xlsx', index=False)

print("RMSE values have been saved to 'total_loss_comparison.xlsx'")

# 绘制对比图
plt.figure(figsize=(10, 6))
x = range(len(avg_rmse1))
width = 0.35  # 条形图的宽度
plt.bar(x, avg_rmse1, width, label='ENTSO', color='#B8D4E9')
# plt.bar([p + width for p in x], total_rmse2, width, label='EIA24_STL', color='#2F7DBB')
plt.bar([p + width + 0.05 for p in x], avg_rmse2, width, label='ENTSO_STL', color='#2F7DBB')
plt.xlabel('Column')
plt.ylabel('Total RMSE Loss')
plt.title('Total RMSE Loss per Column for Two Data Sets')
plt.legend()
# plt.xticks([p + width / 2 for p in x], x)
plt.xticks([p + width / 2 + 0.025 for p in x], x)
plt.tight_layout()
plt.savefig('F:/AAA/GNN_Gaussian/data/ENTSO/ENTSO_test_rmse_comparison_168.png')
plt.show()