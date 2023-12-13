import pandas as pd
import matplotlib.pyplot as plt

# Load the Excel file
data = pd.read_excel("./LXFL Results.xlsx")

# Display the first few rows to understand the structure
print(data.head())

# Extract columns related to CUB dataset
cub_columns = [col for col in data.columns if 'CUB' in col]
cub_data = data[cub_columns]

cub_data.head()

# Filter columns where noise type is "client noise"
client_noise_cols = cub_data.columns[cub_data.loc[4] == "client noise"].tolist()

# Extract data for the configurations with "client noise"
client_noise_data = cub_data[client_noise_cols]


# Extract relevant rows
noise_degree = (client_noise_data.iloc[5].values)*100
model_acc = client_noise_data.iloc[14].values
explanation_acc = client_noise_data.iloc[15].values
explanation_fidelity = client_noise_data.iloc[16].values
methods = client_noise_data.iloc[11].values

# Separate results for FedAvg and LR-XFL
fedavg_indices = [i for i, method in enumerate(methods) if method == "avg"]
lrxfl_indices = [i for i, method in enumerate(methods) if method == "weighted"]
DT_indices = [i for i, method in enumerate(methods) if method == "DT"]

# Adjust the plot with recommended colors and larger markers

recommended_colors = {
    'FedAvg': '#1f77b4',
    'LR-XFL': '#ff7f0e',
    'DT': '#2ca02c'
}

# Increase marker size
marker_size = 7
markers = {
    # 'ModelAcc': 's',
    # 'RuleAcc': 'o',
    # 'RuleFid': '^',
    'FedAvg': 's',
    'LR-XFL': 'o',
    'DT': '^'
}

line_styles = {
    # 'ModelAcc': '-',
    # 'RuleAcc': '--',
    # 'RuleFid': '-.'
    'FedAvg': '-',
    'LR-XFL': '--',
    'DT': '-.'
}

line_width = {
    # 'ModelAcc': 2,
    # 'RuleAcc': 2,
    # 'RuleFid': 2,
    'FedAvg': 2,
    'LR-XFL': 2,
    'DT': 2
}

y_label = ['Model Accuracy (%)', 'Rule Accuracy (%)', 'Rule Fidelity (%)']

titles = ['Model Accuracy', 'Rule Accuracy', 'Rule Fidelity']
data_sets = [
    (model_acc, 'ModelAcc'),
    (explanation_acc, 'RuleAcc'),
    (explanation_fidelity, 'RuleFid')
]
file_names = ['./noise_model_accuracy', './noise_rule_accuracy', './noise_rule_fidelity']

for idx, (data_set, data_key) in enumerate(data_sets):
    if idx != 2:
        plt.figure(figsize=(8, 6))

        plt.plot(noise_degree[DT_indices], data_set[DT_indices], linestyle=line_styles['DT'],
                 color=recommended_colors['DT'], marker=markers['DT'], markersize=marker_size,
                 linewidth=line_width['DT'], label=f'DDT')
        plt.plot(noise_degree[fedavg_indices], data_set[fedavg_indices], linestyle=line_styles['FedAvg'],
                 color=recommended_colors['FedAvg'], marker=markers['FedAvg'], markersize=marker_size,
                 linewidth=line_width['FedAvg'], label=f'FedAvg-Logic')
        plt.plot(noise_degree[lrxfl_indices], data_set[lrxfl_indices], linestyle=line_styles['LR-XFL'],
                 color=recommended_colors['LR-XFL'], marker=markers['LR-XFL'], markersize=marker_size,
                 linewidth=line_width['LR-XFL'], label=f'LR-XFL')


        plt.xlabel('Noise Degree (%)', fontsize=18)
        plt.ylabel(y_label[idx], fontsize=18)
        plt.legend(loc='lower left', fontsize=17)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.ylim(bottom=60)
        plt.tick_params(axis='both', which='major', labelsize=16)
        # plt.title(titles[idx], fontsize=15)

        plt.tight_layout()

        plt.savefig(f"{file_names[idx]}.jpg")
        plt.savefig(f"{file_names[idx]}.pdf")

        plt.show()


