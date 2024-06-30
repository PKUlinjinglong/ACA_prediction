from torch_geometric.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import random
from .compound_tools import *
from .data_process import Dataset_process
from .GNN import GINGraphPooling
from .config import parse_args

seed = 520
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)
calc = Calculator(descriptors, ignore_3D=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
atom_id_names = [
    "atomic_num", "chiral_tag", "degree", "explicit_valence",
    "formal_charge", "hybridization", "implicit_valence",
    "is_aromatic", "total_numHs",
]
bond_id_names = ["bond_dir", "bond_type", "is_in_ring"]
bond_float_names=["bond_length"]
bond_angle_float_names=['bond_angle', 'TPSA', 'RASA', 'RPSA', 'MDEC', 'MATS']
full_atom_feature_dims = get_atom_feature_dims(atom_id_names)
full_bond_feature_dims = get_bond_feature_dims(bond_id_names)

warnings.filterwarnings("ignore", category=FutureWarning)
def q_loss(q,y_true,y_pred):
    e = (y_true-y_pred)
    return torch.mean(torch.maximum(q*e, (q-1)*e))
class GNN_process():
    '''
    GNN的训练或测试等
    '''
    def __init__(self,config):
        self.GNN_mode = config.GNN_mode
        self.num_task = config.num_task
        self.num_layers = config.num_layers
        self.emb_dim = config.emb_dim
        self.drop_ratio = config.drop_ratio
        self.graph_pooling = config.graph_pooling
        self.weight_decay = config.weight_decay
        self.Train_model_path = config.Train_model_path
        self.num_iterations = config.num_iterations
        self.Test_model_path = config.Test_model_path
        self.gnn_train_ratio = config.gnn_train_ratio
        self.Outcome_graph_path = config.Outcome_graph_path
        self.predict_model_path = config.predict_model_path
        self.unknown_descriptor_path = config.unknown_descriptor_path

    def train_gnn(self,model, device, loader_atom_bond, loader_bond_angle, optimizer):
        model.train()
        loss_accum = 0
        for step, batch in enumerate(zip(loader_atom_bond, loader_bond_angle)):
            batch_atom_bond = batch[0]
            batch_bond_angle = batch[1]
            batch_atom_bond = batch_atom_bond.to(device)
            batch_bond_angle = batch_bond_angle.to(device)
            pred = model(batch_atom_bond, batch_bond_angle)[0]
            true = batch_atom_bond.y
            optimizer.zero_grad()
            loss = q_loss(0.1, true, pred[:, 0]) + torch.mean((true - pred[:, 1]) ** 2) + q_loss(0.9, true, pred[:, 2]) \
                   + torch.mean(torch.relu(pred[:, 0] - pred[:, 1])) + torch.mean(
                torch.relu(pred[:, 1] - pred[:, 2])) + torch.mean(torch.relu(2 - pred))
            loss.backward()
            optimizer.step()
            loss_accum += loss.detach().cpu().item()
        return loss_accum / (step + 1)
    def te_gnn(self,model, device, loader_atom_bond, loader_bond_angle):
        model.eval()
        y_pred = []
        y_true = []
        y_pred_10 = []
        y_pred_90 = []
        with torch.no_grad():
            for _, batch in enumerate(zip(loader_atom_bond, loader_bond_angle)):
                batch_atom_bond = batch[0]
                batch_bond_angle = batch[1]
                batch_atom_bond = batch_atom_bond.to(device)
                batch_bond_angle = batch_bond_angle.to(device)
                pred = model(batch_atom_bond, batch_bond_angle)[0]
                y_true.append(batch_atom_bond.y.detach().cpu().reshape(-1, ))
                y_pred.append(pred[:, 1].detach().cpu())
                y_pred_10.append(pred[:, 0].detach().cpu())
                y_pred_90.append(pred[:, 2].detach().cpu())
            y_true = torch.cat(y_true, dim=0)
            y_pred = torch.cat(y_pred, dim=0)
            y_pred_10 = torch.cat(y_pred_10, dim=0)
            y_pred_90 = torch.cat(y_pred_90, dim=0)
        R_square = 1 - (((y_true - y_pred) ** 2).sum() / ((y_true - y_pred.mean()) ** 2).sum())
        test_mae = torch.mean((y_true - y_pred) ** 2)
        print(R_square)
        return y_pred, y_true, R_square, test_mae, y_pred_10, y_pred_90
    def eval(self, model, device, loader_atom_bond, loader_bond_angle):
        model.eval()
        y_true = []
        y_pred = []
        y_pred_10 = []
        y_pred_90 = []

        with torch.no_grad():
            for _, batch in enumerate(zip(loader_atom_bond, loader_bond_angle)):
                batch_atom_bond = batch[0]
                batch_bond_angle = batch[1]
                batch_atom_bond = batch_atom_bond.to(device)
                batch_bond_angle = batch_bond_angle.to(device)
                pred = model(batch_atom_bond, batch_bond_angle)[0]
                y_true.append(batch_atom_bond.y.detach().cpu().reshape(-1))
                y_pred.append(pred[:, 1].detach().cpu())
                y_pred_10.append(pred[:, 0].detach().cpu())
                y_pred_90.append(pred[:, 2].detach().cpu())
        y_true = torch.cat(y_true, dim=0)
        y_pred = torch.cat(y_pred, dim=0)
        y_pred_10 = torch.cat(y_pred_10, dim=0)
        y_pred_90 = torch.cat(y_pred_90, dim=0)
        input_dict = {"y_true": y_true, "y_pred": y_pred}
        return torch.mean((y_true - y_pred) ** 2).data.numpy()
    def Mode(self):
        if self.GNN_mode == 'Train':
            nn_params = {
                'num_tasks': self.num_task,
                'num_layers': self.num_layers,
                'emb_dim': self.emb_dim,
                'drop_ratio': self.drop_ratio,
                'graph_pooling': self.graph_pooling,
                'descriptor_dim': 1827
            }
            config = parse_args()
            data = Dataset_process(config)
            train_loader_atom_bond, valid_loader_atom_bond, test_loader_atom_bond, train_loader_bond_angle, valid_loader_bond_angle, test_loader_bond_angle = data.make_gnn_dataset()
            criterion_fn = torch.nn.MSELoss()
            model = GINGraphPooling(**nn_params).to(device)
            num_params = sum(p.numel() for p in model.parameters())
            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=self.weight_decay)
            scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
            folder_path = self.Train_model_path
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            lowest_mae = float('inf')
            lowest_mae_filename = None
            for epoch in tqdm(range(self.num_iterations)):
                train_mae = self.train_gnn(model, device, train_loader_atom_bond, train_loader_bond_angle, optimizer)
                valid_outcome = []
                if (epoch + 1) % 100 == 0:
                    valid_mae = self.eval(model, device, valid_loader_atom_bond, valid_loader_bond_angle)
                    print(train_mae, valid_mae)
                    valid_outcome.append(valid_mae)
                    current_filename = f'/model_save_{epoch + 1}.pth'
                    torch.save(model.state_dict(), folder_path + current_filename)
                    if valid_mae < lowest_mae:
                        lowest_mae = valid_mae
                        lowest_mae_filename = current_filename
            print(f"The .pth file with the lowest validation MAE ({lowest_mae}) is: {lowest_mae_filename}")

        if self.GNN_mode == 'Test':
            nn_params = {
                'num_tasks': self.num_task,
                'num_layers': self.num_layers,
                'emb_dim': self.emb_dim,
                'drop_ratio': self.drop_ratio,
                'graph_pooling': self.graph_pooling,
                'descriptor_dim': 1827
            }
            model = GINGraphPooling(**nn_params).to(device)
            model.load_state_dict(torch.load(self.Train_model_path+self.Test_model_path))
            config = parse_args()
            data = Dataset_process(config)
            train_loader_atom_bond, valid_loader_atom_bond, test_loader_atom_bond, train_loader_bond_angle, valid_loader_bond_angle, test_loader_bond_angle = data.make_gnn_dataset()
            y_pred, y_true, R_square, test_mae, y_pred_10, y_pred_90 = self.te_gnn(model, device, test_loader_atom_bond,test_loader_bond_angle)
            y_pred = y_pred.cpu().data.numpy()
            y_true = y_true.cpu().data.numpy()
            y_pred_10 = y_pred_10.cpu().data.numpy()
            y_pred_90 = y_pred_90.cpu().data.numpy()
            print('relative_error', np.sqrt(np.sum((y_true - y_pred) ** 2) / np.sum(y_true ** 2)))
            print('MAE', np.mean(np.abs(y_true - y_pred) / y_true))
            print('RMSE', np.sqrt(np.mean((y_true - y_pred) ** 2)))
            R_square = 1 - (((y_true - y_pred) ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum())
            print(R_square)
            plt.figure(1, figsize=(3.5, 3.5), dpi=300)
            plt.style.use('ggplot')
            plt.scatter(y_true, y_pred, c='#8983BF', s=15, alpha=0.4)
            plt.plot(np.arange(0, 14), np.arange(0, 14), linewidth=.5, linestyle='--', color='black')
            plt.yticks(np.arange(0, 14, 2), np.arange(0, 14, 2), fontproperties='Arial', size=8)
            plt.xticks(np.arange(0, 14, 2), np.arange(0, 14, 2), fontproperties='Arial', size=8)
            plt.xlabel('Observed data', fontproperties='Arial', size=8)
            plt.ylabel('Predicted data', fontproperties='Arial', size=8)
            plt.text(4, 11, "r2 = {:.5f}".format(R_square), fontsize=10, ha='center')
            folder_path = self.Outcome_graph_path
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            if self.gnn_train_ratio == 0.9:
                plt.title('GNN_18_1_1')
                plt.savefig(folder_path + "/GNN_18_1_1.svg")
            if self.gnn_train_ratio == 0.8:
                plt.title('GNN_8_1_1')
                plt.savefig(folder_path + "/GNN_8_1_1.svg")
            plt.show()

        if self.GNN_mode == 'Pre':
            nn_params = {
                'num_tasks': self.num_task,
                'num_layers': self.num_layers,
                'emb_dim': self.emb_dim,
                'drop_ratio': self.drop_ratio,
                'graph_pooling': self.graph_pooling,
                'descriptor_dim': 1827
            }
            model = GINGraphPooling(**nn_params).to(device)
            config = parse_args()
            data = Dataset_process(config)
            pre_loader_atom_bond, pre_loader_bond_angle = data.make_gnn_prediction_dataset()
            model.load_state_dict(torch.load(self.Train_model_path+self.predict_model_path))
            y_pred, y_true, R_square, test_mae, y_pred_10, y_pred_90 = self.te_gnn(model, device, pre_loader_atom_bond,pre_loader_bond_angle)
            y_pred = y_pred.cpu().data.numpy()
            y_true = y_true.cpu().data.numpy()
            y_pred_10 = y_pred_10.cpu().data.numpy()
            y_pred_90 = y_pred_90.cpu().data.numpy()
            ACA = pd.read_excel(self.unknown_descriptor_path)
            ACA['Pre_RT_GNN'] = y_pred
            ACA['10th percentile'] = y_pred_10
            ACA['90th percentile'] = y_pred_90
            ACA.to_excel(self.Outcome_graph_path+"/output_prediction.xlsx", index=False)
            return y_pred