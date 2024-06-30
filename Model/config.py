import argparse
import torch

def parse_args():
    parser = argparse.ArgumentParser()

    # --------------parameters for path-----------------------
    parser.add_argument('--Whether_predict', type=bool, default=True)
    parser.add_argument('--shuffle_seed', type=int, default=520)
    parser.add_argument('--known_data_path', type=str, default='./Data/Known.xlsx')
    parser.add_argument('--unknown_data_path', type=str, default='./Data/Predict.xlsx')
    parser.add_argument('--known_descriptor_path', type=str, default='./Data/Known_Descriptor.xlsx')
    parser.add_argument('--unknown_descriptor_path', type=str, default='./Data/Predict_Descriptor.xlsx')

    # --------------parameters for ML train and test-----------------------
    parser.add_argument('--ML_test_ratio', type=float, default=0.1)
    parser.add_argument('--ML_features', nargs='+', type=str, default=['TPSA', 'HBA', 'HBD', 'NROTB', 'MW', 'LogP'])
    parser.add_argument('--ML_label', nargs='+', type=str, default=['RT'])

    # --------------parameters for GNN-----------------------
    parser.add_argument('--GNN_mode', type=str, default='Pre')
    parser.add_argument('--gnn_train_ratio', type=float, default=0.8)
    parser.add_argument('--gnn_valid_ratio', type=float, default=0.1)
    parser.add_argument('--gnn_test_ratio', type=float, default=0.1)
    parser.add_argument('--known_3D_path', type=str, default='./Data/known_3D_mol_1012')
    parser.add_argument('--unknown_3D_path', type=str, default='./Data/unknown_3D_mol_1012')
    parser.add_argument('--Train_model_path', type=str, default='./Outcome/GNN_1012_811')
    parser.add_argument('--Test_model_path', type=str, default='/model_save_300.pth')
    parser.add_argument('--predict_model_path', type=str, default='/model_save_300.pth')
    parser.add_argument('--Outcome_graph_path', type=str, default='./Outcome/GNN_1012_811/Test_Graph')

    parser.add_argument('--task_name', type=str, default='GINGraphPooling')
    parser.add_argument('--num_task', type=int, default=3)
    parser.add_argument('--num_iterations', type=int, default=1500)
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--graph_pooling', type=str, default='sum')
    parser.add_argument('--emb_dim', type=int, default=128)
    parser.add_argument('--drop_ratio', type=float, default=0.)
    parser.add_argument('--save_test', action='store_true')
    parser.add_argument('--gnn_batch_size', type=int, default=2048)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--weight_decay', type=float, default=0.00001)
    parser.add_argument('--early_stop', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--dataset_root', type=str, default="dataset")

    args = parser.parse_args()
    config = parser.parse_args()
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return config
