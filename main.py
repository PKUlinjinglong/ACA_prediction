from Model.data_process import Dataset_process,smiles_to_excel
from Model.config import parse_args
from Model.model import GNN_process
import warnings
warnings.filterwarnings("ignore")

def main():
    config = parse_args()
    data = Dataset_process(config)
    know_des,unkonwn_des = data.Get_data_file()

    GNN = GNN_process(config)
    GNN.GNN_mode = 'Pre'    #Train/Test/Pre
    """Train"""
    # data.Whether_predict = False
    # data.prepare_gnn_3D()
    # GNN.num_iterations = 1500
    # GNN.Mode()
    """Test"""
    # GNN.predict_model_path = '/model_save_300.pth'
    # GNN.Mode()
    """Pre"""
    GNN.predict_model_path = '/model_save_300.pth'
    predict_ACA_smi = ['O=C(N[C@@H](CC(C)C)C(O)=O)C1=CC(O)=CC=C1', 'O=C(N[C@@H](C)C(O)=O)CC1=C(O)C=CC=C1','O=C(N[C@@H](CC1=CNC2=C1C=CC=C2)C(O)=O)/C=C/C3=CC=C(O)C=C3']    # 输入想要预测的ACA的Smiles
    smiles_to_excel(predict_ACA_smi)
    data.Get_data_file()
    data.Whether_predict = True
    data.prepare_gnn_3D()
    pred = GNN.Mode()
    print(pred)

if __name__ == "__main__":
    main()