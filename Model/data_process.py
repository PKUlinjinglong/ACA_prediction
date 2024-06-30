from torch_geometric.data import DataLoader
from torch_geometric.data import Data
from rdkit.Chem import MACCSkeys
from rdkit.Chem import AllChem, PandasTools, Descriptors
from .compound_tools import *
import os
from mordred import Calculator, descriptors,is_missing
import random
from rdkit import Chem, DataStructs

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


def smiles_to_excel(smiles_list):
    df = pd.DataFrame({
        'Compound': range(len(smiles_list)),
        'Smiles': smiles_list
    })
    df.to_excel(r'./Data/Predict.xlsx', index=False)

class Dataset_process():
    def __init__(self,config):
        self.Whether_predict = config.Whether_predict
        self.shuffle_seed = config.shuffle_seed
        self.known_data_path = config.known_data_path
        self.unknown_data_path = config.unknown_data_path
        self.known_descriptor_path = config.known_descriptor_path
        self.unknown_descriptor_path = config.unknown_descriptor_path
        self.ML_test_ratio = config.ML_test_ratio
        self.ML_features = config.ML_features
        self.ML_label = config.ML_label
        self.known_3D_path = config.known_3D_path
        self.unknown_3D_path = config.unknown_3D_path
        self.gnn_train_ratio = config.gnn_train_ratio
        self.gnn_valid_ratio = config.gnn_valid_ratio
        self.gnn_test_ratio = config.gnn_test_ratio
        self.gnn_batch_size = config.gnn_batch_size
        self.num_workers = config.num_workers

    def process_dataframe(self,df):
        random_seed = 520
        smiles_list = df['Smiles'].tolist()
        tpsa, hba, hbd, nrotb, mw, logp = [], [], [], [], [], []
        maccs_columns = [f"MACCS_bit_{i}" for i in range(166)]
        for column in maccs_columns:
            df[column] = 0
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                tpsa.append(Descriptors.TPSA(mol))
                hba.append(Descriptors.NumHAcceptors(mol))
                hbd.append(Descriptors.NumHDonors(mol))
                nrotb.append(Descriptors.NumRotatableBonds(mol))
                mw.append(Descriptors.MolWt(mol))
                logp.append(Descriptors.MolLogP(mol))
                maccs_key = MACCSkeys.GenMACCSKeys(mol)
                for i, bit in enumerate(maccs_key):
                    df.at[smiles_list.index(smiles), f"MACCS_bit_{i}"] = int(bit)
            else:
                tpsa.append(None)
                hba.append(None)
                hbd.append(None)
                nrotb.append(None)
                mw.append(None)
                logp.append(None)
        df['TPSA'] = tpsa
        df['HBA'] = hba
        df['HBD'] = hbd
        df['NROTB'] = nrotb
        df['MW'] = mw
        df['LogP'] = logp
        return df
    def Get_data_file(self):
        known = pd.read_excel(self.known_data_path)
        random_seed = self.shuffle_seed
        known.insert(0, 'Index', known.index)
        known.sample(frac=1, random_state=random_seed)
        known_descriptor = self.process_dataframe(known)
        known_descriptor.to_excel(self.known_descriptor_path, index=False)
        if self.Whether_predict == True:
            unknown = pd.read_excel(self.unknown_data_path)
            unknown.insert(0, 'Index', unknown.index)
            unknown_descriptor = self.process_dataframe(unknown)
            unknown_descriptor.to_excel(self.unknown_descriptor_path, index=False)
            return known_descriptor, unknown_descriptor
        else:
            return known_descriptor
    def make_ml_dataset(self):
        data = pd.read_excel(self.known_descriptor_path)
        feature_columns = self.ML_features
        label_columns = self.ML_label
        X = data[feature_columns]
        y = data[label_columns]
        test_ratio = self.ML_test_ratio

        test_size = int(len(data) * test_ratio)
        train_size = len(data) - test_size
        x_train = X.iloc[:train_size]
        y_train = y.iloc[:train_size]
        x_test = X.iloc[train_size:]
        y_test = y.iloc[train_size:]
        if self.Whether_predict == True:
            data_search = pd.read_excel(self.unknown_descriptor_path)
            x_search = data_search[feature_columns]
            return x_train,y_train,x_test,y_test,x_search
        else:
            return x_train, y_train, x_test, y_test

    def prepare_gnn_3D(self):
        if self.Whether_predict == True:
            df_unknown = pd.read_excel(self.unknown_descriptor_path)
            smiles_unknown = df_unknown['Smiles'].values
            unknown_3D_path = self.unknown_3D_path
            if not os.path.exists(unknown_3D_path):
                os.makedirs(unknown_3D_path)
            bad_Conformation_unknown = save_3D_mol(smiles_unknown, unknown_3D_path)
            np.save(unknown_3D_path + '/bad_Conformation.npy', np.array(bad_Conformation_unknown))
            save_dataset(smiles_unknown, unknown_3D_path, unknown_3D_path + '/Graph_dataset', unknown_3D_path + '/Descriptors', bad_Conformation_unknown)
        else:
            df_known = pd.read_excel(self.known_descriptor_path)
            smiles = df_known['Smiles'].values
            known_3D_path = self.known_3D_path
            if not os.path.exists(known_3D_path):
                os.makedirs(known_3D_path)
            bad_Conformation = save_3D_mol(smiles, known_3D_path)
            np.save(known_3D_path + '/bad_Conformation.npy', np.array(bad_Conformation))
            save_dataset(smiles, known_3D_path, known_3D_path + '/Graph_dataset', known_3D_path + '/Descriptors',
                         bad_Conformation)
    def Construct_dataset(self, dataset, data_index, rt, route):
        graph_atom_bond = []
        graph_bond_angle = []
        big_index = []
        all_descriptor = np.load(route + '/Descriptors.npy')
        for i in range(len(dataset)):
            data = dataset[i]
            atom_feature = []
            bond_feature = []
            for name in atom_id_names:
                atom_feature.append(data[name])
            for name in bond_id_names:
                bond_feature.append(data[name])
            atom_feature = torch.from_numpy(np.array(atom_feature).T).to(torch.int64)
            bond_feature = torch.from_numpy(np.array(bond_feature).T).to(torch.int64)
            bond_float_feature = torch.from_numpy(data['bond_length'].astype(np.float32))
            bond_angle_feature = torch.from_numpy(data['bond_angle'].astype(np.float32))
            y = torch.Tensor([float(rt[i])])
            edge_index = torch.from_numpy(data['edges'].T).to(torch.int64)
            bond_index = torch.from_numpy(data['BondAngleGraph_edges'].T).to(torch.int64)
            data_index_int = torch.from_numpy(np.array(data_index[i])).to(torch.int64)
            TPSA = torch.ones([bond_angle_feature.shape[0]]) * all_descriptor[i, 820] / 100
            RASA = torch.ones([bond_angle_feature.shape[0]]) * all_descriptor[i, 821]
            RPSA = torch.ones([bond_angle_feature.shape[0]]) * all_descriptor[i, 822]
            MDEC = torch.ones([bond_angle_feature.shape[0]]) * all_descriptor[i, 1568]
            MATS = torch.ones([bond_angle_feature.shape[0]]) * all_descriptor[i, 457]

            bond_feature = torch.cat([bond_feature, bond_float_feature.reshape(-1, 1)], dim=1)

            bond_angle_feature = bond_angle_feature.reshape(-1, 1)
            bond_angle_feature = torch.cat([bond_angle_feature.reshape(-1, 1), TPSA.reshape(-1, 1)], dim=1)
            bond_angle_feature = torch.cat([bond_angle_feature, RASA.reshape(-1, 1)], dim=1)
            bond_angle_feature = torch.cat([bond_angle_feature, RPSA.reshape(-1, 1)], dim=1)
            bond_angle_feature = torch.cat([bond_angle_feature, MDEC.reshape(-1, 1)], dim=1)
            bond_angle_feature = torch.cat([bond_angle_feature, MATS.reshape(-1, 1)], dim=1)

            if y[0] > 60:
                big_index.append(i)
                continue
            data_atom_bond = Data(atom_feature, edge_index, bond_feature, y, data_index=data_index_int)
            data_bond_angle = Data(edge_index=bond_index, edge_attr=bond_angle_feature)
            graph_atom_bond.append(data_atom_bond)
            graph_bond_angle.append(data_bond_angle)
        return graph_atom_bond, graph_bond_angle, big_index

    def make_gnn_dataset(self):
        ACA = pd.read_excel(self.known_descriptor_path)
        bad_index = np.load(self.known_3D_path+'/bad_Conformation.npy')
        ACA = ACA.drop(bad_index)
        smiles = ACA['Smiles'].values
        y = ACA[self.ML_label].values
        graph_dataset = np.load(self.known_3D_path+'/Graph_dataset.npy', allow_pickle=True).tolist()
        index_aca = ACA['Index'].values
        dataset_graph_atom_bond, dataset_graph_bond_angle, big_index = self.Construct_dataset(graph_dataset, index_aca, y, self.known_3D_path)  #生成两张图：atom-bond和 bond-angle图
        total_num = len(dataset_graph_atom_bond)
        print('Known data num:', total_num)
        train_ratio = self.gnn_train_ratio
        validate_ratio = self.gnn_valid_ratio
        test_ratio = self.gnn_test_ratio
        data_array = np.arange(0, total_num, 1)
        np.random.shuffle(data_array)
        torch.random.manual_seed(520)
        train_data_atom_bond = []
        valid_data_atom_bond = []
        test_data_atom_bond = []
        train_data_bond_angle = []
        valid_data_bond_angle = []
        test_data_bond_angle = []
        train_num = int(len(data_array) * train_ratio)
        test_num = int(len(data_array) * test_ratio)
        val_num = int(len(data_array) * validate_ratio)
        train_index = data_array[0:train_num]
        valid_index = data_array[train_num:train_num + val_num]
        test_index = data_array[total_num - test_num:]
        for i in test_index:
            test_data_atom_bond.append(dataset_graph_atom_bond[i])
            test_data_bond_angle.append(dataset_graph_bond_angle[i])
        for i in valid_index:
            valid_data_atom_bond.append(dataset_graph_atom_bond[i])
            valid_data_bond_angle.append(dataset_graph_bond_angle[i])
        for i in train_index:
            train_data_atom_bond.append(dataset_graph_atom_bond[i])
            train_data_bond_angle.append(dataset_graph_bond_angle[i])
        train_loader_atom_bond = DataLoader(train_data_atom_bond, batch_size=self.gnn_batch_size, shuffle=False,
                                            num_workers=self.num_workers)
        valid_loader_atom_bond = DataLoader(valid_data_atom_bond, batch_size=self.gnn_batch_size, shuffle=False,
                                            num_workers=self.num_workers)
        test_loader_atom_bond = DataLoader(test_data_atom_bond, batch_size=self.gnn_batch_size, shuffle=False,
                                           num_workers=self.num_workers)
        train_loader_bond_angle = DataLoader(train_data_bond_angle, batch_size=self.gnn_batch_size, shuffle=False,
                                             num_workers=self.num_workers)
        valid_loader_bond_angle = DataLoader(valid_data_bond_angle, batch_size=self.gnn_batch_size, shuffle=False,
                                             num_workers=self.num_workers)
        test_loader_bond_angle = DataLoader(test_data_bond_angle, batch_size=self.gnn_batch_size, shuffle=False,
                                            num_workers=self.num_workers)
        return train_loader_atom_bond,valid_loader_atom_bond,test_loader_atom_bond,train_loader_bond_angle,valid_loader_bond_angle,test_loader_bond_angle

    def make_gnn_prediction_dataset(self):
        ACA = pd.read_excel(self.unknown_descriptor_path)
        bad_index = np.load(self.unknown_3D_path+'/bad_Conformation.npy')
        ACA = ACA.drop(bad_index)
        smiles = ACA['Smiles'].values
        y = [1 for i in range(len(smiles))]
        graph_dataset = np.load(self.unknown_3D_path+'/Graph_dataset.npy',allow_pickle=True).tolist()
        index_aca = ACA.index
        dataset_graph_atom_bond, dataset_graph_bond_angle, big_index = self.Construct_dataset(graph_dataset, index_aca, y, self.unknown_3D_path)
        pre_loader_atom_bond = DataLoader(dataset_graph_atom_bond, batch_size=self.gnn_batch_size, shuffle=False,num_workers=self.num_workers)
        pre_loader_bond_angle = DataLoader(dataset_graph_bond_angle, batch_size=self.gnn_batch_size, shuffle=False,num_workers=self.num_workers)
        return pre_loader_atom_bond,pre_loader_bond_angle