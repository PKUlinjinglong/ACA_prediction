
# ACA Predict Model README

This is a Graph Neural Network model designed for predicting the retention times of amino acid conjugates.

## Environment Setup

Required environment includes:
- **PyTorch Version**: 2.3.0+cu121
- **torch_geometric Version**: 2.5.3

## Library Installation

Use the following commands to install the required libraries:

```bash
pip install torch==2.3.0+cu121 -f https://download.pytorch.org/whl/cu121/torch_stable.html
pip install torch_geometric==2.5.3 -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
```

## Running the Code

1. Set configuration parameters in the `./Model/config.py` file.

2. Run `main.py` in the root directory of the project:

```bash
python main.py
```

## Usage Instructions

1. Input the SMILES strings of the amino acid conjugates you are interested in into the `predict_ACA_smi` list (these can be exported from ChemDraw):

    ```python
    predict_mole_smi = ['O=C(N[C@@H](CC(C)C)C(O)=O)C1=CC(O)=CC=C1', 'O=C(N[C@@H](C)C(O)=O)CC1=C(O)C=CC=C1']
    ```

2. Run `main.py` to predict the retention time of the amino acid conjugates:

    ```bash
    python main.py
    ```

4. Modify the `GNN.GNN_mode` parameter to perform training and testing:

    - **Train Mode**:

        ```python
        GNN.GNN_mode = 'Train'
        data.Whether_predict = False
        data.prepare_gnn_3D()
        GNN.num_iterations = 1500
        GNN.Mode()
        ```

    - **Test Mode**:

        ```python
        GNN.GNN_mode = 'Test'
        GNN.Test_model_path = '/model_save_300.pth'
        GNN.Mode()
        ```

    - **Pre Mode**:

        ```python
        GNN.GNN_mode = 'Pre'
        GNN.predict_model_path = '/model_save_300.pth'
        predict_ACA_smi = ['O=C(N[C@@H](CC(C)C)C(O)=O)C1=CC(O)=CC=C1', 'O=C(N[C@@H](C)C(O)=O)CC1=C(O)C=CC=C1']    
        smiles_to_excel(predict_ACA_smi)
        data.Get_data_file()
        data.Whether_predict = True
        data.prepare_gnn_3D()
        pred = GNN.Mode()
        print(pred)
        ```

## Notes

- Before running the code, ensure all configuration parameters are correctly set in the `./Model/config.py` file.
- Ensure that the installed versions of PyTorch and torch_geometric meet the requirements to avoid compatibility issues.

By following these steps, you can use this GNN to predict the retention time of amino acid conjugates.
