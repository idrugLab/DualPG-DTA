import os
import sys
import torch
import getopt
import numpy as np
from models import DTANet
from rdkit import Chem
from unimol_tools import UniMolRepr
from rdkit.Chem.rdchem import BondType
from torch_geometric.data import Data
from Bio.PDB.PDBParser import PDBParser
import pandas as pd

bond_type = [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC]

ESMModel, ESMAlphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t36_3B_UR50D")
ESMBatchConverter = ESMAlphabet.get_batch_converter()
ESMModel.eval()

ESMFoldModel = torch.hub.load("facebookresearch/esm:main", "esmfold_v1")
ESMFoldModel = ESMFoldModel.to(torch.float32)
ESMFoldModel.eval()


def test(drug_graph, target_graph):
    model = DTANet()
    model.eval()
    assert len(os.listdir("./pts")) == 5
    results = []
    for i in range(0, 5):
        checkpoint_path = f"./pts/fold_{i}.pt"
        with open(checkpoint_path, "rb") as f:
            model.load_state_dict(torch.load(f, map_location=torch.device("cpu")))
        with torch.no_grad():
            output = model(drug_graph, target_graph).view(-1)
        results.append(float(output.item()))
    return results


def smi2graph(smi):

    def one_of_k_encoding(x, allowable_set):
        if x not in allowable_set:
            raise ValueError(
                "input {0} not in allowable set{1}:".format(x, allowable_set)
            )
        return list(map(lambda s: x == s, allowable_set))

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smi}")
    graph = Data()
    bonds = mol.GetBonds()
    clf = UniMolRepr(data_type="molecule", remove_hs=False)
    unimol_repr = clf.get_repr([smi], return_atomic_reprs=True)
    graph.x = torch.Tensor(unimol_repr["atomic_reprs"][0])
    edge_index = []
    edge_attr = []
    for bond in bonds:
        edge_index.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        edge_index.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])
        edge_attr.append(
            one_of_k_encoding(bond.GetBondType(), bond_type)
            + [bond.GetIsAromatic(), bond.GetIsConjugated(), bond.IsInRing()]
        )
        edge_attr.append(
            one_of_k_encoding(bond.GetBondType(), bond_type)
            + [bond.GetIsAromatic(), bond.GetIsConjugated(), bond.IsInRing()]
        )
    graph.edge_index = torch.LongTensor(np.transpose(edge_index))
    graph.edge_attr = torch.Tensor(edge_attr)
    return graph


def run_offline(drug_smiles, target_sequence):
    with torch.no_grad():
        pdb = ESMFoldModel.infer_pdb(target_sequence)
    with open("local.pdb", "w") as f:
        f.write(pdb)
    parser = PDBParser()
    with open("local.pdb", "r") as f:
        structure = parser.get_structure("protein", f)
    residues = [r for r in structure.get_residues() if r.get_id()[0] == " "]
    edge_index = []
    edge_weight = []
    DISTANCE_CUTOFF = 8.0
    for i in range(len(residues)):
        for j in range(len(residues)):
            if i != j:
                distance = np.linalg.norm(
                    residues[i]["CA"].get_coord() - residues[j]["CA"].get_coord()
                )
                if distance < DISTANCE_CUTOFF:
                    edge_index.append([i, j])
                    edge_weight.append(distance)

    edge_weight = np.asarray(edge_weight)
    edge_weight = (edge_weight.max() - edge_weight) / (np.ptp(edge_weight))
    target_graph = Data()
    target_graph.edge_index, target_graph.edge_attr = torch.LongTensor(
        np.transpose(edge_index)
    ), torch.Tensor(edge_weight).unsqueeze(1)
    assert target_graph.edge_index.shape == (2, len(target_graph.edge_attr))

    batch_labels, batch_strs, batch_tokens = ESMBatchConverter(
        [("protein", target_sequence)]
    )
    with torch.no_grad():
        results = ESMModel(batch_tokens, repr_layers=[36], return_contacts=False)
    target_graph.x = torch.Tensor(
        results["representations"][36][0][1 : len(target_sequence) + 1]
    )
    assert target_graph.x.shape[1] == 2560

    output = test(smi2graph(drug_smiles), target_graph)
    return np.array(output).mean()


if __name__ == "__main__":
    if len(sys.argv) == 3:
        try:
            opts, args = getopt.getopt(sys.argv[1:], "i:", ["input="])
        except getopt.GetoptError:
            print("ERROR! usage: python run.py --input <input_csv_filename>")
            sys.exit(2)
        for opt, arg in opts:
            if opt in ("-i", "--input"):
                filename = arg
        if not "filename" in locals():
            print("ERROR! usage: python run.py --input <input_csv_filename>")
            sys.exit(2)
        with open(filename, "r") as f:
            df = pd.read_csv(f)
        assert df.columns.tolist() == ["drug_smiles", "target_sequence"]
        df["result"] = df.apply(
            lambda x: format(
                run_offline(x["drug_smiles"], x["target_sequence"]), "0.3f"
            ),
            axis=1,
        )
        df.to_csv("out_" + filename, index=False)
    elif len(sys.argv) == 5:
        try:
            opts, args = getopt.getopt(sys.argv[1:], "d:t:", ["drug=", "target="])
        except getopt.GetoptError:
            print(
                "ERROR! usage: python run.py --drug <drug_smiles> --target <target_sequence>"
            )
            sys.exit(2)
        for opt, arg in opts:
            if opt in ("-d", "--drug"):
                drug = arg
            elif opt in ("-t", "--target"):
                target = arg
        if not set(["drug", "target"]).issubset(locals()):
            print(
                "ERROR! usage: python run.py --drug <drug_smiles> --target <target_sequence>"
            )
            sys.exit(2)
        run_offline(drug, target)
    else:
        print("ERROR! usage: python run.py --input <input_csv_filename>")
        print(
            "ERROR! usage: python run.py --drug <drug_smiles> --target <target_sequence>"
        )
        sys.exit(2)
