import torch
import numpy as np
import pandas as pd
from rdkit import Chem
from torch_geometric.data import Data
from rdkit.Chem.rdchem import BondType
import json
import pickle
import os
import requests
from Bio.PDB.PDBParser import PDBParser
from unimol_tools import UniMolRepr
import esm

ESMModel, ESMAlphabet = esm.pretrained.esm2_t36_3B_UR50D()
ESMBatchConverter = ESMAlphabet.get_batch_converter()
ESMModel.eval()

UniMolClf = UniMolRepr(data_type="molecule", remove_hs=False)

BOND_TYPE = [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC]


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise ValueError("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def smis2graphs(smis):
    if os.access(os.path.join("smi2repr.pkl"), os.R_OK):
        smi2repr = pickle.load(open(os.path.join("smi2repr.pkl"), "rb"))
    graphs = {}
    for smi in smis:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smi}")

        mol = Chem.AddHs(mol)
        graph = Data()
        atoms = mol.GetAtoms()
        bonds = mol.GetBonds()

        if "smi2repr" in locals() and smi in smi2repr:
            graph.x = torch.Tensor(smi2repr[smi]["atomic_reprs"][0])
        else:
            unimol_repr = UniMolClf.get_repr([smi], return_atomic_reprs=True)
            graph.x = torch.Tensor(unimol_repr["atomic_reprs"][0])
        if graph.x.shape != (len(atoms), 512):
            print(smi, graph.x.shape, len(atoms))
            graphs[smi] = None
            continue

        edge_index = []
        edge_attr = []
        for bond in bonds:
            edge_index.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
            edge_index.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])
            edge_attr.append(
                one_of_k_encoding(bond.GetBondType(), BOND_TYPE)
                + [bond.GetIsAromatic(), bond.GetIsConjugated(), bond.IsInRing()]
            )
            edge_attr.append(
                one_of_k_encoding(bond.GetBondType(), BOND_TYPE)
                + [bond.GetIsAromatic(), bond.GetIsConjugated(), bond.IsInRing()]
            )
        graph.edge_index = torch.LongTensor(np.transpose(edge_index))
        assert graph.edge_index.shape == (2, len(bonds) * 2), AssertionError(
            "[smis2graphs] Wrong shape of graph.edge_index"
        )
        graph.edge_attr = torch.Tensor(edge_attr)
        assert graph.edge_attr.shape == (len(bonds) * 2, 7), AssertionError(
            "[smis2graphs] Wrong shape of graph.edge_attr"
        )
        graphs[smi] = graph
    return graphs


def seqs2graphs(ids_seqs):

    def distance_map(id, seq):
        DISTANCE_CUTOFF = 8.0
        parser = PDBParser()
        if not os.access(os.path.join("PDBs", id + ".pdb"), os.R_OK):
            assert len(seq) <= 400
            r = requests.post(
                "https://api.esmatlas.com/foldSequence/v1/pdb/",
                data=seq,
                verify=False,
            )
            assert r.status_code == 200
            os.mkdir("PDBs") if not os.path.exists("PDBs") else None
            with open(os.path.join("PDBs", id + ".pdb"), "w") as f:
                f.write(r.text)
        with open(os.path.join("PDBs", id + ".pdb"), "r") as f:
            structure = parser.get_structure(id, f)
        residues = [r for r in structure.get_residues() if r.get_id()[0] == " "]
        edge_index = []
        edge_weight = []
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
        return (
            torch.LongTensor(np.transpose(edge_index)),
            torch.Tensor(edge_weight).unsqueeze(1),
        )

    if os.access(os.path.join("id2repr.pkl"), os.R_OK):
        id2repr = pickle.load(open(os.path.join("id2repr.pkl"), "rb"))
    graphs = {}
    for id, seq in ids_seqs.items():
        graph = Data()
        if "id2repr" in locals() and id in id2repr:
            graph.x = torch.Tensor(id2repr[id]["token_representations"])
        else:
            data = [(id, seq)]
            batch_labels, batch_strs, batch_tokens = ESMBatchConverter(data)
            batch_lens = (batch_tokens != ESMAlphabet.padding_idx).sum(1)
            with torch.no_grad():
                results = ESMModel(
                    batch_tokens, repr_layers=[36], return_contacts=False
                )
            token_representations = results["representations"][36][0][
                1 : batch_lens[0] - 1
            ]
            graph.x = torch.Tensor(token_representations)
        assert graph.x.shape[1] == 2560

        graph.edge_index, graph.edge_attr = distance_map(id, seq)
        assert graph.edge_index.shape == (2, len(graph.edge_attr))
        graphs[id] = graph
    return graphs


def load_data(params):
    dataset = params["dataset"]
    assert dataset in ["Davis", "KIBA"]
    fold = params["fold"] if "fold" in params else None
    valid = params["valid"]
    if valid:
        assert fold in [0, 1, 2, 3, 4]

    if os.access(os.path.join("dataset", dataset, "processed.pkl"), os.R_OK):
        data = pickle.load(
            open(os.path.join("dataset", dataset, "processed.pkl"), "rb")
        )
    else:
        print(f"[{dataset}] Cache not found, processing data...")
        full = pd.read_csv(open(os.path.join("dataset", dataset, "full.csv"), "r"))
        drug_graphs = smis2graphs(set(full["ligand"]))
        prot_graphs = seqs2graphs(
            eval(open(os.path.join("dataset", dataset, "proteins.txt"), "r").read())
        )
        data = []
        for _, row in full.iterrows():
            if drug_graphs[row[0]] and prot_graphs[row[1]] and not np.isnan(row[2]):
                data.append(
                    [drug_graphs[row[0]], prot_graphs[row[1]], torch.Tensor([row[2]])]
                )
            else:
                data.append(None)
        print(f"[{dataset}] Processing finished, length: {len(data)}")
        pickle.dump(data, open(os.path.join("dataset", dataset, "processed.pkl"), "wb"))

    if valid:
        train_data = []
        valid_data = []
        test_data = []

        train_folds = json.load(
            open(os.path.join("dataset", dataset, "train_folds.txt"), "r")
        )
        train_ids = []
        valid_ids = []
        for i in range(5):
            if i != fold:
                train_ids.extend(train_folds[i])
            else:
                valid_ids.extend(train_folds[i])
        for i in train_ids:
            if data[i]:
                train_data.append(data[i])
        for i in valid_ids:
            if data[i]:
                valid_data.append(data[i])
        test_folds = json.load(
            open(os.path.join("dataset", dataset, "test_fold.txt"), "r")
        )
        for i in test_folds:
            if data[i]:
                test_data.append(data[i])
        return train_data, valid_data, test_data
    else:
        np.random.shuffle(data)
        if dataset == "Davis":
            assert len(data) == 25046 + 5010
            return data[:25046], data[25046:]
        elif dataset == "KIBA":
            assert len(data) == 98545 + 19709
            return data[:98545], data[98545:]
