import os
import torch
import numpy
import random
import itertools
from torch import Tensor, LongTensor
from tqdm import tqdm
import datetime
from ogb.linkproppred import PygLinkPropPredDataset
from typing import Literal, NamedTuple, List, Optional
from tap import Tap

from torch_geometric.data import Data
from torch_geometric.datasets import SNAPDataset, FacebookPagePage, WikipediaNetwork
from torch_geometric.utils import negative_sampling, to_undirected

import tools


class Arguments(Tap):
    dataset: List[
        Literal[
            "ogbl-ddi",
            "ogbl-ppa",
            "ogbl-collab",
            "soc-epinions1",
            "soc-livejournal1",
            "soc-pokec",
            "soc-slashdot0811",
            "soc-slashdot0922",
            "facebook",
            "wikipedia",
        ]
    ]
    # the dataset to run the experiment on
    dataset_dir: str = "data"  # directory containing the dataset files
    method: List[
        Literal[
            "jaccard",
            "adamic-adar",
            "common-neighbors",
            "resource-allocation",
            "dothash-jaccard",
            "dothash-adamic-adar",
            "dothash-common-neighbors",
            "dothash-resource-allocation",
            "minhash",
            "simhash",
        ]
    ]
    # method to run the experiment with
    dimensions: List[int]
    # number of dimensions to use (does not affect the exact method)
    batch_size: int = 16384  # number of nodes to evaluate at once
    result_dir: str = "results"  # directory to write the results to
    device: List[str] = ["cpu"]  # which device to run the experiment on
    seed: List[int] = [1]  # random number generator seed


class Config(NamedTuple):
    dataset: str  # the dataset to run the experiment on
    method: str  # method to run the experiment with
    dimensions: int  # number of dimensions to use (does not affect the exact method)
    device: torch.device  # which device to run the experiment on
    seed: int  # random number generator seed


class Result(NamedTuple):
    output_pos: torch.Tensor
    output_neg: torch.Tensor
    init_time: float
    calc_time: float
    dimensions: int


METRICS = [
    "method",
    "dataset",
    "dimensions",
    "hits@20",
    "hits@50",
    "hits@100",
    "init_time",
    "calc_time",
    "num_node_pairs",
    "device",
]


class Method:
    signatures: Optional[Tensor]
    num_neighbors: Optional[Tensor]

    def __init__(self, dimensions, batch_size, device):
        self.dimensions = dimensions
        self.device = device
        self.batch_size = batch_size

    def init_signatures(self, edge_index: LongTensor, num_nodes: int):
        NotImplemented

    def calc_scores(self, node_ids: LongTensor, other_ids: LongTensor) -> Tensor:
        NotImplemented


class Jaccard(Method):
    def init_signatures(self, edge_index: LongTensor, num_nodes: int):
        values = torch.ones(edge_index.size(1))
        self.signatures = tools.to_scipy_csr_array(edge_index.cpu(), num_nodes, values)
        self.num_neighbors = tools.get_num_neighbors(edge_index, num_nodes).float()

    def calc_scores(self, node_ids: LongTensor, other_ids: LongTensor) -> Tensor:
        return tools.dot_jaccard(
            self.signatures[node_ids],
            self.signatures[other_ids],
            self.num_neighbors[node_ids],
            self.num_neighbors[other_ids],
        )


class AdamicAdar(Method):
    def init_signatures(self, edge_index: LongTensor, num_nodes: int):
        node_scaling = tools.get_adamic_adar_node_scaling(edge_index, num_nodes)
        values = node_scaling[edge_index[1]]
        self.signatures = tools.to_scipy_csr_array(edge_index.cpu(), num_nodes, values)

    def calc_scores(self, node_ids: LongTensor, other_ids: LongTensor) -> Tensor:
        return tools.dot(
            self.signatures[node_ids],
            self.signatures[other_ids],
        )


class CommonNeighbors(Method):
    def init_signatures(self, edge_index: LongTensor, num_nodes: int):
        values = torch.ones(edge_index.size(1))
        self.signatures = tools.to_scipy_csr_array(edge_index.cpu(), num_nodes, values)

    def calc_scores(self, node_ids: LongTensor, other_ids: LongTensor) -> Tensor:
        return tools.dot(
            self.signatures[node_ids],
            self.signatures[other_ids],
        )


class ResourceAllocation(Method):
    def init_signatures(self, edge_index: LongTensor, num_nodes: int):
        node_scaling = tools.get_resource_allocation_node_scaling(edge_index, num_nodes)
        self.signatures = tools.to_scipy_csr_array(
            edge_index.cpu(), num_nodes, node_scaling[edge_index[1]]
        )

    def calc_scores(self, node_ids: LongTensor, other_ids: LongTensor) -> Tensor:
        return tools.dot(
            self.signatures[node_ids],
            self.signatures[other_ids],
        )


class DotHashJaccard(Method):
    def init_signatures(self, edge_index: LongTensor, num_nodes: int):
        node_vectors = tools.get_random_node_vectors(
            num_nodes, self.dimensions, device=self.device
        )
        self.signatures = tools.get_node_signatures(
            edge_index, node_vectors, self.batch_size
        )
        self.num_neighbors = tools.get_num_neighbors(edge_index, num_nodes).float()

    def calc_scores(self, node_ids: LongTensor, other_ids: LongTensor) -> Tensor:
        return tools.dot_jaccard(
            self.signatures[node_ids],
            self.signatures[other_ids],
            self.num_neighbors[node_ids],
            self.num_neighbors[other_ids],
        )


class DotHashAdamicAdar(Method):
    def init_signatures(self, edge_index: LongTensor, num_nodes: int):
        node_vectors = tools.get_random_node_vectors(
            num_nodes, self.dimensions, device=self.device
        )
        node_scaling = tools.get_adamic_adar_node_scaling(edge_index, num_nodes)
        node_vectors.mul_(node_scaling.unsqueeze(1))
        self.signatures = tools.get_node_signatures(
            edge_index, node_vectors, self.batch_size
        )

    def calc_scores(self, node_ids: LongTensor, other_ids: LongTensor) -> Tensor:
        return tools.dot(
            self.signatures[node_ids],
            self.signatures[other_ids],
        )


class DotHashCommonNeighbors(Method):
    def init_signatures(self, edge_index: LongTensor, num_nodes: int):
        node_vectors = tools.get_random_node_vectors(
            num_nodes, self.dimensions, device=self.device
        )
        self.signatures = tools.get_node_signatures(
            edge_index, node_vectors, self.batch_size
        )

    def calc_scores(self, node_ids: LongTensor, other_ids: LongTensor) -> Tensor:
        return tools.dot(
            self.signatures[node_ids],
            self.signatures[other_ids],
        )


class DotHashResourceAllocation(Method):
    def init_signatures(self, edge_index: LongTensor, num_nodes: int):
        node_vectors = tools.get_random_node_vectors(
            num_nodes, self.dimensions, device=self.device
        )
        node_scaling = tools.get_resource_allocation_node_scaling(edge_index, num_nodes)
        node_vectors.mul_(node_scaling.unsqueeze(1))
        self.signatures = tools.get_node_signatures(
            edge_index, node_vectors, self.batch_size
        )

    def calc_scores(self, node_ids: LongTensor, other_ids: LongTensor) -> Tensor:
        return tools.dot(
            self.signatures[node_ids],
            self.signatures[other_ids],
        )


class MinHash(Method):
    def init_signatures(self, edge_index: LongTensor, num_nodes: int):
        self.signatures = tools.get_minhash_signatures(
            edge_index, num_nodes, self.dimensions, self.batch_size
        )

    def calc_scores(self, node_ids: LongTensor, other_ids: LongTensor) -> Tensor:
        return tools.minhash_jaccard(
            self.signatures[node_ids],
            self.signatures[other_ids],
        )


class SimHash(Method):
    def init_signatures(self, edge_index: LongTensor, num_nodes: int):
        node_vectors = tools.get_random_binary_node_vectors(
            num_nodes, self.dimensions, device=self.device
        )
        self.signatures = tools.get_simhash_signatures(
            edge_index, node_vectors, self.batch_size
        )
        self.num_neighbors = tools.get_num_neighbors(edge_index, num_nodes).float()

    def calc_scores(self, node_ids: LongTensor, other_ids: LongTensor) -> Tensor:
        return tools.simhash_jaccard(
            self.signatures[node_ids],
            self.signatures[other_ids],
            self.num_neighbors[node_ids],
            self.num_neighbors[other_ids],
        )


class LinkPredDataset:
    def __init__(self, graph: Data):
        self.graph = graph

        edge_index = graph.edge_index
        num_nodes = graph.num_nodes
        num_edges = graph.num_edges
        num_test_edges = num_edges // 20

        self.edge_neg = negative_sampling(
            edge_index=edge_index,
            num_nodes=num_nodes,
            method="sparse",
            force_undirected=True,
            num_neg_samples=num_test_edges,
        )

        permuted_indices = torch.randperm(num_edges)
        self.edge = edge_index[:, permuted_indices[:num_test_edges]]
        self.data = Data(
            edge_index=edge_index[:, permuted_indices[num_test_edges:]],
            num_nodes=num_nodes,
        )

    def get_edge_split(self):
        train = None
        valid = None

        test = {
            "edge": self.edge.T,
            "edge_neg": self.edge_neg.T,
        }

        return {"train": train, "valid": valid, "test": test}


def evaluate_hits_at(pred_pos: Tensor, pred_neg: Tensor, K: int) -> float:
    if len(pred_neg) < K:
        return 1.0

    kth_score_in_negative_edges = torch.topk(pred_neg, K)[0][-1]
    num_hits = torch.sum(pred_pos > kth_score_in_negative_edges).cpu()
    hitsK = float(num_hits) / len(pred_pos)
    return hitsK


def executor(args: Arguments, method: Method, dataset, device=None):
    graph = dataset.data.to(device)
    split_edge = dataset.get_edge_split()
    pos_test_edge = split_edge["test"]["edge"].to(device)
    neg_test_edge = split_edge["test"]["edge_neg"].to(device)

    get_duration = tools.stopwatch()
    method.init_signatures(to_undirected(graph.edge_index), graph.num_nodes)
    init_time = get_duration()

    pos_scores = []
    neg_scores = []
    pos_test_edge_loader = pos_test_edge.split(args.batch_size)
    neg_test_edge_loader = neg_test_edge.split(args.batch_size)

    get_duration = tools.stopwatch()
    for edge_batch in tqdm(pos_test_edge_loader, leave=False):
        node_ids, other_ids = edge_batch[:, 0], edge_batch[:, 1]
        scores = method.calc_scores(node_ids, other_ids)
        pos_scores.append(scores.cpu())

    for edge_batch in tqdm(neg_test_edge_loader, leave=False):
        node_ids, other_ids = edge_batch[:, 0], edge_batch[:, 1]
        scores = method.calc_scores(node_ids, other_ids)
        neg_scores.append(scores.cpu())

    calc_time = get_duration()
    pos_scores = torch.cat(pos_scores)
    neg_scores = torch.cat(neg_scores)

    dimensions = method.signatures.shape[1]

    return Result(pos_scores, neg_scores, init_time, calc_time, dimensions)


def get_dataset(name: str, root: str):
    if name.startswith("ogbl-"):
        return PygLinkPropPredDataset(name, root)
    elif name.startswith("soc-"):
        dataset = SNAPDataset(root, name)
        return LinkPredDataset(dataset.data)
    elif name == "facebook":
        dataset = FacebookPagePage(os.path.join(root, "facebookpagepage"))
        return LinkPredDataset(dataset.data)
    elif name == "wikipedia":
        dataset = WikipediaNetwork("data", "crocodile", geom_gcn_preprocess=False)
        return LinkPredDataset(dataset.data)
    else:
        raise NotImplementedError()


def get_hits(result: Result):
    output = []

    for K in [20, 50, 100]:
        test_hits = evaluate_hits_at(result.output_pos, result.output_neg, K)
        output.append(test_hits)

    return tuple(output)


def get_metrics(conf: Config, args: Arguments, dataset, device=None):
    if conf.method == "jaccard":
        method_cls = Jaccard
    elif conf.method == "adamic-adar":
        method_cls = AdamicAdar
    elif conf.method == "common-neighbors":
        method_cls = CommonNeighbors
    elif conf.method == "resource-allocation":
        method_cls = ResourceAllocation
    elif conf.method == "dothash-jaccard":
        method_cls = DotHashJaccard
    elif conf.method == "dothash-adamic-adar":
        method_cls = DotHashAdamicAdar
    elif conf.method == "dothash-common-neighbors":
        method_cls = DotHashCommonNeighbors
    elif conf.method == "dothash-resource-allocation":
        method_cls = DotHashResourceAllocation
    elif conf.method == "minhash":
        method_cls = MinHash
    elif conf.method == "simhash":
        method_cls = SimHash
    else:
        raise NotImplementedError(f"requested method {conf.method} is not implemented")

    method = method_cls(conf.dimensions, args.batch_size, device)

    result = executor(args, method, dataset, device=device)
    total_time = result.init_time + result.calc_time
    num_node_pairs = result.output_pos.size(0) + result.output_neg.size(0)
    hits = get_hits(result)
    print(
        f"{conf.method}: Hits: {hits[0]:.3g}@20, {hits[1]:.3g}@50, {hits[2]:.3g}@100; Time: {result.init_time:.3g}s + {result.calc_time:.3g}s = {total_time:.3g}s; Dims: {result.dimensions}"
    )

    return {
        "dimensions": result.dimensions,
        "hits@20": hits[0],
        "hits@50": hits[1],
        "hits@100": hits[2],
        "init_time": result.init_time,
        "calc_time": result.calc_time,
        "num_node_pairs": num_node_pairs,
    }


def main(conf: Config, args: Arguments, result_file: str):

    torch.manual_seed(conf.seed)
    numpy.random.seed(conf.seed)
    random.seed(conf.seed)

    print("Device:", conf.device)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    with tools.open_metric_writer(result_file, METRICS) as write:

        print("Dataset:", conf.dataset)
        dataset = get_dataset(conf.dataset, args.dataset_dir)

        try:
            metrics = get_metrics(conf, args, dataset, device=conf.device)
            metrics["method"] = conf.method
            metrics["dataset"] = conf.dataset
            metrics["device"] = conf.device.type
            write(metrics)
        except Exception as e:
            print(e)


def default_to_cpu(device: str) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(device)
    else:
        return torch.device("cpu")


if __name__ == "__main__":

    args = Arguments(underscores_to_dashes=True).parse_args()

    result_filename = (
        "link_prediction-"
        + datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        + ".csv"
    )

    result_file = os.path.join(args.result_dir, result_filename)
    os.makedirs(args.result_dir, exist_ok=True)

    devices = {default_to_cpu(d) for d in args.device}

    options = (args.seed, devices, args.dimensions, args.dataset, args.method)
    for seed, device, dimensions, dataset, method in itertools.product(*options):
        config = Config(dataset, method, dimensions, device, seed)
        main(config, args, result_file)
