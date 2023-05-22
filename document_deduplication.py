import os
import itertools
import torch
from tqdm import tqdm
import datetime
import tools
import torchhd
import numpy as np
import numpy
import random
import tools
import math
import pandas as pd
import time
from tap import Tap
from typing import List, Literal, NamedTuple


class Arguments(Tap):
    dataset: List[Literal["restaurant", "cddb", "core"]]
    # the dataset to run the experiment on
    dataset_dir: str = "data"  # directory containing the dataset files
    method: List[
        Literal[
            "dothash",
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
    start: List[int] = [0] # the index of the first document to use


class Config(NamedTuple):
    dataset: str  # the dataset to run the experiment on
    method: str  # method to run the experiment with
    dimensions: int  # number of dimensions to use (does not affect the exact method)
    device: torch.device  # which device to run the experiment on
    seed: int  # random number generator seed
    start: int = 0  # the index of the first document to use


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
    "hits@5",
    "hits@25",
    "hits@50",
    "time",
    "device",
]

_mersenne_prime = torch.tensor((1 << 61) - 1, dtype=torch.long)
_max_hash = torch.tensor((1 << 32) - 1, dtype=torch.long)

def get_minhash_signatures(
    edge_index,
    num_shing: int,
    num_doc: int,
    dimensions: int,
    batch_size: int,
):
    device = edge_index.device
    node_ids = torch.arange(0, num_shing).unsqueeze_(1)

    # Create parameters for a random bijective permutation function
    # http://en.wikipedia.org/wiki/Universal_hashing
    a, b = torch.randint(0, _mersenne_prime, (2, 1, dimensions))
    node_vectors = torch.bitwise_and(
        (node_ids * a + b) % _mersenne_prime, _max_hash
    )

    to_nodes, from_nodes = edge_index
    # index_reduce_ is only implemented for long type on CPU
    to_batches = torch.split(to_nodes.cpu(), batch_size)
    from_batches = torch.split(from_nodes.cpu(), batch_size)

    size = (num_doc, dimensions)
    signatures = torch.full(size, _max_hash, dtype=torch.long)
    for to_batch, from_batch in zip(to_batches, from_batches):
        from_node_vectors = torch.index_select(node_vectors, 0, from_batch)
        signatures.index_reduce_(0, to_batch, from_node_vectors, "amin")

    return signatures.to(device)


def get_simhash_signatures(
    edge_index, node_vectors, batch_size: int, num_doc: int, dimensions
):
    to_nodes, from_nodes = edge_index

    to_batches = torch.split(to_nodes, batch_size)
    from_batches = torch.split(from_nodes, batch_size)

    signatures = torch.zeros((num_doc, dimensions))
    for to_batch, from_batch in zip(to_batches, from_batches):
        from_node_vectors = torch.index_select(node_vectors, 0, from_batch)
        signatures.index_add_(0, to_batch, from_node_vectors)

    return signatures.greater(0)


def get_random_node_vectors(num_nodes: int, dimensions: int, device=None):
    scale = math.sqrt(1 / dimensions)

    node_vectors = torchhd.random(num_nodes, dimensions, device=device)
    node_vectors.mul_(scale)  # make them unit vectors
    return node_vectors


def w_shingle(string, w):
    """Return the set of contiguous sequences (shingles) of `w` words
    in `string`."""
    if isinstance(string, str):
        words = string.split()
        num_words = len(words)

        if w > num_words or w == 0:
            return string

        return [" ".join(words[i : i + w]) for i in range(len(words) - w + 1)]
    else:
        return []


def get_metrics(conf: Config, args: Arguments, df_info, df_pairs, device=None):

        adamic_adar_res = {}
        words_dict = {}
        documents = {}
        added_docs = []
        shing_doc = []

        for index, i in df_info[conf.start : (conf.start + 10000)].iterrows():
            dups = list(df_pairs[df_pairs["itemID_1"] == i.itemID].itemID_2)
            added_docs += dups
            doc = {"itemID": i.itemID, "description": i.description, "duplicate": dups}
            documents[i.itemID] = doc

        for i in added_docs:
            if i not in documents:
                dups = list(df_pairs[df_pairs["itemID_1"] == i].itemID_2)
                doc = {
                    "itemID": i,
                    "description": df_info[df_info["itemID"] == i].description.tolist()[
                        0
                    ],
                    "duplicate": dups,
                }
                documents[i] = doc

        words_dict1 = {}

        # Count number of time a word appears in all the documents

        if conf.method == "dothash":
            for key, value in tqdm(documents.items()):
                aux_dict = {}
                for j in w_shingle(value["description"], 2):
                    if j not in words_dict1 and j not in aux_dict:
                        words_dict1[j] = 1
                        aux_dict[j] = 1
                    elif j not in aux_dict:
                        words_dict1[j] += 1
                        aux_dict[j] = 1

            documents_dict = {}

            dimensions = conf.dimensions
            del documents_dict
            del adamic_adar_res
            del words_dict
            documents_dict = {}
            num_dcs = len(documents)
            adamic_adar_res = {}
            words_dict = {}
            torch.cuda.empty_cache()
            for k, v in tqdm(words_dict1.items()):
                if v > 1:
                    words_dict[k] = ((np.sqrt(num_dcs / v))) * torchhd.random(
                        1, dimensions, device=device
                    ).mul_(math.sqrt(1 / dimensions))
                else:
                    words_dict[k] = torchhd.random(
                        1, dimensions, device=device
                    ).mul_(math.sqrt(1 / dimensions))

            for index, (key, value) in enumerate(tqdm(documents.items())):
                hv_doc = torch.zeros(dimensions, device=device)
                for j in w_shingle(value["description"], 2):
                    hv_doc = torchhd.bundle(hv_doc, words_dict[j])
                documents_dict[key] = hv_doc

        elif conf.method == "minhash":
            count = 0
            words_dict = {}
            for key, value in tqdm(documents.items()):
                for j in w_shingle(value["description"], 2):
                    if j not in words_dict:
                        words_dict[j] = count
                        count += 1

            documents_dict = {}

            for index, row in tqdm(documents.items()):
                shing_lis = w_shingle(row["description"], 2)
                for j in shing_lis:
                    shing_doc.append([index, words_dict[j]])

            dimensions = conf.dimensions
            res = None
            del res
            torch.cuda.empty_cache()
            res = get_minhash_signatures(
                torch.transpose(torch.tensor(shing_doc), 0, 1),
                count,
                np.array(shing_doc).max() + 1,
                dimensions,
                args.batch_size,
            )
            documents_dict = {}
            for index, (key, value) in enumerate(tqdm(documents.items())):
                documents_dict[key] = res[
                    documents[list(documents.keys())[index]]["itemID"]
                ]

        elif conf.method == "simhash":
            count = 0
            words_dict = {}
            for key, value in tqdm(documents.items()):
                for j in w_shingle(value["description"], 2):
                    if j not in words_dict:
                        words_dict[j] = count
                        count += 1

            documents_dict = {}

            sizes = []
            for index, row in tqdm(documents.items()):
                shing_lis = w_shingle(row["description"], 2)
                countt = 0
                for j in shing_lis:
                    countt += 1
                    shing_doc.append([index, words_dict[j]])
                sizes.append(count)

            dimensions = conf.dimensions
            res = None
            graph = None
            node_vectors = None
            del res
            del graph
            del node_vectors
            torch.cuda.empty_cache()
            graph = torch.transpose(torch.tensor(shing_doc), 0, 1)
            node_vectors = get_random_node_vectors(count, dimensions)
            res = get_simhash_signatures(
                graph,
                node_vectors,
                args.batch_size,
                np.array(shing_doc).max() + 1,
                dimensions,
            )
            documents_dict = {}
            for index, (key, value) in enumerate(tqdm(documents.items())):
                documents_dict[key] = res[
                    documents[list(documents.keys())[index]]["itemID"]
                ]

        values = list(documents_dict.values())
        keys = list(documents_dict.keys())
        doc_len = len(documents_dict)
        reps = int(math.ceil(doc_len / args.batch_size))

        total_time = 0

        for index, (k, v) in enumerate(tqdm(list(documents_dict.items())[:-2])):
            doc_class = None
            first = True
            for i in range(reps):
                t = time.time()
                compare_vals = (
                    torch.stack(values[(i * args.batch_size) : ((i + 1) * args.batch_size)])
                    .squeeze(-2)
                    .to(device)
                )

                if conf.method == "dothash":
                    result = tools.dot(
                        values[index].expand(len(compare_vals), dimensions).to(device),
                        compare_vals,
                    )

                elif conf.method == "minhash":
                    result = tools.minhash_jaccard(
                        values[index].expand(len(compare_vals), dimensions).to(device),
                        compare_vals,
                    )

                elif conf.method == "simhash":
                    result = tools.simhash_jaccard(
                        values[index].expand(len(compare_vals), dimensions).to(device),
                        compare_vals,
                        torch.tensor(sizes[index]).expand(len(compare_vals)).to(device),
                        torch.tensor(sizes[(i * args.batch_size) : ((i + 1) * args.batch_size)]).to(device),
                    )
                    
                total_time += time.time() - t
                if first:
                    doc_class = result
                    first = False
                else:
                    doc_class = torch.cat((doc_class, result), 0)
            top = torch.topk(doc_class, 51)
            adamic_adar_res[index] = top

        hits = [0, 0, 0]
        misses = [0, 0, 0]
        tops = [6, 26, 51]

        for index, (k, v) in enumerate(adamic_adar_res.items()):
            for topI, hh in enumerate(range(len(hits))):
                hit = False
                for j in v.indices[: tops[topI]]:
                    if index != j:
                        if (
                            not hit
                            and keys[j.item()] in documents[keys[index]]["duplicate"]
                        ):
                            hits[hh] += 1
                            hit = True
                if not hit and documents[keys[index]]["duplicate"] != []:
                    misses[hh] += 1

        return {
            "time": total_time / len(adamic_adar_res.items()),
            "hits@5": hits[0] / (hits[0] + misses[0]),
            "hits@25": hits[1] / (hits[1] + misses[1]),
            "hits@50": hits[2] / (hits[2] + misses[2]),
        }


def select_data(dataset_name, dataset_dir):
    if dataset_name == "cddb":
        path = os.path.join(dataset_dir, "cddb.csv")
        df_info = pd.read_csv(path, encoding="ISO-8859-1")
        df_info = df_info.replace(np.nan, "", regex=True)
        df_info = df_info.replace("null", "", regex=True)

        cols = ["artist", "title", "category", "genre", "year", "tracks (merge)"]
        df_info["itemID"] = df_info["pk"]
        df_info["description"] = df_info[cols].apply(
            lambda row: " ".join(row.values.astype(str)), axis=1
        )

        path = os.path.join(dataset_dir, "cddb_gold.csv")
        df_pairs = pd.read_csv(path, sep=";")

        df_pairs["itemID_1"] = df_pairs["disc1_id"]
        df_pairs["itemID_2"] = df_pairs["disc2_id"]

    elif dataset_name == "core":
        # download files at: https://drive.google.com/file/d/1uBPhyHnv74ApCw7ldMnrHpWs1Yv2pbYh/view?usp=share_link
        path = os.path.join(dataset_dir, "ItemInfo_test.csv")
        df_info = pd.read_csv(path)

        path = os.path.join(dataset_dir, "ItemPairs_test.csv")
        df_pairs = pd.read_csv(path)

    elif dataset_name == "restaurant":
        path = os.path.join(dataset_dir, "restaurant.csv")
        df_info = pd.read_csv(path)
        df_info = df_info.replace(np.nan, "", regex=True)

        cols = ["name", "addr", "city", "phone", "type"]
        df_info["itemID"] = df_info["id"]
        df_info["description"] = df_info[cols].apply(
            lambda row: " ".join(row.values.astype(str)), axis=1
        )

        path = os.path.join(dataset_dir, "restaurant_gold.csv")
        df_pairs = pd.read_csv(path)
        df_pairs["itemID_1"] = df_pairs["id_1"]
        df_pairs["itemID_2"] = df_pairs["id_2"]

    return df_info, df_pairs


def main(conf: Config, args: Arguments, result_file: str):
    torch.manual_seed(conf.seed)
    numpy.random.seed(conf.seed)
    random.seed(conf.seed)

    print("Device:", conf.device)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    with tools.open_metric_writer(result_file, METRICS) as write:
        print("Dataset:", conf.dataset)
        df_info, df_pairs = select_data(conf.dataset, args.dataset_dir)

        try:
            metrics = get_metrics(conf, args, df_info, df_pairs, device=conf.device)
            metrics["dimensions"] = conf.dimensions
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
        "deduplication-"
        + datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        + ".csv"
    )

    result_file = os.path.join(args.result_dir, result_filename)
    os.makedirs(args.result_dir, exist_ok=True)

    devices = {default_to_cpu(d) for d in args.device}

    options = (args.seed, devices, args.dimensions, args.dataset, args.method, args.start)
    for seed, device, dimensions, dataset, method, start in itertools.product(*options):
        config = Config(dataset, method, dimensions, device, seed, start)
        main(config, args, result_file)
