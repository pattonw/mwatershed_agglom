from funlib.segment.arrays import relabel, replace_values
from funlib.geometry import Coordinate
from funlib.persistence import open_ds, Array, graphs

import mwatershed as mws
import daisy

import pymongo
from scipy.ndimage import measurements
import numpy as np
from scipy.ndimage.filters import gaussian_filter

import logging
import json
import sys
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def watershed_in_block(
    affs,
    block,
    context,
    rag_provider,
    fragments_out,
    num_voxels_in_block,
    filter_fragments=0.5,
):
    """
    Args:
        filter_fragments (float):
            Filter fragments that have an average affinity lower than this
            value.
    """

    logger.info("reading affs from %s", block.read_roi)

    offsets = affs.data.attrs["offsets"]
    affs = affs.intersect(block.read_roi)
    affs.materialize()

    if affs.dtype == np.uint8:
        logger.info("Assuming affinities are in [0,255]")
        max_affinity_value = 255.0
        affs.data = affs.data.astype(np.float64)
    else:
        max_affinity_value = 1.0

    if affs.data.max() < 1e-3:
        return

    affs.data /= max_affinity_value

    # extract fragments
    adjacent_edge_bias = -0.4  # bias towards merging
    lr_edge_bias = -0.7  # bias heavily towards splitting

    # add some random noise to affs (this is particularly necessary if your affs are
    #  stored as uint8 or similar)
    # If you have many affinities of the exact same value the order they are processed
    # in may be fifo, so you can get annoying streaks.
    random_noise = np.random.randn(*affs.data.shape) * 0.001
    # add smoothed affs, to solve a similar issue to the random noise. We want to bias
    # towards processing the central regions of objects first.
    smoothed_affs = (
        gaussian_filter(affs.data, sigma=(0, *(Coordinate(context) / 3))) - 0.5
    ) * 0.01
    shift = np.array(
        [adjacent_edge_bias if max(offset) <= 1 else lr_edge_bias for offset in offsets]
    ).reshape((-1, *((1,) * (len(affs.data.shape) - 1))))
    fragments_data = mws.agglom(
        affs.data + shift + random_noise + smoothed_affs,
        offsets=offsets,
    )

    if filter_fragments > 0:
        average_affs = np.mean(affs.data, axis=0)

        filtered_fragments = []

        fragment_ids = np.unique(fragments_data)

        for fragment, mean in zip(
            fragment_ids, measurements.mean(average_affs, fragments_data, fragment_ids)
        ):
            if mean < filter_fragments:
                filtered_fragments.append(fragment)

        filtered_fragments = np.array(filtered_fragments, dtype=fragments_data.dtype)
        replace = np.zeros_like(filtered_fragments)
        replace_values(fragments_data, filtered_fragments, replace, inplace=True)

    fragments = Array(fragments_data, affs.roi, affs.voxel_size)

    # crop fragments to write_roi
    fragments = fragments[block.write_roi]
    fragments.materialize()
    max_id = fragments.data.max()

    fragments.data, max_id = relabel(fragments.data)
    assert max_id < num_voxels_in_block

    # ensure unique IDs
    id_bump = block.block_id[1] * num_voxels_in_block
    logger.info("bumping fragment IDs by %i", id_bump)
    fragments.data[fragments.data > 0] += id_bump
    fragment_ids = range(1, max_id + 1)

    # store fragments
    fragments_out[block.write_roi] = fragments

    # following only makes a difference if fragments were found
    if max_id == 0:
        return

    # get fragment centers
    fragment_centers = {
        fragment: block.write_roi.get_offset() + affs.voxel_size * Coordinate(center)
        for fragment, center in zip(
            fragment_ids,
            measurements.center_of_mass(fragments.data, fragments.data, fragment_ids),
        )
        if not np.isnan(center[0])
    }

    # store nodes
    rag = rag_provider[block.write_roi]
    rag.add_nodes_from(
        [
            (node, {"center_z": c[0], "center_y": c[1], "center_x": c[2]})
            for node, c in fragment_centers.items()
        ]
    )
    rag.write_nodes(block.write_roi)


def extract_fragments_worker(input_config):
    logger.info(sys.argv)

    with open(input_config, "r") as f:
        config = json.load(f)

    logger.info(config)

    sample_name = config["sample_name"]
    affs_file = config["affs_file"]
    affs_dataset = config["affs_dataset"]
    fragments_file = config["fragments_file"]
    fragments_dataset = config["fragments_dataset"]
    db_name = config["db_name"]
    db_host = config["db_host"]
    context = config["context"]
    num_voxels_in_block = config["num_voxels_in_block"]
    epsilon_agglomerate = config["epsilon_agglomerate"]
    filter_fragments = config["filter_fragments"]

    logger.info("Reading affs from %s", affs_file)
    affs = open_ds(affs_file, affs_dataset, mode="r")

    logger.info("writing fragments to %s", fragments_file)
    fragments = open_ds(fragments_file, fragments_dataset, mode="r+")

    if config["mask_file"] is None:
        logger.info("Reading mask from %s", config["mask_file"])
        mask = open_ds(config["mask_file"], config["mask_dataset"], mode="r")

    else:
        mask = None

    # open RAG DB
    logger.info("Opening RAG DB...")
    rag_provider = graphs.MongoDbGraphProvider(
        db_name,
        host=db_host,
        mode="r+",
        directed=False,
        position_attribute=["center_z", "center_y", "center_x"],
        edges_collection=f"{sample_name}_edges",
        nodes_collection=f"{sample_name}_nodes",
        meta_collection=f"{sample_name}_meta",
    )
    logger.info("RAG DB opened")

    # open block done DB
    mongo_client = pymongo.MongoClient(db_host)
    db = mongo_client[db_name]
    blocks_extracted = db[f"{sample_name}_fragment_blocks_extracted"]

    client = daisy.Client()

    while True:
        logger.info("getting block")
        with client.acquire_block() as block:
            logger.info(f"got block {block}")

            if block is None:
                break

            start = time.time()

            logger.info("block read roi begin: %s", block.read_roi.get_begin())
            logger.info("block read roi shape: %s", block.read_roi.get_shape())
            logger.info("block write roi begin: %s", block.write_roi.get_begin())
            logger.info("block write roi shape: %s", block.write_roi.get_shape())

            watershed_in_block(
                affs,
                block,
                context,
                rag_provider,
                fragments,
                num_voxels_in_block=num_voxels_in_block,
                mask=mask,
                epsilon_agglomerate=epsilon_agglomerate,
                filter_fragments=filter_fragments,
            )

            document = {
                "block_id": block.block_id,
                "read_roi": (block.read_roi.get_begin(), block.read_roi.get_shape()),
                "write_roi": (block.write_roi.get_begin(), block.write_roi.get_shape()),
                "start": start,
                "duration": time.time() - start,
            }
            blocks_extracted.insert_one(document)
            logger.info(f"releasing block: {block}")


if __name__ == "__main__":
    extract_fragments_worker(sys.argv[1])
