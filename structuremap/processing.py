#!python

# builtin
import json
import os
import socket
import re
from itertools import groupby
import urllib.request
import random
import logging
import ssl
import tempfile

# external
import numba
import numpy as np
import pandas as pd
import tqdm
import h5py
import statsmodels.stats.multitest
import Bio.PDB.MMCIF2Dict
import scipy.stats
import sys

if getattr(sys, 'frozen', False):
    print('Using frozen version. Setting SSL context to unverified.')
    ssl._create_default_https_context = ssl._create_unverified_context

def download_alphafold_cif(
    proteins: list,
    out_folder: str,
    out_format: str = "{}.cif",
    alphafold_cif_url: str = 'https://alphafold.ebi.ac.uk/files/AF-{}-F1-model_v1.cif',
    timeout: int = 60,
    verbose_log: bool = False,
) -> tuple:
    """
    Function to download .cif files of protein structures predicted by AlphaFold.

    Parameters
    ----------
    proteins : list
        List (or any other iterable) of UniProt protein accessions for which to
        download the structures.
    out_folder : str
        Path to the output folder.
    alphafold_cif_url : str
        The base link from where to download cif files.
        The brackets {} are replaced by a protein name from the proteins list.
        Default is 'https://alphafold.ebi.ac.uk/files/AF-{}-F1-model_v1.cif'.
    timeout : int
        Time to wait for reconnection of downloads.
        Default is 60.
    verbose_log: bool
        Whether to write verbose logging information.
        Default is False.

    Returns
    -------
    : (list, list, list)
        The lists of valid, invalid and existing protein accessions.
    """
    socket.setdefaulttimeout(timeout)
    valid_proteins = []
    invalid_proteins = []
    existing_proteins = []
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    for protein in tqdm.tqdm(proteins):
        name_in = alphafold_cif_url.format(protein)
        name_out = os.path.join(
            out_folder,
            out_format.format(protein)
        )
        if os.path.isfile(name_out):
            existing_proteins.append(protein)
        else:
            try:
                urllib.request.urlretrieve(name_in, name_out)
                valid_proteins.append(protein)
            except urllib.error.HTTPError:
                if verbose_log:
                    logging.info(f"Protein {protein} not available for CIF download.")
                invalid_proteins.append(protein)
    logging.info(f"Valid proteins: {len(valid_proteins)}")
    logging.info(f"Invalid proteins: {len(invalid_proteins)}")
    logging.info(f"Existing proteins: {len(existing_proteins)}")
    return(valid_proteins, invalid_proteins, existing_proteins)


def download_alphafold_pae(
    proteins: list,
    out_folder: str,
    out_format: str = "pae_{}.hdf",
    alphafold_pae_url: str = 'https://alphafold.ebi.ac.uk/files/AF-{}-F1-predicted_aligned_error_v1.json',
    timeout: int = 60,
    verbose_log: bool = False,
) -> tuple:
    """
    Function to download paired aligned errors (pae) for protein structures
    predicted by AlphaFold.

    Parameters
    ----------
    proteins : list
        List (or any other iterable) of UniProt protein accessions for which to
        download the structures.
    out_folder : str
        Path to the output folder.
    out_format : str
        The default file name of the cif files to be saved.
        The brackets {} are replaced by a protein name from the proteins list.
        Default is 'pae_{}.hdf'.
    alphafold_pae_url : str
        The base link from where to download pae files.
        The brackets {} are replaced by a protein name from the proteins list.
        Default is 'https://alphafold.ebi.ac.uk/files/AF-{}-F1-predicted_aligned_error_v1.json'.
    timeout : int
        Time to wait for reconnection of downloads.
        Default is 60.
    verbose_log: bool
        Whether to write verbose logging information.
        Default is False.

    Returns
    -------
    : (list, list, list)
        The valid, invalid and existing proteins.
    """
    socket.setdefaulttimeout(timeout)
    valid_proteins = []
    invalid_proteins = []
    existing_proteins = []
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    for protein in tqdm.tqdm(proteins):
        name_out = os.path.join(
            out_folder,
            out_format.format(protein)
        )
        if os.path.isfile(name_out):
            existing_proteins.append(protein)
        else:
            try:
                name_in = alphafold_pae_url.format(protein)
                with tempfile.TemporaryDirectory() as tmp_pae_dir:
                    tmp_pae_file_name = os.path.join(
                        tmp_pae_dir,
                        "pae_{protein}.json"
                    )
                    urllib.request.urlretrieve(name_in, tmp_pae_file_name)
                    with open(tmp_pae_file_name) as tmp_pae_file:
                        data = json.loads(tmp_pae_file.read())
                dist = np.array(data[0]['distance'])
                data_list = [('dist', dist)]
                if getattr(sys, 'frozen', False):
                    print('Using frozen h5py w/ gzip compression')
                    with h5py.File(name_out, 'w') as hdf_root:
                        for key, data in data_list:
                            print(f'h5py {key}')
                            hdf_root.create_dataset(
                                                name=key,
                                                data=data,
                                                compression="gzip",
                                                shuffle=True,
                                            )
                    print('Done')
                else:
                    with h5py.File(name_out, 'w') as hdf_root:
                        for key, data in data_list:
                            hdf_root.create_dataset(
                                                name=key,
                                                data=data,
                                                compression="lzf",
                                                shuffle=True,
                                            )

                valid_proteins.append(protein)
            except urllib.error.HTTPError:
                if verbose_log:
                    logging.info(f"Protein {protein} not available for PAE download.")
                # @ Include HDF IO errors as well, which should probably be handled differently.
                invalid_proteins.append(protein)
    logging.info(f"Valid proteins: {len(valid_proteins)}")
    logging.info(f"Invalid proteins: {len(invalid_proteins)}")
    logging.info(f"Existing proteins: {len(existing_proteins)}")
    return(valid_proteins, invalid_proteins, existing_proteins)


def format_alphafold_data(
    directory: str,
    protein_ids: list,
) -> pd.DataFrame:
    """
    Function to import structure files and format them into a combined dataframe.

    Parameters
    ----------
    directory : str
        Path to the folder with all .cif files.
    proteins : list
        List of UniProt protein accessions to create an annotation table.
        If an empty list is provided, all proteins in the provided directory
        are used to create the annotation table.

    Returns
    -------
    : pd.DataFrame
        A dataframe with structural information presented in following columns:
        ['protein_id', 'protein_number', 'AA', 'position', 'quality',
        'x_coord_c', 'x_coord_ca', 'x_coord_cb', 'x_coord_n', 'y_coord_c',
        'y_coord_ca', 'y_coord_cb', 'y_coord_n', 'z_coord_c', 'z_coord_ca',
        'z_coord_cb', 'z_coord_n', 'secondary_structure', 'structure_group',
        'BEND', 'HELX', 'STRN', 'TURN', 'unstructured']
    """

    alphafold_annotation_l = []
    protein_number = 0

    for file in tqdm.tqdm(sorted(os.listdir(directory))):

        if file.endswith("cif"):
            filepath = os.path.join(directory, file)

            protein_id = re.sub(r'.cif', '', file)

            if ((protein_id in protein_ids) or (len(protein_ids) == 0)):

                protein_number += 1

                structure = Bio.PDB.MMCIF2Dict.MMCIF2Dict(filepath)

                df = pd.DataFrame({'protein_id': structure['_atom_site.pdbx_sifts_xref_db_acc'],
                                   'protein_number': protein_number,
                                   'AA': structure['_atom_site.pdbx_sifts_xref_db_res'],
                                   'position': structure['_atom_site.label_seq_id'],
                                   'quality': structure['_atom_site.B_iso_or_equiv'],
                                   'atom_id': structure['_atom_site.label_atom_id'],
                                   'x_coord': structure['_atom_site.Cartn_x'],
                                   'y_coord': structure['_atom_site.Cartn_y'],
                                   'z_coord': structure['_atom_site.Cartn_z']})

                df = df[df.atom_id.isin(['CA', 'CB', 'C', 'N'])].reset_index(drop=True)
                df = df.pivot(index=['protein_id',
                                     'protein_number',
                                     'AA', 'position',
                                     'quality'],
                              columns="atom_id")
                df = pd.DataFrame(df.to_records())

                df = df.rename(columns={"('x_coord', 'CA')": "x_coord_ca",
                                        "('y_coord', 'CA')": "y_coord_ca",
                                        "('z_coord', 'CA')": "z_coord_ca",
                                        "('x_coord', 'CB')": "x_coord_cb",
                                        "('y_coord', 'CB')": "y_coord_cb",
                                        "('z_coord', 'CB')": "z_coord_cb",
                                        "('x_coord', 'C')": "x_coord_c",
                                        "('y_coord', 'C')": "y_coord_c",
                                        "('z_coord', 'C')": "z_coord_c",
                                        "('x_coord', 'N')": "x_coord_n",
                                        "('y_coord', 'N')": "y_coord_n",
                                        "('z_coord', 'N')": "z_coord_n"})

                df = df.apply(pd.to_numeric, errors='ignore')

                df['secondary_structure'] = 'unstructured'

                if '_struct_conf.conf_type_id' in structure.keys():
                    start_idx = [int(i) for i in structure['_struct_conf.beg_label_seq_id']]
                    end_idx = [int(i) for i in structure['_struct_conf.end_label_seq_id']]
                    note = structure['_struct_conf.conf_type_id']

                    for i in np.arange(0, len(start_idx)):
                        df['secondary_structure'] = np.where(
                            df['position'].between(
                                start_idx[i],
                                end_idx[i]),
                            note[i],
                            df['secondary_structure'])

                alphafold_annotation_l.append(df)

    alphafold_annotation = pd.concat(alphafold_annotation_l)
    alphafold_annotation = alphafold_annotation.sort_values(
        by=['protein_number', 'position']).reset_index(drop=True)

    alphafold_annotation['structure_group'] = [re.sub('_.*', '', i)
                                               for i in alphafold_annotation[
                                               'secondary_structure']]
    str_oh = pd.get_dummies(alphafold_annotation['structure_group'],
                            dtype='int64')
    alphafold_annotation = alphafold_annotation.join(str_oh)

    return(alphafold_annotation)


@numba.njit
def get_3d_dist(
    coordinate_array_1: np.ndarray,
    coordinate_array_2: np.ndarray,
    idx_1: int,
    idx_2: int
) -> float:
    """
    Function to get the distance between two coordinates in 3D space.
    Input are two coordinate arrays and two respective indices that specify
    for which points in the coordinate arrays the distance should be calculated.

    Parameters
    ----------
    coordinate_array_1 : np.ndarray
        Array of 3D coordinates.
        Must be 3d, e.g. np.float64[:,3]
    coordinate_array_2 : np.ndarray
        Array of 3D coordinates.
        Must be 3d, e.g. np.float64[:,3]
    idx_1 : int
        Integer to select an index in coordinate_array_1.
    idx_2 : int
        Integer to select an index in coordinate_array_2.

    Returns
    -------
    : float
        Distance between the two selected 3D coordinates.
    """
    dist = np.sqrt(
        (
            coordinate_array_1[idx_1, 0] - coordinate_array_2[idx_2, 0]
        )**2 + (
            coordinate_array_1[idx_1, 1] - coordinate_array_2[idx_2, 1]
        )**2 + (
            coordinate_array_1[idx_1, 2] - coordinate_array_2[idx_2, 2]
        )**2
    )
    return(dist)


@numba.njit
def rotate_vector_around_axis(
    vector: np.ndarray,
    axis: np.ndarray,
    theta: float
) -> np.ndarray:
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta degrees.
    (https://stackoverflow.com/questions/6802577/rotation-of-3d-vector)

    Parameters
    ----------
    vector : np.ndarray
        3D vector which should be rotated.
    axis : np.ndarray
        3D vector around which the vector should be rotated.
    theta : float
        Angle (in degrees) by which the vector should be rotated around the axis.

    Returns
    -------
    : np.ndarray
        Rotation matrix.
    """
    theta = np.radians(theta)
    axis = axis / np.linalg.norm(axis)
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    rotation_matrix = np.array(
        [[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
         [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
         [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
    rotated_vector = np.dot(rotation_matrix, vector)
    return rotated_vector


@numba.njit
def get_gly_vector(
    coord_a: np.ndarray,
    coord_c: np.ndarray,
    coord_n: np.ndarray,
    idx_1: int,
    theta: float = -120
) -> np.ndarray:
    """
    Return a pseudo vector Ca -> Cb for a Glycine residue.
    The pseudo vector is centered at the origin and the
    Ccoord=N coord rotated over -120 degrees
    along the CA-C axis (see Bio.PDB package).

    Parameters
    ----------
    coord_a : np.ndarray
        Array of 3D coordinates of alpha carbon atoms across different
        amino acids.
    coord_c : np.ndarray
        Array of 3D coordinates of carboxy carbon atoms across different
        amino acids.
    coord_n : np.ndarray
        Array of 3D coordinates of amino nitrogen atoms across different
        amino acids.
    idx_1 : int
        Integer to select a specific amino acid in the coordinate arrays.
    theta : float
        The theta for the rotation.
        Default is -120.

    Returns
    -------
    : np.ndarray
        Pseudo vector Ca -> Cb for a Glycine residue.
    """
    # get unit vectors
    uv_n = (coord_n[idx_1] - coord_a[idx_1]) / get_3d_dist(coord_n, coord_a, idx_1, idx_1)
    uv_c = (coord_c[idx_1] - coord_a[idx_1]) / get_3d_dist(coord_c, coord_a, idx_1, idx_1)
    # rotation of uv_n around uv_c over -120 deg
    uv_b = rotate_vector_around_axis(vector=uv_n, axis=uv_c, theta=theta)
    return uv_b

@numba.njit
def get_angle(
    coord_a: np.ndarray,
    coord_b: np.ndarray,
    coord_c: np.ndarray,
    coord_n: np.ndarray,
    idx_1: int,
    idx_2: int
) -> float:
    """
    Calculate the angle between the vector of the target amino acid's
    side chain (Ca1 -> Cb1) and the vector pointing from the target
    amino acid's alpha carbon atom to a different amino acid's
    alpha carbon atom (Ca1 -> Ca2).

    Parameters
    ----------
    coord_a : np.ndarray
        Array of 3D coordinates of alpha carbon atoms across different
        amino acids.
    coord_b : np.ndarray
        Array of 3D coordinates of beta carbon atoms across different
        amino acids.
    coord_c : np.ndarray
        Array of 3D coordinates of carboxy carbon atoms across different
        amino acids.
    coord_n : np.ndarray
        Array of 3D coordinates of amino nitrogen atoms across different
        amino acids.
    idx_1 : int
        Integer to select a first amino acid in the coordinate arrays.
    idx_2 : int
        Integer to select a second amino acid in the coordinate arrays.

    Returns
    -------
    : float
        Angle between the side chain of the first amino acid and a second
        amino acid.
    """
    if np.isnan(coord_b[idx_1, 0]):
        # Get pseudo vector Ca -> Cb for a Gly residue.
        uv_1 = get_gly_vector(coord_a,
                              coord_c,
                              coord_n,
                              idx_1)
    else:
        # Calculate unit vector for Ca1 -> Cb1
        uv_1 = (coord_b[idx_1] - coord_a[idx_1]) / get_3d_dist(coord_b, coord_a, idx_1, idx_1)
    # Calculate unit vector for Ca1 -> Ca2
    uv_d = (coord_a[idx_2] - coord_a[idx_1]) / get_3d_dist(coord_a, coord_a, idx_1, idx_2)
    # Calculate the angle between the two unit vectors
    dot_p = np.dot(uv_1, uv_d)
    # angle = np.arccos(np.clip(dot_p, -1.0, 1.0))
    angle = np.arccos(dot_p)
    # Convert radians in degrees
    angle_deg = np.rad2deg(angle)
    return(angle_deg)


@numba.njit
def get_paired_error(
    position: np.ndarray,
    error_dist: np.ndarray,
    idx_1: int,
    idx_2: int
) -> float:
    """
    Extract paired aligned error of AlphaFold from a complete
    error matrix (error_dist) at specific sequence positions.

    Parameters
    ----------
    position : np.ndarray
        Array of amino acid positions from which to choose specific indeces.
    error_dist : np.ndarray
        Matrix of paired aligned errors of AlphaFold across all amino acids
        in a protein qequence.
    idx_1 : int
        Integer to select a first amino acid in the position array.
    idx_2 : int
        Integer to select a second amino acid in the position array.

    Returns
    -------
    : float
        Paired aligned error of the first amino acid and a second amino acid.
    """
    pos1 = position[idx_1]
    pos2 = position[idx_2]
    err = error_dist[pos1 - 1, pos2 - 1]
    return(err)


@numba.njit
def get_neighbors(
    idx_list: np.ndarray, # Technically this is not a list and it could/should be renamed.
    coord_a: np.ndarray,
    coord_b: np.ndarray,
    coord_c: np.ndarray,
    coord_n: np.ndarray,
    position: np.ndarray,
    error_dist: np.ndarray,
    max_dist: float,
    max_angle: float
) -> np.ndarray:
    """
    Get the number of amino acids within the specified distance and angle
    relative to the target amino acid.

    Parameters
    ----------
    idx_list : np.ndarray
        Array of amino acid indeces.
    coord_a : np.ndarray
        Array of 3D coordinates of alpha carbon atoms across different
        amino acids.
    coord_b : np.ndarray
        Array of 3D coordinates of beta carbon atoms across different
        amino acids.
    coord_c : np.ndarray
        Array of 3D coordinates of carboxy carbon atoms across different
        amino acids.
    coord_n : np.ndarray
        Array of 3D coordinates of amino nitrogen atoms across different
        amino acids.
    position : np.ndarray
        Array of amino acid positions.
    error_dist: : np.ndarray
        Matrix of paired aligned errors of AlphaFold across all amino acids
        in a protein qequence.
    max_dist : float
        Float specifying the maximum distance between two amino acids.
    max_angle : float
        Float specifying the maximum angle (in degrees) between two
        amino acids.

    Returns
    -------
    : np.ndarray
        Number of amino acids within the specified distance and angle.
    """
    res = []
    for x1 in idx_list:
        n_neighbors = 0
        for x2 in idx_list:
            if x1 != x2:
                paired_error = get_paired_error(
                    position=position,
                    error_dist=error_dist,
                    idx_1=x1,
                    idx_2=x2)
                if (paired_error <= max_dist):
                    dist = get_3d_dist(
                        coordinate_array_1=coord_a,
                        coordinate_array_2=coord_a,
                        idx_1=x1,
                        idx_2=x2)
                    if (dist + paired_error <= max_dist):
                        angle = get_angle(
                            coord_a=coord_a,
                            coord_b=coord_b,
                            coord_c=coord_c,
                            coord_n=coord_n,
                            idx_1=x1,
                            idx_2=x2)
                        if angle <= max_angle:
                            n_neighbors += 1
        res.append(n_neighbors)
    return(np.array(res))


@numba.njit
def find_end(
    label: int,
    start_index: int,
    values: np.ndarray
) -> int:
    """Find when the label changes.

    This assumes a sorted values array.

    Parameters
    ----------
    label : int
        The label of interest.
    start_index : int
        The previous endindex index of the previous label,
        which normally is the start_index for the current label.
    values : int
        An array with values.

    Returns
    -------
    int
        The end_index index of the label in values.
    """
    while values[start_index] == label:
        start_index += 1
        if start_index == len(values):
            break
    return start_index


def partition_df_by_prots(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generator function to split a dataframe into seperate proteins.

    NOTE: This function is significantly faster if the input df is already
    sorted by protein_number!

    Parameters
    ----------
    df : pd.DataFrame
        pd.DataFrame of formatted AlphaFold data across various proteins.

    Yields
    -------
    : pd.DataFrame
        Subset of the input dataframe only containing a single protein.
    """
    df = df.astype({'position': 'int64'})
    if not df.protein_number.is_monotonic_increasing:
        df = df.sort_values(by='protein_number').reset_index(drop=True)
    unique_proteins = df.protein_number.unique()
    end = 0
    for protein_i in tqdm.tqdm(unique_proteins):
        start = end
        end = find_end(protein_i, end, df.protein_number.values)
        prot_df = df[start:end]
        if not prot_df.position.is_monotonic_increasing:
            prot_df.sort_values(by='position', inplace=True)
        yield prot_df.reset_index(drop=True)


def annotate_accessibility(
    df: pd.DataFrame,
    max_dist: float,
    max_angle: float,
    error_dir: str,
    filename_format: str = "pae_{}.hdf",
) -> pd.DataFrame:
    """
    Half sphere exposure as calculated in
    https://onlinelibrary.wiley.com/doi/10.1002/prot.20379
    but with paired aligned error metric included.

    Parameters
    ----------
    df : pd.DataFrame
        pd.DataFrame of formatted AlphaFold data across various proteins.
        Such a dataframe is gerated by format_alphafold_data.
    max_dist : float
        Float specifying the maximum distance between two amino acids.
    max_angle : float
        Float specifying the maximum angle (in degrees) between two
        amino acids.
    error_dir: : str
        Path to the directory where the hdf files containing the matrices of
        paired aligned errors of AlphaFold are stored. This should correspond
        to the out_folder used in download_alphafold_pae.
    filename_format : str
        The file name of the pae files saved by download_alphafold_pae.
        The brackets {} are replaced by a protein name from the proteins list.
        Default is 'pae_{}.hdf'.

    Returns
    -------
    : pd.DataFrame
        Dataframe repporting the number of neighboring amino acids at the
        specified maximum distance and angle per protein, amino acid and
        position.
    """
    proteins = list()
    AA = list()
    AA_p = list()
    a_AA = list()
    for df_prot in partition_df_by_prots(df):
        protein_accession = df_prot.protein_id.values[0]
        if error_dir is not None:
            with h5py.File(os.path.join(
                error_dir,
                filename_format.format(protein_accession))
            ) as hdf_root:
                error_dist = hdf_root['dist'][...]
            size = int(np.sqrt(len(error_dist)))
            error_dist = error_dist.reshape(size, size)
            use_pae = 'pae'
        else:
            error_dist = np.zeros((df_prot.shape[0], df_prot.shape[0]))
            use_pae = 'nopae'
        idx_list = np.arange(0, df_prot.shape[0])
        res_a = get_neighbors(
            idx_list=idx_list,
            coord_a=np.vstack([df_prot.x_coord_ca.values,
                              df_prot.y_coord_ca.values,
                              df_prot.z_coord_ca.values]).T,
            coord_b=np.vstack([df_prot.x_coord_cb.values,
                              df_prot.y_coord_cb.values,
                              df_prot.z_coord_cb.values]).T,
            coord_c=np.vstack([df_prot.x_coord_c.values,
                              df_prot.y_coord_c.values,
                              df_prot.z_coord_c.values]).T,
            coord_n=np.vstack([df_prot.x_coord_n.values,
                              df_prot.y_coord_n.values,
                              df_prot.z_coord_n.values]).T,
            # If this step is slow, consider avoiding the vstack to create new arrays
            # Alternatively, it might be faster to use e.g. df[["x", "y", "z"]].values
            # as pandas might force this into a view rather than a new array
            position=df_prot.position.values,
            error_dist=error_dist,
            max_dist=max_dist,
            max_angle=max_angle)
        proteins.append(df_prot.protein_id.values)
        # using numeracal prot_numbers might be better.
        # In general it is good practice to reduce strings/objects in arrays/dfs
        # as much possible. Especially try to avoid repetetion of such types and
        # just use indices and a reference array. Rarely do you need this actual
        # values anyways.
        AA.append(df_prot.AA.values)
        AA_p.append(df_prot.position.values)
        a_AA.append(res_a)
    proteins = np.concatenate(proteins)
    AA = np.concatenate(AA)
    AA_p = np.concatenate(AA_p)
    a_AA = np.concatenate(a_AA)
    accessibility_df = pd.DataFrame({'protein_id': proteins,
                                     'AA': AA, 'position': AA_p})
    accessibility_df[f'nAA_{max_dist}_{max_angle}_{use_pae}'] = a_AA
    return(accessibility_df)


@numba.njit
def smooth_score(score: np.ndarray,
                 half_window: int
                 ) -> np.ndarray:
    """
    Get an average value for each position in a score array, considering all
    values within a window that spans up to half_window positions before and
    after a given target position.

    Parameters
    ----------
    score : np.ndarray
        Array of numeric score values.
    half_window : int
        Integer specifying the number of positions to consider both before and
        after a given target position.

    Returns
    -------
    : np.ndarray
        Array of smoothed score values.
    """
    smooth_score = []
    for i in range(len(score)):
        low_window_bound = i - half_window
        if low_window_bound < 0:
            low_window_bound = 0
        high_window_bound = i + half_window
        if high_window_bound > len(score):
            high_window_bound = len(score)
        window_score = score[low_window_bound: high_window_bound + 1]
        window_mean = np.mean(window_score)
        smooth_score.append(window_mean)
    return np.array(smooth_score)


def get_smooth_score(df: pd.DataFrame,
                     scores: np.ndarray,
                     half_windows: list,
                     ) -> pd.DataFrame:
    """
    Select columns in a dataframe and smooth the values per protein based on a
    provided window.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with AlphaFold annotations, as generated by
        format_alphafold_data.
    scores : np.ndarray
        Array of column names in the dataframe that should be smoothed.
    half_windows : list
        List of one or more integers specifying the number of positions
        to consider both before and after a given target position.

    Returns
    -------
    : pd.DataFrame
        Copy of the input dataframe with additional columns containing the
        smoothed scores.
    """
    df_out = []
    for df_prot in partition_df_by_prots(df):
        for score in scores:
            for w in half_windows:
                df_prot[f"{score}_smooth{w}"] = smooth_score(
                    score=df_prot[score].values,
                    half_window=w)
        df_out.append(df_prot)
    df_out = pd.concat(df_out)
    return df_out


@numba.njit
def get_avg_3d_dist(idx_list: np.ndarray,  # as before, technically not a list but an array. Rename?
                    coord: np.ndarray,
                    position: np.ndarray,
                    error_dist: np.ndarray,
                    metric: str = 'mean',
                    error_operation: str = 'minus',
                    average_aa_size: float = 3.5,
                    ) -> float:
    """
    Get average 3D distance between a group of amino acids.

    Parameters
    ----------
    idx_list : np.ndarray
        Array of amino acid indeces.
    coord: np.ndarray
        Array of 3D coordinates of alpha carbon atoms across different
        amino acids.
    position : np.ndarray
        Array of amino acid positions.
    error_dist: : np.ndarray
        Matrix of paired aligned errors of AlphaFold across all amino acids in
        a protein qequence.
    metric : str
        Metric to aggregate distances across all pairs for a given amino acid.
        'mean' or 'min' can be chosen. Default is 'mean'.
    error_operation : str
        Metric to include paired aligned error in the distance calculation.
        'minus' or 'plus' can be chosen. Default is 'minus'.
    average_aa_size : float
        Average size of an AA.
        Default is 3.5 Å

    Returns
    -------
    : float
        Average 3D distance between all selected amino acids.
    """
    if metric not in ['mean', 'min']:
        raise ValueError('Select mean or min as metric.')
    if error_operation not in ['minus', 'plus']:
        raise ValueError('Select minus or plus as error_operation.')
    metric_dist = []
    for x1 in idx_list:
        all_dist = []
        for x2 in idx_list:
            if x1 != x2:
                dist_i = get_3d_dist(
                    coordinate_array_1=coord,
                    coordinate_array_2=coord,
                    idx_1=x1,
                    idx_2=x2)
                error_i = get_paired_error(
                    position=position,
                    error_dist=error_dist,
                    idx_1=x1,
                    idx_2=x2)
                if error_operation == 'minus':
                    dist_error_i = dist_i - error_i
                    if dist_error_i < average_aa_size:
                        dist_error_i = average_aa_size
                    all_dist.append(dist_error_i)
                elif error_operation == 'plus':
                    dist_error_i = dist_i + error_i
                    nAA_diff = abs(position[x1] - position[x2])
                    nAA_dist = nAA_diff * average_aa_size
                    if dist_error_i > nAA_dist:
                        all_dist.append(nAA_dist)
                    else:
                        all_dist.append(dist_error_i)
        # Probably the 5 lines below can be optimized, but likely not worth
        # the speed improvement?
        all_dist_d = np.array(all_dist)
        if metric == 'mean':
            m_d = np.mean(all_dist_d)
        elif metric == 'min':
            m_d = np.min(all_dist_d)
        metric_dist.append(m_d)
    metric_dist = np.array(metric_dist)
    avg_metric_dist = np.mean(metric_dist)
    return(avg_metric_dist)


@numba.njit
def get_avg_1d_dist(idx_list: np.ndarray,
                    position: np.ndarray,
                    metric: str = 'mean'
                    ) -> float:
    """
    Get average 1D distance between a group of amino acids.

    Parameters
    ----------
    idx_list : np.ndarray
        Array of amino acid indeces.
    position : np.ndarray
        Array of amino acid positions.
    metric : str
        Metric to aggregate distances across all pairs for a given amino acid.
        'mean' or 'min' can be chosen. Default is 'mean'.

    Returns
    -------
    : float
        Average 1D distance between all selected amino acids.
    """

    if metric not in ['mean', 'min']:
        raise ValueError('Select mean or min as metric.')
    metric_dist = []
    for x1 in idx_list:
        all_dist = []
        for x2 in idx_list:
            if x1 != x2:
                all_dist.append(abs(position[x1] - position[x2]))
        all_dist_d = np.array(all_dist)
        if metric == 'mean':
            m_d = np.mean(all_dist_d)
        elif metric == 'min':
            m_d = np.min(all_dist_d)
        metric_dist.append(m_d)
    metric_dist = np.array(metric_dist)
    avg_min_dist = np.mean(metric_dist)
    return(avg_min_dist)


def get_proximity_pvals(df: pd.DataFrame,
                        ptm_types: np.ndarray,
                        ptm_site_dict: dict,
                        error_dir: str,
                        filename_format: str = "pae_{}.hdf",
                        per_site_metric: str = 'mean',
                        error_operation: str = 'minus',
                        n_random: int = 10000,
                        random_seed: int = 44  # should obviously be 42;) Might mess up your testing though
                        ) -> pd.DataFrame:
    """
    Get proximity p-values for selected PTMs.

    Parameters
    ----------
    df : pd.DataFrame
        pd.DataFrame of formatted AlphaFold data across various proteins.
    ptm_types: np.ndarray
        Array of PTM modifications for which to perform the proximity analysis.
    ptm_site_dict : dict
        Dictionary containing the possible amino acid sites for each PTM.
    error_dir : str
        Path to the directory where the hdf files containing the matrices of
        paired aligned errors of AlphaFold are stored.
    filename_format : str
        The file name of the pae files saved by download_alphafold_pae.
        The brackets {} are replaced by a protein name from the proteins list.
        Default is 'pae_{}.hdf'.
    per_site_metric : str
        Metric to aggregate distances across all pairs for a given amino acid.
        'mean' or 'min' can be chosen. Default is 'mean'.
    error_operation : str
        Metric to include paired aligned error in the distance calculation.
        'minus' or 'plus' can be chosen. Default is 'minus'.
    n_random : int
        Number of random permutations to perform. Default is 10'000.
        The higher the number of permutations, the more confidence the analysis
        can achieve. However, a very high number of permutations increases
        processing time. No fewer than 1'000 permutations should be used.
    random_seed : int
        Random seed for the analysis. Default is 44.

    Returns
    -------
    : pd.DataFrame
        Dataframe reporting 3D and 1D proximity p-values for each protein and
        selected PTM.
    """
    random.seed(random_seed)
    proteins = list()
    ptm_type = list()
    n_ptms = list()
    pvals_3d = list()
    pvals_1d = list()
    for df_prot in partition_df_by_prots(df):
        protein_accession = df_prot.protein_id.values[0]
        for ptm_i in ptm_types:
            acc_aa = ptm_site_dict[ptm_i]
            df_ptm_prot = df_prot[df_prot.AA.isin(acc_aa)].reset_index(drop=True)
            n_aa_mod = np.sum(df_ptm_prot[ptm_i])
            n_aa_all = df_ptm_prot.shape[0]
            if ((n_aa_mod >= 2) & (n_aa_mod < n_aa_all)):
                with h5py.File(os.path.join(
                    error_dir,
                    filename_format.format(protein_accession))) as hdf_root:
                    error_dist = hdf_root['dist'][...]
                size = int(np.sqrt(len(error_dist)))
                error_dist = error_dist.reshape(size, size)
                # subset to ptm possible positions
                # calculate real distance
                real_idx = df_ptm_prot.index[df_ptm_prot[ptm_i] == 1].tolist()
                avg_dist_3d = get_avg_3d_dist(
                    idx_list=np.array(real_idx),
                    coord=np.vstack([
                        df_ptm_prot["x_coord_ca"].values,
                        df_ptm_prot["y_coord_ca"].values,
                        df_ptm_prot["z_coord_ca"].values]).T,
                    position=df_ptm_prot["position"].values,
                    error_dist=error_dist,
                    metric=per_site_metric,
                    error_operation=error_operation)
                avg_dist_1d = get_avg_1d_dist(
                    idx_list=np.array(real_idx),
                    position=df_ptm_prot["position"].values,
                    metric=per_site_metric)
                # get background distribution
                rand_idx_list = [np.array(random.sample(
                    range(n_aa_all), len(real_idx))) for i in range(n_random)]
                rand_avg_dist_3d = [get_avg_3d_dist(
                    idx_list=idx_l,
                    coord=np.vstack([
                        df_ptm_prot["x_coord_ca"].values,
                        df_ptm_prot["y_coord_ca"].values,
                        df_ptm_prot["z_coord_ca"].values]).T,
                    position=df_ptm_prot["position"].values,
                    error_dist=error_dist,
                    metric=per_site_metric,
                    error_operation=error_operation) for idx_l in rand_idx_list]
                rand_avg_dist_1d = [get_avg_1d_dist(
                    idx_list=idx_l,
                    position=df_ptm_prot["position"].values,
                    metric=per_site_metric) for idx_l in rand_idx_list]
                # get empirical p-values
                pvalue_3d = np.sum(np.array(rand_avg_dist_3d) <= avg_dist_3d)/n_random
                pvalue_1d = np.sum(np.array(rand_avg_dist_1d) <= avg_dist_1d)/n_random
                # If this is a slow step, there are several ways to still
                # optimize this I think.
                # Especially the creation of 10000 elements in both a list and
                # array seem concerning to me.
                # Probably a >> 10 fold is still possible here.
            else:
                pvalue_3d = np.nan
                pvalue_1d = np.nan
            pvals_3d.append(pvalue_3d)
            pvals_1d.append(pvalue_1d)
            n_ptms.append(n_aa_mod)
            proteins.append(protein_accession)
            ptm_type.append(ptm_i)
    res_df = pd.DataFrame({'protein_id': proteins,
                           'ptm': ptm_type,
                           'n_ptms': n_ptms,
                           'pvalue_1d': pvals_1d,
                           'pvalue_3d': pvals_3d})
    res_df_noNan = res_df.dropna(subset=['pvalue_3d','pvalue_1d']).reset_index(drop=True)
    # Why are these then stored explicitly above? # This was to know which IDs these are.
    res_df_noNan['pvalue_1d_adj_bh'] = statsmodels.stats.multitest.multipletests(pvals=res_df_noNan.pvalue_1d, alpha=0.1, method='fdr_bh')[1]
    res_df_noNan['pvalue_3d_adj_bh'] = statsmodels.stats.multitest.multipletests(pvals=res_df_noNan.pvalue_3d, alpha=0.1, method='fdr_bh')[1]
    return(res_df_noNan)


def perform_enrichment_analysis(df: pd.DataFrame,
                                ptm_types: list,
                                rois: list,
                                quality_cutoffs: list,
                                ptm_site_dict: dict,
                                multiple_testing: bool = True) -> pd.DataFrame:
    """
    Get enrichment p-values for selected PTMs acros regions of interest (ROIs).

    Parameters
    ----------
    df : pd.DataFrame
        pd.DataFrame of formatted AlphaFold data across various proteins.
    ptm_types: list
        List of PTM modifications for which to perform the enrichment analysis.
    rois : list
        List of regions of interest (one hot encoded columns in df) for which
        to perform the enrichment analysis.
    quality_cutoffs : list
        List of quality cutoffs (AlphaFold pLDDDT values) to filter for.
    ptm_site_dict : dict
        Dictionary containing the possible amino acid sites for each PTM.
    multiple_testing : bool
        Bool if multiple hypothesis testing correction should be performed.
        Default is 'True'.

    Returns
    -------
    : pd.DataFrame
        Dataframe reporting p-values for the enrichment of all selected
        ptm_types across selected rois.
    """

    enrichment = []
    for q_cut in quality_cutoffs:
        # Is quality_cutoffs expected to be a big list?
        # If so, we can still optimize the function below reasonably I think...
        seq_ann_qcut = df[df.quality >= q_cut]
        for ptm in ptm_types:
            seq_ann_qcut_aa = seq_ann_qcut[seq_ann_qcut.AA.isin(ptm_site_dict[ptm])]
            for roi in rois:
                seq_ann_qcut_aa_roi1 = seq_ann_qcut_aa[roi] == 1
                seq_ann_qcut_aa_roi0 = seq_ann_qcut_aa[roi] == 0
                seq_ann_qcut_aa_ptm1 = seq_ann_qcut_aa[ptm] == 1
                seq_ann_qcut_aa_ptm0 = seq_ann_qcut_aa[ptm] == 0
                n_ptm_in_roi = seq_ann_qcut_aa[seq_ann_qcut_aa_roi1 & seq_ann_qcut_aa_ptm1].shape[0]
                n_ptm_not_in_roi = seq_ann_qcut_aa[seq_ann_qcut_aa_roi0 & seq_ann_qcut_aa_ptm1].shape[0]
                n_naked_in_roi = seq_ann_qcut_aa[seq_ann_qcut_aa_roi1 & seq_ann_qcut_aa_ptm0].shape[0]
                n_naked_not_in_roi = seq_ann_qcut_aa[seq_ann_qcut_aa_roi0 & seq_ann_qcut_aa_ptm0].shape[0]
                fisher_table = np.array([[n_ptm_in_roi, n_naked_in_roi], [n_ptm_not_in_roi, n_naked_not_in_roi]])
                oddsr, p = scipy.stats.fisher_exact(fisher_table,
                                                    alternative='two-sided')
                res = pd.DataFrame({'quality_cutoff': [q_cut],
                                    'ptm': [ptm],
                                    'roi': [roi],
                                    'n_aa_ptm':  np.sum(seq_ann_qcut_aa_ptm1),
                                    'n_aa_roi':  np.sum(seq_ann_qcut_aa_roi1),
                                    'n_ptm_in_roi': n_ptm_in_roi,
                                    'n_ptm_not_in_roi': n_ptm_not_in_roi,
                                    'n_naked_in_roi': n_naked_in_roi,
                                    'n_naked_not_in_roi': n_naked_not_in_roi,
                                    'oddsr': [oddsr],
                                    'p': [p]})
                enrichment.append(res)
    enrichment_df = pd.concat(enrichment)
    if multiple_testing:
        enrichment_df['p_adj_bf'] = statsmodels.stats.multitest.multipletests(
            pvals=enrichment_df.p, alpha=0.01, method='bonferroni')[1]
        enrichment_df['p_adj_bh'] = statsmodels.stats.multitest.multipletests(
            pvals=enrichment_df.p, alpha=0.01, method='fdr_bh')[1]
    return(enrichment_df)


def perform_enrichment_analysis_per_protein(
    df: pd.DataFrame,
    ptm_types: list,
    rois: list,
    quality_cutoffs: list,
    ptm_site_dict: dict
) -> pd.DataFrame:
    """
    Get per protein enrichment p-values for selected PTMs acros regions of
    interest (ROIs).

    Parameters
    ----------
    df : pd.DataFrame
        pd.DataFrame of formatted AlphaFold data across various proteins.
    ptm_types: list
        List of PTM modifications for which to perform the enrichment analysis.
    rois : list
        List of regions of interest (one hot encoded columns in df) for which
        to perform the enrichment analysis.
    quality_cutoffs : list
        List of quality cutoffs (AlphaFold pLDDDT values) to filter for.
    ptm_site_dict : dict
        Dictionary containing the possible amino acid sites for each PTM.

    Returns
    -------
    : pd.DataFrame
        Dataframe reporting p-values for the enrichment of all selected
        ptm_types across selected rois on a per protein basis.
    """
    enrichment_list = list()
    for df_prot in partition_df_by_prots(df):
        protein_accession = df_prot.protein_id.values[0]
        res = perform_enrichment_analysis(df=df_prot,
                                          ptm_types=ptm_types,
                                          rois=rois,
                                          quality_cutoffs=quality_cutoffs,
                                          ptm_site_dict=ptm_site_dict,
                                          multiple_testing=False)
        res.insert(loc=0, column='protein_id', value=np.repeat(
            protein_accession, res.shape[0]))
        enrichment_list.append(res)
    enrichment_per_protein = pd.concat(enrichment_list)
    enrichment_per_protein = enrichment_per_protein[(enrichment_per_protein.n_aa_ptm >= 2) & (enrichment_per_protein.n_aa_roi >= enrichment_per_protein.n_aa_ptm)]
    enrichment_per_protein.reset_index(drop=True, inplace=True)
    enrichment_per_protein['p_adj_bf'] = statsmodels.stats.multitest.multipletests(
        pvals=enrichment_per_protein.p, alpha=0.01, method='bonferroni')[1]
    enrichment_per_protein['p_adj_bh'] = statsmodels.stats.multitest.multipletests(
        pvals=enrichment_per_protein.p, alpha=0.01, method='fdr_bh')[1]
    return enrichment_per_protein


def find_idr_pattern(
    idr_list: list,
    min_structured_length: int = 100,
    max_unstructured_length: int = 30
) -> tuple:
    """
    Find short intrinsically disordered regions.

    Parameters
    ----------
    idr_list : list
        Nested list specifying the binary IDR condition and its length.
        For example: [[1,10],[0,30],[1,5]].
    min_structured_length : int
        Integer specifying the minimum number of amino acids in flanking
        structured regions.
    max_unstructured_length : int
        Integer specifying the maximum number of amino acids in the short
        intrinsically unstructured regions.

    Returns
    -------
    : tuple
        (bool, list) If a pattern was found and the list of start end end
        positions of short IDRs.
    """
    window = np.array([0, 1, 2])
    i = 0
    pattern = False
    pos_list = list()
    while i < (len(idr_list) - 2):
        window_i = window + i
        if idr_list[window_i[0]][0] == 0:
            if idr_list[window_i[0]][1] >= min_structured_length:
                if idr_list[window_i[1]][1] <= max_unstructured_length:
                    if idr_list[window_i[2]][1] >= min_structured_length:
                        pattern = True
                        idr_start = np.sum([x[1] for x in idr_list[0: i + 1]]) + 1
                        idr_end = idr_start + idr_list[i + 1][1] - 1
                        pos_list.append([idr_start, idr_end])
        i += 1
    return pattern, pos_list


def annotate_proteins_with_idr_pattern(
    df: pd.DataFrame,
    min_structured_length: int = 100,
    max_unstructured_length: int = 30
) -> pd.DataFrame:
    """
    Find short intrinsically disordered regions.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with AlphaFold annotations.
    min_structured_length : int
        Integer specifying the minimum number of amino acids in flanking
        structured regions.
    max_unstructured_length : int
        Integer specifying the maximum number of amino acids in the short
        intrinsically unstructured regions.

    Returns
    -------
    : pd.DataFrame
        Input dataframe with an additional column 'flexible_pattern'.
    """

    res_out = list()
    proteins = list()
    loop_pattern = list()
    pattern_position = list()
    for df_prot in partition_df_by_prots(df):
        df_prot['flexible_pattern'] = 0
        protein_accession = df_prot.protein_id.values[0]
        idr_list = [[k, len(list(g))] for k, g in groupby(df_prot.IDR.values)]
        pattern, pos_list = find_idr_pattern(
            idr_list,
            min_structured_length=min_structured_length,
            max_unstructured_length=max_unstructured_length)
        pattern_position_list = list()
        if pattern:
            proteins.append(protein_accession)
            loop_pattern.append(pattern)
            pattern_position.append(pos_list)

            pattern_position_list = pattern_position_list + [list(np.arange(p[0], p[1] + 1)) for p in pos_list]
            pattern_position_list = [item for sublist in pattern_position_list for item in sublist]

            selected_locations = np.flatnonzero(df_prot.position.isin(
                pattern_position_list))
            df_prot.loc[selected_locations, 'flexible_pattern'] = 1
        res_out.append(df_prot)
    res_out = pd.concat(res_out)
    return res_out


@numba.njit
def extend_flexible_pattern(
    pattern: np.ndarray,
    window: int
) -> np.ndarray:
    """
    This function takes an array of binary values (0 or 1) and extends streches
    of 1s to either side by the provided window.

    Parameters
    ----------
    pattern : np.ndarray
        Array of binary pattern values.
    window : int
        Integer specifying the number of positions to consider both before
        and after the provided pattern.

    Returns
    -------
    : np.ndarray
        Array with of binary values, where streches of 1s in the input array
        were extended to both sides.
    """
    extended_pattern = []
    for i in range(len(pattern)):
        low_window_bound = i - window
        if low_window_bound < 0:
            low_window_bound = 0
        high_window_bound = i + window
        if high_window_bound > len(pattern):
            high_window_bound = len(pattern)
        window_patterns = pattern[low_window_bound: high_window_bound + 1]
        window_max = np.max(window_patterns)
        extended_pattern.append(window_max)
    return np.array(extended_pattern)


def get_extended_flexible_pattern(
    df: pd.DataFrame,
    patterns: np.ndarray,
    windows: list,
) -> pd.DataFrame:
    """
    Select columns in a dataframe for which to extend the pattern by the
    provided window.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with AlphaFold annotations.
    patterns : np.ndarray
        Array of column names in the dataframe with binary values that should
        be extended.
    windows : list
        List of one or more integers specifying the number of positions
        to consider both before and after a pattern.

    Returns
    -------
    : pd.DataFrame
        Input dataframe with additional columns containing the extended
        patterns.
    """
    df_out = []
    for df_prot in partition_df_by_prots(df):
        for pattern in patterns:
            for w in windows:
                df_prot[f'{pattern}_extended_{w}'] = extend_flexible_pattern(
                    pattern=df_prot[pattern].values,
                    window=w)
        df_out.append(df_prot)
    df_out = pd.concat(df_out)
    return df_out


#  This function could be numba compatible
def calculate_distances_between_ptms(
    background_idx_list: list,
    target_aa_idx: np.ndarray,
    coords: np.ndarray,
    positions: np.ndarray,
    error_dist: np.ndarray
) -> [list, list]:
    """
    Calculate the distances from a target amino acid to a list of background
    amino acids.

    Parameters
    ----------
    background_idx_list : list
        List of amino acid indices that make up the background.
    target_aa_idx : np.ndarray
        Array of target amino acid indices.
    coords : np.ndarray
        Array of 3D coordinates of alpha carbon atoms across different
        amino acids.
    positions : np.ndarray
        Array of amino acid positions from which to choose the specific indeces.
    error_dist: : np.ndarray
        Matrix of paired aligned errors of AlphaFold across all amino acids
        in a protein qequence.

    Returns
    -------
    : [list, list]
        List of 3D distance results and list of 1D distance results
    """
    distance_res = list()
    distance_1D_res = list()
    for idx_list in background_idx_list:
        aa_dist_list = list()
        aa_1D_dist_list = list()
        for i in idx_list:
            aa_dist = list()
            aa_1D_dist = list()
            for aa_i in target_aa_idx:
                aa_dist_i = get_3d_dist(
                    coordinate_array_1=coords,
                    coordinate_array_2=coords,
                    idx_1=i,
                    idx_2=aa_i)
                aa_error_i = get_paired_error(
                    position=positions,
                    error_dist=error_dist,
                    idx_1=i,
                    idx_2=aa_i)
                aa_dist.append(aa_dist_i+aa_error_i)
                aa_1D_dist.append(abs(positions[i]-positions[aa_i]))
            aa_dist_list.append(aa_dist)
            aa_1D_dist_list.append(aa_1D_dist)
        distance_res.append(aa_dist_list)
        distance_1D_res.append(aa_1D_dist_list)
    return distance_res, distance_1D_res


def get_ptm_distance_list(
    df: pd.DataFrame,
    ptm_target: str,
    ptm_background: str,
    ptm_dict: dict,
    error_dir: str,
    filename_format: str = "pae_{}.hdf",
    n_random: int = 10000,
    random_seed: int = 44,
) -> [list, list, list]:
    """
    Extract a lists of 3D and 1D distances between target amino acids and a
    random background.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with AlphaFold annotations.
    ptm_target : str
        String specifying the PTM type for which you want to evaluate if it
        is in colocalizing with the background.
    ptm_background : str
        String specifying the PTM type that is used as background.
    ptm_dict : dict
        Dictionary containing the possible amino acid sites for each PTM.
    error_dir : str
        Path to the directory where the hdf files containing the matrices of
        paired aligned errors of AlphaFold are stored.
    filename_format : str
        The file name of the pae files saved by download_alphafold_pae.
        The brackets {} are replaced by a protein name from the proteins list.
        Default is 'pae_{}.hdf'.
    n_random : int
        Number of random permutations to perform. Default is 10'000.
        The higher the number of permutations, the more confidence the analysis
        can achieve. However, a very high number of permutations increases
        processing time. No fewer than 1'000 permutations should be used.
    random_seed : int
        Random seed for the analysis. Default is 44.

    Returns
    -------
    : [list, list, list]
        List of 3D distances, list of 1D distances and
        list of modified indices.
    """
    random.seed(random_seed)
    prot_distances = list()
    prot_distances_1D = list()
    prot_mod_idx = list()
    for df_prot in partition_df_by_prots(df):
        protein_accession = df_prot.protein_id.values[0]
        if error_dir is not None:
            with h5py.File(
                os.path.join(
                    error_dir,
                    filename_format.format(protein_accession))
                    ) as hdf_root:
                error_dist = hdf_root['dist'][...]
            size = int(np.sqrt(len(error_dist)))
            error_dist = error_dist.reshape(size, size)
        else:
            error_dist = np.zeros((df_prot.shape[0], df_prot.shape[0]))
        # amino acid residues of background PTM
        background_aa = ptm_dict[ptm_background]
        # indices of background_aa
        background_idx = list(np.flatnonzero(df_prot.AA.isin(background_aa)))
        # number of observed background modifications
        n_aa_background_mod = np.sum(df_prot[ptm_background] == 1)
        if n_aa_background_mod >= 1:
            # indices of observed background PTMs
            real_background_idx = df_prot.index[df_prot[ptm_background] == 1].tolist()
            # list of random index lists for background PTMs
            # @TODO: probably slowish due to making lists of 10000 elements,
            # perhaps this can be avoided
            background_idx_list = [random.sample(
                background_idx,
                len(real_background_idx)) for i in np.arange(0, n_random)]
            # Combine real and random backround list with the real indices at
            # position 0
            background_idx_list.insert(0,real_background_idx)
            # amino acid residues of target PTM
            target_aa = ptm_dict[ptm_target]
            # indices of target_aa
            target_aa_idx = list(np.flatnonzero(df_prot.AA.isin(target_aa)))
            # indices of observed target PTMs
            target_mod_idx = df_prot.index[df_prot[ptm_target] == 1].tolist()
            # index of observed PTMs within index list of all target_aa
            target_aa_idx_mod_idx = [i for i in np.arange(len(target_aa_idx)) if target_aa_idx[i] in target_mod_idx]
            distance_res, distance_1D_res = calculate_distances_between_ptms(
                background_idx_list=np.array(background_idx_list),
                target_aa_idx=np.array(target_aa_idx),
                coords=np.vstack([
                    df_prot.x_coord_ca.values,
                    df_prot.y_coord_ca.values,
                    df_prot.z_coord_ca.values]).T,
                positions=df_prot.position.values,
                error_dist=error_dist)
            prot_distances.append(distance_res)
            prot_distances_1D.append(distance_1D_res)
            prot_mod_idx.append(target_aa_idx_mod_idx)
    return prot_distances, prot_distances_1D, prot_mod_idx


#  This function could be numba compatible
def get_mod_ptm_fraction(
    distances: list,
    mod_idx: list,
    min_dist: int,
    max_dist: int
) -> float:
    """
    Calculate the fraction of modified PTM acceptor residues within
    a distance range.

    Parameters
    ----------
    distances: list
        List of 1D or 3D distances.
    mod_idx: lists
        List of indices to select which distances to consider.
    min_dist: int
        Minimum distance of the bin.
    max_dist: int
        Maximum distance of the bin.

    Returns
    -------
    : float
        Fraction of modified PTM acceptor residues within
        the provided distance range.
    """
    n_aa = [0]*len(distances[0])
    n_aa_mod = [0]*len(distances[0])
    for idx, p in enumerate(distances):
        rand_count = 0
        for rand in p:
            for back in rand:
                n_aa[rand_count] += len([i for i in back if ((i > min_dist) & (i <= max_dist))])
                mod_back = [back[i] for i in mod_idx[idx]]
                n_aa_mod[rand_count] += len([i for i in mod_back if ((i > min_dist) & (i <= max_dist))])
            rand_count += 1
    mod_fraction = [mod/aa if aa>0 else np.nan for aa,mod in zip(n_aa, n_aa_mod)]
    return mod_fraction


def evaluate_ptm_colocalization(
    df: pd.DataFrame,
    ptm_target: str,
    ptm_types: list,
    ptm_dict: dict,
    pae_dir: str,
    filename_format: str = "pae_{}.hdf",
    n_random: int = 5,
    random_seed: int = 44,
    min_dist: float = -0.01,
    max_dist: float = 35,
    dist_step: float = 5
) -> pd.DataFrame:
    """
    Evaluate for a given target PTM type if modifications preferentially occur
    closer to the provided background PTM types than expected by chance or at
    distance bins that are further away.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with AlphaFold annotations.
    ptm_target : str
        String specifying the PTM type for which you want to evaluate if it
        is in colocalizing with the background.
    ptm_types : list of strings
        List of strings specifying the PTM types that should be used as
        background. If "self" is included, this means that the ptm_target
        is used also as backround modification.
    ptm_dict : dict
        Dictionary containing the possible amino acid sites for each PTM.
    pae_dir : str
        Path to the directory where the hdf files containing the matrices of
        paired aligned errors of AlphaFold are stored.
    filename_format : str
        The file name of the pae files saved by download_alphafold_pae.
        The brackets {} are replaced by a protein name from the proteins list.
        Default is 'pae_{}.hdf'.
    n_random : int
        Number of random permutations to perform. Default is 10'000.
        The higher the number of permutations, the more confidence the analysis
        can achieve. However, a very high number of permutations increases
        processing time. No fewer than 1'000 permutations should be used.
    random_seed : int
        Random seed for the analysis. Default is 44.
    min_dist : float
        Minimum distance to consider.
        Default is 0, meaning that the target amino acid is included itself.
    max_dist : float
        Maximum distance to consider.
        Default is 35.
    dist_step : float
        Stepsize for distance bins between min_dist and max_dist.
        Default is 5.

    Returns
    -------
    : pd.DataFrame
        Dataframe with following columns: 'context', 'ptm_types', 'cutoff',
        'std_random_fraction', 'variable', 'value'
    """
    distance_cutoffs = np.arange(min_dist, max_dist, dist_step)
    # might want to change to np.linspace above
    cutoff_list = list()
    ptm_list = list()
    real_fraction_3D = list()
    mean_random_fraction_3D = list()
    std_random_fraction_3D = list()
    ttest_pval_3D = list()
    real_fraction_1D = list()
    mean_random_fraction_1D = list()
    std_random_fraction_1D = list()
    ttest_pval_1D = list()
    for ptm_type in ptm_types:
        if ptm_target == 'self':
            ptm_target = ptm_type
        distances_3D, distances_1D, mod_idx = get_ptm_distance_list(
            df=df,
            ptm_target=ptm_target,
            ptm_background=ptm_type,
            ptm_dict=ptm_dict,
            error_dir=pae_dir,
            filename_format=filename_format,
            n_random=n_random,
            random_seed=random_seed
        )
        dist_i = 0
        for dist_cut in distance_cutoffs:
            ptm_list.append(ptm_type)
            cutoff_list.append(dist_cut+dist_step)
            if dist_i == 0:
                # make sure that the minimum is incuded
                dist_step_mod = 0.001
            else:
                dist_step_mod = 0
            mod_fraction_3D = get_mod_ptm_fraction(
                distances_3D,
                mod_idx,
                min_dist=dist_cut-dist_step_mod,
                max_dist=dist_cut+dist_step)
            real_fraction_3D.append(mod_fraction_3D[0])
            mean_random_fraction_3D.append(np.mean(mod_fraction_3D[1:]))
            std_random_fraction_3D.append(np.std(mod_fraction_3D[1:]))
            ttest_pval_3D.append(scipy.stats.ttest_1samp(mod_fraction_3D[1:], mod_fraction_3D[0]).pvalue)
            mod_fraction_1D = get_mod_ptm_fraction(
                distances_1D,
                mod_idx,
                min_dist=dist_cut-dist_step_mod,
                max_dist=dist_cut+dist_step)
            real_fraction_1D.append(mod_fraction_1D[0])
            mean_random_fraction_1D.append(np.mean(mod_fraction_1D[1:]))
            std_random_fraction_1D.append(np.std(mod_fraction_1D[1:]))
            ttest_pval_1D.append(scipy.stats.ttest_1samp(mod_fraction_1D[1:], mod_fraction_1D[0]).pvalue)
            dist_i += 1
    res_df_3D = pd.DataFrame({
        'context': np.repeat('3D', len(cutoff_list)),
        'cutoff': cutoff_list,
        'ptm_types': ptm_list,
        'Observed': real_fraction_3D,
        'Random sampling': mean_random_fraction_3D,
        'std_random_fraction': std_random_fraction_3D,
        'pvalue': ttest_pval_3D})
    res_df_1D = pd.DataFrame({
        'context': np.repeat('1D', len(cutoff_list)),
        'cutoff': cutoff_list,
        'ptm_types': ptm_list,
        'Observed': real_fraction_1D,
        'Random sampling': mean_random_fraction_1D,
        'std_random_fraction': std_random_fraction_1D,
        'pvalue': ttest_pval_1D})
    res_df_3D = res_df_3D.melt(
        id_vars=["context", "ptm_types", "cutoff", "std_random_fraction","pvalue"])
    res_df_1D = res_df_1D.melt(
        id_vars=["context", "ptm_types", "cutoff", "std_random_fraction","pvalue"])
    res_df = pd.concat([res_df_3D, res_df_1D])
    res_df['std_random_fraction'] = np.where(
        res_df.variable == 'Observed', 0, res_df.std_random_fraction)
    return res_df


def extract_motifs_in_proteome(
    alphafold_df: pd.DataFrame,
    motif_df: pd.DataFrame
):
    """
    Function to find occurences of short linear motifs in the proteome.

    Parameters
    ----------
    alphafold_df : pd.DataFrame
        Dataframe with AlphaFold annotations.
    motif_df : pd.DataFrame
        Dataframe with following columns: 'enzyme', 'motif', 'mod_pos'.

    Returns
    -------
    : pd.DataFrame
        Dataframe containing information about short linear motifs in the
        proteome. Following columns are privided: 'protein_id', 'enzyme',
        'motif','position','AA','motif_start','motif_end','sequence_window'
    """
    proteins = list()
    enzyme_list = list()
    motif_list = list()
    site_list = list()
    start_list = list()
    end_list = list()
    AA_list = list()
    sequence_window_list = list()
    for df_prot in partition_df_by_prots(alphafold_df):
        df_prot['flexible_pattern'] = 0
        protein_accession = df_prot.protein_id.values[0]
        sequence = ''.join(df_prot.AA)
        for i in np.arange(0, motif_df.shape[0]):
            for j in re.finditer(motif_df.motif.values[i], sequence):
                proteins.append(protein_accession)
                enzyme_list.append(motif_df.enzyme.values[i])
                motif_list.append(motif_df.motif.values[i])
                site_list.append(j.start() + motif_df.mod_pos.values[i] + 1)
                start_list.append(j.start() + 1)
                end_list.append(j.end())
                AA_list.append(sequence[j.start() + motif_df.mod_pos.values[i]])
                sequence_window_list.append(sequence[(j.start() + motif_df.mod_pos.values[i] - 10): (j.start() + motif_df.mod_pos.values[i] + 10)])
    motif_res = pd.DataFrame({
        'protein_id': proteins,
        'enzyme': enzyme_list,
        'motif': motif_list,
        'position': site_list,
        'AA': AA_list,
        'motif_start': start_list,
        'motif_end': end_list,
        'sequence_window': sequence_window_list})
    return motif_res


def import_ptms_for_structuremap(
    file: str,
    organism: str
) -> pd.DataFrame:
    """
    Function to import PTM datasets.

    Parameters
    ----------
    file : str
        Path to the PTM dataset to load.
        This can be processed by AlphaPept, Spectronaut, MaxQuant, DIA-NN or
        FragPipe.
    organism : str
        Organism for which a fasta file should be imported.

    Returns
    -------
    : pd.DataFrame
        Dataframe with PTM information. It contains following columns:
        protein_id: a unique UniProt identifier;
        AA: the one letter amino acid abbreviation of the PTM acceptor;
        position: the sequence position of the PTM acceptor
        (the first amino acid has position 1);
        <PTM types>: N columns for N different PTM types where 1 indicates that
        the PTM is present at the given amino acid postition
        and 0 indicates no modification
    """
    try:
        from alphamap.organisms_data import import_fasta
        from alphamap.importing import import_data
        from alphamap.preprocessing import format_input_data
    except ModuleNotFoundError:
        raise ModuleNotFoundError(f"Please install alphamap. Subsequently install pandas==1.4.0.")
    fasta_in = import_fasta('Human')
    df = import_data(file)
    df = format_input_data(df=df,
                           fasta=fasta_in,
                           modification_exp=r'\[.*?\]')
    ptm_df = df.explode(['PTMsites', 'PTMtypes'])
    ptm_df = ptm_df.dropna(subset=['PTMsites', 'PTMtypes'])
    ptm_df = ptm_df.astype({'PTMsites': 'int32'})
    ptm_df["AA"] = ptm_df.apply(
        lambda x: x["naked_sequence"][x["PTMsites"]],
        axis=1)
    ptm_df["position"] = ptm_df.apply(
        lambda x: x["start"]+x["PTMsites"]+1,
        axis=1)
    ptm_df = ptm_df[["unique_protein_id", "AA", "position", "PTMtypes"]]
    ptm_df = pd.get_dummies(
        ptm_df, prefix="", prefix_sep='', columns=["PTMtypes"])
    ptm_df = ptm_df.rename(columns={"unique_protein_id": "protein_id"})
    ptm_df = ptm_df.groupby(['protein_id', 'AA', 'position'])
    ptm_df = ptm_df.max()
    ptm_df = ptm_df.reset_index()
    ptm_df = ptm_df.drop_duplicates()
    ptm_df = ptm_df.reset_index(drop=True)
    return ptm_df


def format_for_3Dviz(
    df: pd.DataFrame,
    ptm_dataset: str
) -> pd.DataFrame:
    df_mod = df[["protein_id", "AA", "position", ptm_dataset]]
    df_mod = df_mod.rename(columns={"protein_id": "unique_protein_id",
                                    "AA": "modified_sequence",
                                    "position": "start"})
    df_mod["modified_sequence"] = [mod+"_"+str(i) for i,mod in enumerate(df_mod["modified_sequence"])]
    df_mod["all_protein_ids"] = df_mod["unique_protein_id"]
    df_mod["PTMsites"] = 0
    df_mod["start"] = df_mod["start"]-1
    df_mod["end"] = df_mod["start"]
    df_mod["PTMsites"] = [[i] for i in df_mod["PTMsites"]]
    df_mod = df_mod[df_mod[ptm_dataset] == 1]
    df_mod["marker_symbol"] = 1
    df_mod["PTMtypes"] = [[ptm_dataset] for i in df_mod["PTMsites"]]
    df_mod = df_mod.dropna(subset=['PTMtypes']).reset_index(drop=True)
    return df_mod
