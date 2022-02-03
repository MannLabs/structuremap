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

# external
import numba
import numpy as np
import pandas as pd
import tqdm
import h5py
import statsmodels.stats.multitest
import Bio.PDB.MMCIF2Dict
import scipy.stats

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
        List (or any other iterable) of UniProt protein accessions for which to download the structures.
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
    Function to download paired aligned errors (pae) for protein structures predicted by AlphaFold.

    Parameters
    ----------
    proteins : list
        List (or any other iterable) of UniProt protein accessions for which to download the structures.
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
                with urllib.request.urlopen(name_in) as url:
                    data = json.loads(url.read().decode())
                dist = np.array(data[0]['distance'])
                data_list = [('dist', dist)]
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

            if ((protein_id in protein_ids) or (len(protein_ids)==0)):

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
                            df['position'].between(start_idx[i],
                                                   end_idx[i]),
                                                   note[i],
                                                   df['secondary_structure'])

                alphafold_annotation_l.append(df)


    alphafold_annotation = pd.concat(alphafold_annotation_l)
    alphafold_annotation = alphafold_annotation.sort_values(
        by=['protein_number', 'position']).reset_index(drop=True)

    alphafold_annotation['structure_group'] = [re.sub('_.*', '', i) for i in alphafold_annotation['secondary_structure']]
    
    str_oh = pd.get_dummies(alphafold_annotation['structure_group'], dtype='int64')
    alphafold_annotation = alphafold_annotation.join(str_oh)
    return(alphafold_annotation)


@numba.njit
def get_3d_dist(coordinate_array_1: np.ndarray,
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
    coordinate_array_2 : np.ndarray)
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
def rotate_vector_around_axis(vector: np.ndarray,
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

    rotation_matrix = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                                [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                                [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

    rotated_vector = np.dot(rotation_matrix, vector)

    return rotated_vector


@numba.njit
def get_gly_vector(coord_a: np.ndarray,
                   coord_c: np.ndarray,
                   coord_n: np.ndarray,
                   idx_1: int,
                   theta: float = -120) -> np.ndarray:
    """
    Return a pseudo vector Ca -> Cb for a Glycine residue.
    The pseudo vector is centered at the origin and the
    Ccoord=N coord rotated over -120 degrees
    along the CA-C axis (see Bio.PDB package).

    Parameters
    ----------
    coord_a : np.ndarray
        Array of 3D coordinates of alpha carbon atoms across different amino acids.
    coord_c : np.ndarray
        Array of 3D coordinates of carboxy carbon atoms across different amino acids.
    coord_n : np.ndarray
        Array of 3D coordinates of amino nitrogen atoms across different amino acids.
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
def get_angle(coord_a: np.ndarray,
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
        Array of 3D coordinates of alpha carbon atoms across different amino acids.
    coord_b : np.ndarray
        Array of 3D coordinates of beta carbon atoms across different amino acids.
    coord_c : np.ndarray
        Array of 3D coordinates of carboxy carbon atoms across different amino acids.
    coord_n : np.ndarray
        Array of 3D coordinates of amino nitrogen atoms across different amino acids.
    idx_1 : int
        Integer to select a first amino acid in the coordinate arrays.
    idx_2 : int
        Integer to select a second amino acid in the coordinate arrays.

    Returns
    -------
    : float
        Angle between the side chain of the first amino acid and a second amino acid.
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
def get_paired_error(position: np.ndarray,
                     error_dist: np.ndarray,
                     idx_1: int,
                     idx_2: int
                     ) -> float:
    """
    Extract paired aligned error of AlphaFold from a complete error matrix (error_dist)
    at specific sequence positions.

    Parameters
    ----------
    position : np.ndarray
        Array of amino acid positions from which to choose specific indeces.
    error_dist: : np.ndarray
        Matrix of paired aligned errors of AlphaFold across all amino acids in a protein qequence.
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
def get_neighbors(idx_list: np.ndarray, # Technically this is not a list and it could/should be renamed.
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
        Array of 3D coordinates of alpha carbon atoms across different amino acids.
    coord_b : np.ndarray
        Array of 3D coordinates of beta carbon atoms across different amino acids.
    coord_c : np.ndarray
        Array of 3D coordinates of carboxy carbon atoms across different amino acids.
    coord_n : np.ndarray
        Array of 3D coordinates of amino nitrogen atoms across different amino acids.
    position : np.ndarray
        Array of amino acid positions.
    error_dist: : np.ndarray
        Matrix of paired aligned errors of AlphaFold across all amino acids in a protein qequence.
    max_dist : float
        Float specifying the maximum distance between two amino acids.
    max_angle : float
        Float specifying the maximum angle (in degrees) between two amino acids.

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
def find_end(label: int, start_index: int, values: np.ndarray) -> int:
    """Find when the label changes.

    This assumes a sorted values array

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


def annotate_accessibility(df: pd.DataFrame,
                           max_dist: float,
                           max_angle: float,
                           error_dir: str
                           ) -> pd.DataFrame:
    """
    Half sphere exposure as calculated in https://onlinelibrary.wiley.com/doi/10.1002/prot.20379
    but with paired aligned error metric included.

    Parameters
    ----------
    df : pd.DataFrame
        pd.DataFrame of formatted AlphaFold data across various proteins.
        TODO: from which internal function? Note that this is good to mention for other arguments in other functions as well to show how functions are connected
    max_dist : float
        Float specifying the maximum distance between two amino acids.
    max_angle : float
        Float specifying the maximum angle (in degrees) between two amino acids.
    error_dir: : str
        Path to the directory where the hdf files containing the matrices of
        paired aligned errors of AlphaFold are stored.

    Returns
    -------
    : pd.DataFrame
        Dataframe repporting the number of neighboring amino acids at the specified
        maximum distance and angle per protein, amino acid and position.
    """
    # idxs = np.argsort(df.protein_number.values)
    # df_sorted = df['protein_number', 'position'][idxs]
    df_sorted = df.sort_values(by=['protein_number', 'position']).reset_index(drop=True)
    # Is the df not already sorted when it is inputted?
    # Quick check possible with: pd.Series.is_monotonic_increasing

    unique_proteins = df_sorted.protein_number.unique()

    end = 0

    proteins = list()
    AA = list()
    AA_p = list()
    a_AA = list()

    for protein_i in tqdm.tqdm(unique_proteins):

        start = end
        end = find_end(protein_i, end, df_sorted.protein_number.values)

        df_prot = df_sorted[start:end].reset_index(drop=True)

        protein_accession = df_prot.protein_id.values[0]

        if error_dir is not None:
            with h5py.File(r''+error_dir+'/pae_'+protein_accession+'.hdf', 'r') as hdf_root:
                # Always use os.path.join. Windows and linux / are not necessarily the same!
                # Using + for string concatenation should be considered deprecated.
                # Use f-strings or "".join functions.
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
          # Alternatively, it might be faster to use e.g. df[["x", "y", "z"]].values as pandas might force this into a view rather than a new array
            position=df_prot.position.values.astype(np.int64),  # should this cast not be enforced on the base df already?
            error_dist=error_dist,
            max_dist=max_dist,
            max_angle=max_angle)

        proteins.append(df_prot.protein_id.values)
        # using numeracal prot_numbers might be better.
        # In general it is good practive to reduce strings/objects in arrays/dfs as much possible. Especially try to avoid repetetion of such types and just use indices and a reference array. Rarely do you need this actual values anyways.
        AA.append(df_prot.AA.values)
        AA_p.append(df_prot.position.values)
        a_AA.append(res_a)

    proteins = np.concatenate(proteins)
    AA = np.concatenate(AA)
    AA_p = np.concatenate(AA_p)

    a_AA = np.concatenate(a_AA)

    accessibility_df = pd.DataFrame({'protein_id': proteins, 'AA': AA, 'position': AA_p})
    accessibility_df['nAA_' + str(max_dist) + '_' + str(max_angle) + '_' + use_pae] = a_AA
    # f_string? Again, consider using indices instead

    # glycine_vol = 60.1
    # spherical_sector_volume = ((2*(np.pi)*(max_dist**3))/3)*(1-np.cos(np.deg2rad(max_angle)))
    # n_max_AA = spherical_sector_volume/glycine_vol

    # accessibility_df['nAA_'+str(max_dist)+'_'+str(max_angle)+'_'+use_pae+'_rel'] = a_AA/n_max_AA

    return(accessibility_df)


@numba.njit
def smooth_score(score: np.ndarray,
                 half_window: int
                 ) -> np.ndarray:
    """
    Get an average value for each position in a score array, considering all values
    within a window that spans up to half_window positions before and after a given
    target position.

    Parameters
    ----------
    score : np.ndarray
        Array of numeric score values.
    half_window : int
        Integer specifying the number of positions to consider both before and after
        a given target position.

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
    Select columns in a dataframe and smooth the values per protein based on a provided window.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with AlphaFold annotations. TODO: which format, obtained from format_alphafold_data function?
    scores : np.ndarray
        Array of column names in the dataframe that should be smoothed.
    half_windows : list
        List of one or more integers specifying the number of positions
        to consider both before and after a given target position.

    Returns
    -------
    : pd.DataFrame
        Copy of the input dataframe with additional columns containing the smoothed scores.
    """
    df_sorted = df.sort_values(by=['protein_number', 'position']).reset_index(drop=True)
    unique_proteins = df_sorted.protein_number.unique()
    end = 0

    df_out = []

    for protein_i in tqdm.tqdm(unique_proteins):

        start = end
        end = find_end(protein_i, end, df_sorted.protein_number.values)

        df_prot = df_sorted[start:end].reset_index(drop=True)

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
        Array of 3D coordinates of alpha carbon atoms across different amino acids.
    position : np.ndarray
        Array of amino acid positions.
    error_dist: : np.ndarray
        Matrix of paired aligned errors of AlphaFold across all amino acids in a protein qequence.
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
                    if dist_error_i < 0:  # Should this not be: if dist_error_i < average_aa_size:
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

        # Probably the 5 lines below can be optimized, but likely not worth the speed improvement?
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
    per_site_metric : str
        Metric to aggregate distances across all pairs for a given amino acid.
        'mean' or 'min' can be chosen. Default is 'mean'.
    error_operation : str
        Metric to include paired aligned error in the distance calculation.
        'minus' or 'plus' can be chosen. Default is 'minus'.
    n_random : int
        Number of random permutations to perform. Default is 10'000.
        TODO: give recommendations to de/increase and the effect on quality/speed?
    random_seed : int
        Random seed for the analysis. Default is 44.

    Returns
    -------
    : pd.DataFrame
        Dataframe reporting 3D and 1D proximity p-values for each protein and selected PTM.
    """
    random.seed(random_seed)

    df_sorted = df.sort_values(by=['protein_number', 'position']).reset_index(drop=True)
    unique_proteins = df_sorted.protein_number.unique()
    end = 0

    proteins = list()
    ptm_type = list()
    n_ptms = list()
    pvals_3d = list()
    pvals_1d = list()

    for protein_i in tqdm.tqdm(unique_proteins):

        start = end
        end = find_end(protein_i, end, df_sorted.protein_number.values)

        df_prot = df_sorted[start:end].reset_index(drop=True)
        protein_accession = df_prot.protein_id.values[0]

        for ptm_i in ptm_types:
            acc_aa = ptm_site_dict[ptm_i]
            df_ptm_prot = df_prot[df_prot.AA.isin(acc_aa)].reset_index(drop=True)

            n_aa_mod = np.sum(df_ptm_prot[ptm_i])
            n_aa_all = df_ptm_prot.shape[0]

            if ((n_aa_mod >= 2) & (n_aa_mod < n_aa_all)):

                with h5py.File(r'' + error_dir + '/pae_' + protein_accession + '.hdf', 'r') as hdf_root:
                    error_dist = hdf_root['dist'][...]
                size = int(np.sqrt(len(error_dist)))
                error_dist = error_dist.reshape(size, size)

                # subset to ptm possible positions
                # calculate real distance
                real_idx = df_ptm_prot.index[df_ptm_prot[ptm_i] == 1].tolist()
                # print(real_idx)
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
                rand_idx_list = [np.array(random.sample(range(n_aa_all), len(real_idx))) for i in range(n_random)]
                # print(rand_idx_list)
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
                # If this is a slow step, there are several ways to still optimize this I think. Especially the creation of 10000 elements in both a list and array seem concerning to me. Probably a >> 10 fold is still possible here.

            else:

                pvalue_3d = np.nan
                pvalue_1d = np.nan

            pvals_3d.append(pvalue_3d)
            pvals_1d.append(pvalue_1d)
            n_ptms.append(n_aa_mod)
            proteins.append(protein_accession)
            ptm_type.append(ptm_i)


    res_df = pd.DataFrame({'protein_id': proteins, 'ptm': ptm_type, 'n_ptms': n_ptms, 'pvalue_1d': pvals_1d, 'pvalue_3d': pvals_3d})

    res_df_noNan = res_df.dropna(subset=['pvalue_3d','pvalue_1d']).reset_index(drop=True)
    # Why are these then stored explicitly above?

    res_df_noNan['pvalue_1d_adj_bh'] = statsmodels.stats.multitest.multipletests(pvals=res_df_noNan.pvalue_1d, alpha=0.1, method='fdr_bh')[1]
    res_df_noNan['pvalue_3d_adj_bh'] = statsmodels.stats.multitest.multipletests(pvals=res_df_noNan.pvalue_3d, alpha=0.1, method='fdr_bh')[1]
    # res_df_noNan['pvalue_1d_adj_bf'] = statsmodels.stats.multitest.multipletests(pvals=res_df_noNan.pvalue_1d, alpha=0.1, method='bonferroni')[1]
    # res_df_noNan['pvalue_3d_adj_bf'] = statsmodels.stats.multitest.multipletests(pvals=res_df_noNan.pvalue_3d, alpha=0.1, method='bonferroni')[1]

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
        Dataframe reporting p-values for the enrichment of all selected ptm_types
        across selected rois.
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
                oddsr, p = scipy.stats.fisher_exact(fisher_table, alternative='two-sided')

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
        enrichment_df['p_adj_bf'] = statsmodels.stats.multitest.multipletests(pvals=enrichment_df.p, alpha=0.01, method='bonferroni')[1]
        enrichment_df['p_adj_bh'] = statsmodels.stats.multitest.multipletests(pvals=enrichment_df.p, alpha=0.01, method='fdr_bh')[1]

    return(enrichment_df)


def perform_enrichment_analysis_per_protein(df: pd.DataFrame,
                                            ptm_types: list,
                                            rois: list,
                                            quality_cutoffs: list,
                                            ptm_site_dict: dict) -> pd.DataFrame:

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
        Dataframe reporting p-values for the enrichment of all selected ptm_types
        across selected rois on a per protein basis.
    """

    df_sorted = df.sort_values(by=['protein_number', 'position']).reset_index(drop=True)

    unique_proteins = df_sorted.protein_number.unique()

    end = 0

    enrichment_list = list()

    for protein_i in tqdm.tqdm(unique_proteins):

        start = end
        end = find_end(protein_i, end, df_sorted.protein_number.values)
        df_prot = df_sorted[start:end].reset_index(drop=True)
        protein_accession = df_prot.protein_id.values[0]

        res = perform_enrichment_analysis(df=df_prot,
                                          ptm_types=ptm_types,
                                          rois=rois,
                                          quality_cutoffs=quality_cutoffs,
                                          ptm_site_dict=ptm_site_dict,
                                          multiple_testing=False)
        res.insert(loc=0, column='protein_id', value=np.repeat(protein_accession, res.shape[0]))

        enrichment_list.append(res)

    enrichment_per_protein = pd.concat(enrichment_list)
    enrichment_per_protein = enrichment_per_protein[(enrichment_per_protein.n_aa_ptm >= 2) & (enrichment_per_protein.n_aa_roi >= enrichment_per_protein.n_aa_ptm)]
    enrichment_per_protein.reset_index(drop=True, inplace=True)

    enrichment_per_protein['p_adj_bf'] = statsmodels.stats.multitest.multipletests(pvals=enrichment_per_protein.p, alpha=0.01, method='bonferroni')[1]
    enrichment_per_protein['p_adj_bh'] = statsmodels.stats.multitest.multipletests(pvals=enrichment_per_protein.p, alpha=0.01, method='fdr_bh')[1]

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
        Nested list specifying the binary IDR condition and its length. For example: [[1,10],[0,30],[1,5]].
    min_structured_length : int
        Integer specifying the minimum number of amino acids in flanking structured regions.
    max_unstructured_length : int
        Integer specifying the maximum number of amino acids in the short intrinsically unstructured regions.

    Returns
    -------
    : tuple
        (bool, list) If a pattern was found and the list of start end end positions of short IDRs.
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
        Integer specifying the minimum number of amino acids in flanking structured regions.
    max_unstructured_length : int
        Integer specifying the maximum number of amino acids in the short intrinsically unstructured regions.

    Returns
    -------
    : pd.DataFrame
        Input dataframe with an additional column 'flexible_pattern'.
    """

    df_sorted = df.sort_values(by=['protein_number', 'position']).reset_index(drop=True)
    df_sorted['flexible_pattern'] = 0

    unique_proteins = df_sorted.protein_number.unique()

    end = 0

    proteins = list()
    loop_pattern = list()
    pattern_position = list()

    for protein_i in tqdm.tqdm(unique_proteins):

        start = end
        end = find_end(protein_i, end, df_sorted.protein_number.values)

        df_prot = df_sorted[start:end].reset_index(drop=True)

        protein_accession = df_prot.protein_id.values[0]

        idr_list = [[k, len(list(g))] for k, g in groupby(df_prot.IDR.values)]
        pattern, pos_list = find_idr_pattern(idr_list,
                                   min_structured_length=min_structured_length,
                                   max_unstructured_length=max_unstructured_length)

        pattern_position_list = list()
        if pattern:
            proteins.append(protein_accession)
            loop_pattern.append(pattern)
            pattern_position.append(pos_list)

            pattern_position_list = pattern_position_list + [list(np.arange(p[0], p[1] + 1)) for p in pos_list]
            pattern_position_list = [item for sublist in pattern_position_list for item in sublist]

            selected_locations = start + np.flatnonzero(df_prot.position.isin(pattern_position_list))
            df_sorted.loc[selected_locations, 'flexible_pattern'] = 1

    return df_sorted


@numba.njit
def extend_flexible_pattern(
    pattern: np.ndarray,
    window: int
) -> np.ndarray:
    """
    Extend a pattern by the specified amino acid window

    Parameters
    ----------
    pattern : np.ndarray
        Array of binary pattern values.
    window : int
        Integer specifying the number of positions to consider both before and after
        the provided pattern.

    Returns
    -------
    : np.ndarray
        Array with the extended pattern.
    """
    # Purely by documentation, I still have no idea what this function does.
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
    Select columns in a dataframe for which to extend the pattern by the provided window.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with AlphaFold annotations.
    patterns : np.ndarray
        Array of column names in the dataframe with binary values that should be extended.
    windows : list
        List of one or more integers specifying the number of positions
        to consider both before and after a pattern.

    Returns
    -------
    : pd.DataFrame
        Input dataframe with additional columns containing the extended patterns.
    """
    df_sorted = df.sort_values(by=['protein_number', 'position']).reset_index(drop=True)
    # You this very frequently. Probably it is best to do it just once and require the input df to be sorted.
    unique_proteins = df_sorted.protein_number.unique()
    end = 0

    df_out = []

    for protein_i in tqdm.tqdm(unique_proteins):

        start = end
        end = find_end(protein_i, end, df_sorted.protein_number.values)

        df_prot = df_sorted[start:end].reset_index(drop=True)
        # You often repeat this pattern to select a prot_df. Perhaps put in a single function/generator and recycle everywhere?

        for pattern in patterns:
            for w in windows:
                df_prot[pattern + '_extended_' + str(w)] = extend_flexible_pattern(
                    pattern=df_prot[pattern].values,
                    window=w)

        df_out.append(df_prot)
    df_out = pd.concat(df_out)
    return df_out
