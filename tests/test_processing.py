#!python -m unittest tests.test_processing
import numba
import numpy as np
import pandas as pd
import tqdm
import h5py
import random
import statsmodels.stats.multitest
import urllib.request, json
import os
import socket
import re
import Bio.PDB.MMCIF2Dict
from itertools import groupby
import unittest

from structuremap.processing import download_alphafold_cif, \
    download_alphafold_pae, \
    format_alphafold_data, \
    get_3d_dist, \
    rotate_vector_around_axis, \
    get_angle, \
    get_paired_error, \
    get_neighbors, \
    annotate_accessibility, \
    smooth_score, \
    get_smooth_score, \
    get_avg_3d_dist, \
    get_avg_1d_dist, \
    find_idr_pattern, \
    annotate_proteins_with_idr_pattern, \
    extend_flexible_pattern, \
    get_extended_flexible_pattern


THIS_FOLDER = os.path.dirname(__file__)
TEST_FOLDER = os.path.join(
    f"{os.path.dirname(THIS_FOLDER)}",
    "data",
    "test_files",
)

class TestProcessing(unittest.TestCase):
    def test_download_alphafold_cif(self, ):
        valid, invalid, existing = download_alphafold_cif(
            proteins=['O15552','Q5VSL9','Q7Z6M3','O15552yy'],
            out_folder=TEST_FOLDER)

        np.testing.assert_equal(valid, np.array(['Q5VSL9']))
        np.testing.assert_equal(invalid, np.array(['O15552yy']))
        np.testing.assert_equal(existing, np.array(['O15552','Q7Z6M3']))

        os.remove(
            os.path.join(
                TEST_FOLDER,
                'Q5VSL9.cif'
            )
        )

    def test_download_alphafold_pae(self, ):
        valid, invalid, existing = download_alphafold_pae(
            proteins=['O15552','Q5VSL9','Q7Z6M3','O15552yy'],
            out_folder=TEST_FOLDER)

        np.testing.assert_equal(valid, np.array(['Q5VSL9']))
        np.testing.assert_equal(invalid, np.array(['O15552yy']))
        np.testing.assert_equal(existing, np.array(['O15552','Q7Z6M3']))

        os.remove(
            os.path.join(
                TEST_FOLDER,
                'pae_Q5VSL9.hdf'
            )
        )

    def test_format_alphafold_data(self, ):
        alphafold_formatted = format_alphafold_data(
            directory=TEST_FOLDER, protein_ids=["Q7Z6M3","O15552"])

        alphafold_formatted_ini = pd.read_csv(
            os.path.join(
                TEST_FOLDER,
                'test_alphafold_annotation.csv'
            )
        )

        pd.testing.assert_frame_equal(alphafold_formatted, alphafold_formatted_ini)

    def test_get_3d_dist(self, ):
        x = np.array([1.1,1.1,1.1,1.1,5.1])
        y = np.array([1.1,2.1,3.1,1.1,10.1])
        z = np.array([1.1,3.1,5.1,1.1,4.1])
        coordinate_array = np.vstack([x,y,z]).T
        np.testing.assert_equal(2.236068, np.round(get_3d_dist(coordinate_array, coordinate_array, 0, 1), decimals=6))
        np.testing.assert_equal(4.472136, np.round(get_3d_dist(coordinate_array, coordinate_array, 0, 2), decimals=6))
        np.testing.assert_equal(4.472136, np.round(get_3d_dist(coordinate_array, coordinate_array, 2, 0), decimals=6))

    from scipy.spatial.transform import Rotation as R

    def rotate_vector_around_axis_scipy(self, vector, axis, theta):
        theta = np.radians(theta)
        axis_norm = axis / np.linalg.norm(axis)
        r = R.from_rotvec(theta * axis_norm)
        return(r.apply(vector))

    def test_rotate_vector_around_axis(self, ):
        v = np.array([3.0, 5.0, 0.0])
        a = np.array([4.0, 4.0, 1.0])
        t = 90

        res_real = rotate_vector_around_axis(v, a, t)
        res_scipy = self.rotate_vector_around_axis_scipy(v, a, t)

        np.testing.assert_almost_equal(res_real, res_scipy, decimal=10)

    def test_get_angle(self, ):
        x_a = np.array([1.1,1.1,1.1])
        y_a = np.array([1.1,2.1,-3.1])
        z_a = np.array([1.1,3.1,5.1])
        x_b = np.array([1.5,np.nan,1.5])
        y_b = np.array([1.5,2.5,3.5])
        z_b = np.array([1.5,3.5,5.5])
        x_c = np.array([1.5,1.5,10.6])
        y_c = np.array([1.5,2.5,11.6])
        z_c = np.array([1.5,3.5,5.6])
        x_n = np.array([4.5,1.8,1.5])
        y_n = np.array([40.5,7.8,3.5])
        z_n = np.array([3.5,3.8,5.5])

        coordinate_array_a = np.vstack([x_a,y_a,z_a]).T
        coordinate_array_b = np.vstack([x_b,y_b,z_b]).T
        coordinate_array_c = np.vstack([x_c,y_c,z_c]).T
        coordinate_array_n = np.vstack([x_n,y_n,z_n]).T

        np.testing.assert_equal(39.231520,
                                np.round(get_angle(coordinate_array_a, coordinate_array_b,
                                                   coordinate_array_c, coordinate_array_n,
                                                   0, 1), decimals=6))
        np.testing.assert_equal(91.140756,
                                np.round(get_angle(coordinate_array_a, coordinate_array_b,
                                                   coordinate_array_c, coordinate_array_n,
                                                   0, 2), decimals=6))
        np.testing.assert_equal(47.168228,
                                np.round(get_angle(coordinate_array_a, coordinate_array_b,
                                                   coordinate_array_c, coordinate_array_n,
                                                   2, 0), decimals=6))

        # test gly
        np.testing.assert_equal(93.985035,
                                np.round(get_angle(coordinate_array_a, coordinate_array_b,
                                                   coordinate_array_c, coordinate_array_n,
                                                   1, 2), decimals=6))

    def test_get_paired_error(self, ):
        pos = np.array([1,2,3])
        error = np.array([[0,2,10],[1,0,5],[10,4,0]])
        np.testing.assert_equal(2, get_paired_error(pos, error, 0,1))
        np.testing.assert_equal(0, get_paired_error(pos, error, 2,2))

        pos = np.array([1,3])
        np.testing.assert_equal(10, get_paired_error(pos, error, 0,1))

    def test_get_neighbors(self, ):
        idxl = np.array([0,1,2])
        x_a = np.array([1.1,1.1,1.1])
        y_a = np.array([1.1,2.1,-3.1])
        z_a = np.array([1.1,3.1,5.1])
        x_b = np.array([1.5,np.nan,1.5])
        y_b = np.array([1.5,2.5,3.5])
        z_b = np.array([1.5,3.5,5.5])
        x_c = np.array([1.5,1.5,10.6])
        y_c = np.array([1.5,2.5,11.6])
        z_c = np.array([1.5,3.5,5.6])
        x_n = np.array([4.5,1.8,1.5])
        y_n = np.array([40.5,7.8,3.5])
        z_n = np.array([3.5,3.8,5.5])

        coordinate_array_a = np.vstack([x_a,y_a,z_a]).T
        coordinate_array_b = np.vstack([x_b,y_b,z_b]).T
        coordinate_array_c = np.vstack([x_c,y_c,z_c]).T
        coordinate_array_n = np.vstack([x_n,y_n,z_n]).T

        pos=np.array([1,2,3])
        error = np.array([[0,2,10],[1,0,5],[10,4,0]])

        np.testing.assert_equal(np.array([1, 0, 0]),
                                get_neighbors(idxl, coordinate_array_a, coordinate_array_b,
                                              coordinate_array_c, coordinate_array_n,
                                              pos, error, 5, 40))
        np.testing.assert_equal(np.array([1, 1, 0]),
                                get_neighbors(idxl, coordinate_array_a, coordinate_array_b,
                                              coordinate_array_c, coordinate_array_n,
                                              pos, error, 5, 150))
        np.testing.assert_equal(np.array([2, 2, 2]),
                                get_neighbors(idxl, coordinate_array_a, coordinate_array_b,
                                              coordinate_array_c, coordinate_array_n,
                                              pos, error, 50, 140))

    from Bio import PDB

    def test_annotate_accessibility(self, ):
        radius = 12.0

        alphafold_annotation = pd.read_csv(
            os.path.join(
                TEST_FOLDER,
                'test_alphafold_annotation.csv'
            )
        )

        res_accessability = annotate_accessibility(
            df=alphafold_annotation[alphafold_annotation.protein_id=="Q7Z6M3"],
            max_dist=12,
            max_angle=90,
            error_dir=None)

        # comparison to https://biopython.org/docs/dev/api/Bio.PDB.HSExposure.html#Bio.PDB.HSExposure.HSExposureCB
        with open(
            os.path.join(
                TEST_FOLDER,
                'Q7Z6M3.pdb'
            )
        ) as pdbfile:
            import Bio.PDB as PDB
            p=PDB.PDBParser()
            s=p.get_structure('X', pdbfile)
            m=s[0]
            hse=PDB.HSExposureCB(m, radius)
            residue_list=PDB.Selection.unfold_entities(m,'R')
            res_hse = []
            for r in residue_list:
                res_hse.append(r.xtra['EXP_HSE_B_U'])

        np.testing.assert_equal(np.array(res_hse), res_accessability.nAA_12_90_nopae.values)

        # @ToDo: test with actual error_dir

    def test_smooth_score(self, ):
        np.testing.assert_equal(np.array([1.5, 2. , 3. , 4. , 4.5]),smooth_score(score=np.array([1,2,3,4,5]), half_window=1))

    def test_get_smooth_score(self, ):
        testdata = pd.DataFrame({'protein_id':[1,1,1,1,1,1,2,2,2,2,2,2],
                                 'protein_number':[1,1,1,1,1,1,2,2,2,2,2,2],
                                 'position':[1,2,3,4,5,6,1,2,3,4,5,6],
                                 'score':[1,2,3,4,5,6,7,8,9,10,11,12],
                                 'score_2':[10,20,30,40,50,60,70,80,90,100,110,120]})
        test_res = get_smooth_score(testdata, np.array(['score','score_2']), [1])
        np.testing.assert_equal([1.5,2,3,4,5,5.5,7.5,8,9,10,11,11.5], test_res.score_smooth1.values)
        np.testing.assert_equal([15,20,30,40,50,55,75,80,90,100,110,115], test_res.score_2_smooth1.values)

    def test_get_avg_3d_dist(self, ):
        x = np.array([1.1,1.1,1.1,1.1,1.1,1.1])
        y = np.array([1.1,2.1,3.1,1.1,10.1,20.1])
        z = np.array([1.1,3.1,5.1,10.1,11.1,12.1])
        pos = np.array([1,2,3,4,5,6])
        error = np.array([[0,2,10,2,3,4],[1,0,5,3,2,9],[10,4,0,3,6,7],[10,4,5,0,6,7],[10,4,5,3,0,7],[10,4,0,3,6,0]])

        coordinate_array = np.vstack([x,y,z]).T

        np.testing.assert_equal(6.953624, np.round(get_avg_3d_dist(np.array([0,4]), coordinate_array, pos, error), decimals=6))
        np.testing.assert_equal(3.5, np.round(get_avg_3d_dist(np.array([0,2]), coordinate_array, pos, error), decimals=6))

        np.testing.assert_equal(5.586336, np.round(get_avg_3d_dist(np.array([0,3,4]), coordinate_array, pos, error), decimals=6))
        np.testing.assert_equal(4.503003, np.round(get_avg_3d_dist(np.array([0,3,4]), coordinate_array, pos, error, metric='min'), decimals=6))

        np.testing.assert_equal(14, np.round(get_avg_3d_dist(np.array([0,4]), coordinate_array, pos, error, error_operation='plus'), decimals=6))
        error = 0.1*error
        np.testing.assert_equal(13.876812, np.round(get_avg_3d_dist(np.array([0,4]), coordinate_array, pos, error, error_operation='plus'), decimals=6))

        x = np.array([1.1,1.1,1.1,1.1])
        y = np.array([1.1,1.1,10.1,20.1])
        z = np.array([1.1,10.1,11.1,12.1])
        pos = np.array([1,4,5,6])
        error = np.array([[0,2,10,2,3,4],[1,0,5,3,2,9],[10,4,0,3,6,7],[10,4,5,0,6,7],[10,4,5,3,0,7],[10,4,0,3,6,0]])

        coordinate_array = np.vstack([x,y,z]).T

        np.testing.assert_equal(6.953624, np.round(get_avg_3d_dist(np.array([0,2]), coordinate_array, pos, error), decimals=6))

    def test_get_avg_1d_dist(self, ):
        pos = np.array([1,2,3,4,5,6])
        np.testing.assert_equal(4, np.round(get_avg_1d_dist(np.array([0,4]), pos), decimals=6))
        np.testing.assert_equal(2.666667, np.round(get_avg_1d_dist(np.array([0,3,4]), pos), decimals=6))
        np.testing.assert_equal(1.666667, np.round(get_avg_1d_dist(np.array([0,3,4]), pos, metric='min'), decimals=6))

        pos = np.array([1,4,5,6])
        np.testing.assert_equal(4, np.round(get_avg_1d_dist(np.array([0,2]), pos), decimals=6))
        np.testing.assert_equal(2.666667, np.round(get_avg_1d_dist(np.array([0,1,2]), pos), decimals=6))

    def test_find_idr_pattern(self, ):
        assert find_idr_pattern(idr_list = [[0,300],[1,10],[0,500],[1,500]])[0] == True
        assert find_idr_pattern(idr_list = [[0,300],[1,50],[0,500]])[0] == False
        assert find_idr_pattern(idr_list = [[0,50],[0,50],[1,50],[0,500]])[0] == False
        assert find_idr_pattern(idr_list = [[0,30],[0,300],[1,50],[0,50]])[0] == False
        assert find_idr_pattern(idr_list = [[0,30]])[0] == False

        assert find_idr_pattern(idr_list = [[0,300],[1,10],[0,500],[1,500]])[1][0][0] == [301]
        assert find_idr_pattern(idr_list = [[0,300],[1,10],[0,500],[1,500]])[1][0][1] == [310]
        assert find_idr_pattern(idr_list = [[1,10],[0,300],[1,10],[0,500],[1,500]])[1][0][0] == [311]
        assert find_idr_pattern(idr_list = [[1,10],[0,300],[1,10],[0,500],[1,500]])[1][0][1] == [320]

    def test_annotate_proteins_with_idr_pattern(self, ):
        testdata = pd.DataFrame({'protein_id':[1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2],
                                 'protein_number':[1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2],
                                 'position':[1,2,3,4,5,6,7,8,9,10,11,12,1,2,3,4,5,6],
                                 'IDR':[0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,1,0,0]})
        test_res = annotate_proteins_with_idr_pattern(testdata, 3, 3)
        np.testing.assert_equal([0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                list(test_res.flexible_pattern.values))

    def test_extend_flexible_pattern(self, ):
        np.testing.assert_equal(np.array([1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0]),
                                extend_flexible_pattern(np.array([1,1,1,0,0,0,0,1,1,0,0,0,0]),1))

    def test_get_extended_flexible_pattern(self, ):
        testdata = pd.DataFrame({'protein_id':[1,1,1,1,1,1,2,2,2,2,2,2],
                                 'protein_number':[1,1,1,1,1,1,2,2,2,2,2,2],
                                 'position':[1,2,3,4,5,6,1,2,3,4,5,6],
                                 'score':[1,1,0,0,0,1,1,1,0,0,0,0],
                                 'score_2':[0,0,0,0,0,0,0,0,0,0,0,1]})
        test_res = get_extended_flexible_pattern(testdata, np.array(['score','score_2']), [1])
        np.testing.assert_equal([1,1,1,0,1,1,1,1,1,0,0,0], test_res.score_extended_1.values)
        test_res = get_extended_flexible_pattern(testdata, np.array(['score','score_2']), [2])
        np.testing.assert_equal([1,1,1,1,1,1,1,1,1,1,0,0], test_res.score_extended_2.values)
        np.testing.assert_equal([0,0,0,0,0,0,0,0,0,1,1,1], test_res.score_2_extended_2.values)

if __name__ == "__main__":
    unittest.main()
