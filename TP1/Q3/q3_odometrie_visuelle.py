import os
import numpy as np
import cv2
import scipy as sp

# infos : pour la librairie bokeh, installer la version 2.4.0 pour des questions de compatibilité
from lib.visualization import plotting
from lib.visualization.video import play_trip

# Pour faire afficher raisonnablement bien les images très allongées du dataset KITTI
def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


def apply_Lowe_test(matches, threshold):
    good = []
    try:
        for m, n in matches:
            # utiliser les valeur de m.distance et n.distance pour faire un test de Lowe
            # En ce moment, tous les matchs vont être retournés.
            if m.distance/n.distance < threshold:
                good.append(m)
    except ValueError:
        pass
    return good

# fonction pour faire la vérification mutuelle
def apply_cross_check(matchs_1_vers_2, matchs_2_vers_1):
    # ici il faut utiliser les .trainIdx et .queryIdx des matchs pour trouver
    # ceux qui sont mutuellement voisins.
    mutual_good = []

    return mutual_good


class VisualOdometry():
    def __init__(self, data_dir):
        self.nombre_de_features_ORB = 1000
        self.K, self.P = self._load_calib(os.path.join(data_dir, 'calib.txt'))
        self.gt_poses = self._load_poses(os.path.join(data_dir,"poses.txt"))
        self.images = self._load_images(os.path.join(data_dir,"image_l"))
        self.orb = cv2.ORB_create(self.nombre_de_features_ORB)
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)

    @staticmethod
    def _load_calib(filepath):
        """
        Loads the calibration of the camera
        Parameters
        ----------
        filepath (str): The file path to the camera file

        Returns
        -------
        K (ndarray): Intrinsic parameters
        P (ndarray): Projection matrix
        """
        with open(filepath, 'r') as f:
            params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
            P = np.reshape(params, (3, 4))
            K = P[0:3, 0:3]
        return K, P

    @staticmethod
    def _load_poses(filepath):
        """
        Loads the GT poses

        Parameters
        ----------
        filepath (str): The file path to the poses file

        Returns
        -------
        poses (ndarray): The GT poses
        """
        poses = []
        with open(filepath, 'r') as f:
            for line in f.readlines():
                T = np.fromstring(line, dtype=np.float64, sep=' ')
                T = T.reshape(3, 4)
                T = np.vstack((T, [0, 0, 0, 1]))
                poses.append(T)
        return poses

    @staticmethod
    def _load_images(filepath):
        """
        Loads the images

        Parameters
        ----------
        filepath (str): The file path to image dir

        Returns
        -------
        images (list): grayscale images
        """
        image_paths = [os.path.join(filepath, file) for file in sorted(os.listdir(filepath))]
        return [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]

    @staticmethod
    def _form_transf(R, t):
        """
        Makes a transformation matrix from the given rotation matrix and translation vector

        Parameters
        ----------
        R (ndarray): The rotation matrix
        t (list): The translation vector

        Returns
        -------
        T (ndarray): The transformation matrix
        """
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def get_matches(self, i):
        """
        This function detect and compute keypoints and descriptors from the i-1'th and i'th image using the class orb object

        Parameters
        ----------
        i (int): The current frame

        Returns
        -------
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image
        """
        # des paramètres que vous pouvez modifier
        #self.orb.setFastThreshold(10)
        #self.orb.setEdgeThreshold(15)

        # Trouver les keypoints et les descripteurs avec ORB
        kp1, des1 = self.orb.detectAndCompute(self.images[i - 1], None)
        kp2, des2 = self.orb.detectAndCompute(self.images[i], None)

        # Faire l'appariement
        matches1 = self.flann.knnMatch(des1, des2, k=2)
        matches2 = self.flann.knnMatch(des2, des1, k=2)

        # Appliquer le test de Lowe avec un ratio de 0.7
        good = apply_Lowe_test(matches1, 0.7)

        draw_params = dict(matchColor = -1, # draw matches in green color
                 singlePointColor = None,
                 matchesMask = None, # draw only inliers
                 flags = 2)

        # choisir entre montrer les keypoints ou les matchs
        show_keypoints = 1
        if (show_keypoints):
            img3 = cv2.drawKeypoints(self.images[i], kp1,None)
        else:
            img3 = cv2.drawMatches(self.images[i], kp1, self.images[i-1],kp2, good ,None,**draw_params)
        resize = ResizeWithAspectRatio(img3, width=1280)  # Resize by width OR
        cv2.imshow('resize', resize)
        cv2.waitKey(100)

        # Récupérer les coordonnées des keypoints des matchs conservés.
        q1 = np.float32([kp1[m.queryIdx].pt for m in good])
        q2 = np.float32([kp2[m.trainIdx].pt for m in good])
        return q1, q2

    def get_pose(self, q1, q2):
        """
        Calculates the transformation matrix

        Parameters
        ----------
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image

        Returns
        -------
        transformation_matrix (ndarray): The transformation matrix
        """
        # Essential matrix
        E, _ = cv2.findEssentialMat(q1, q2, self.K, threshold=1)
        pts, R, t, mask_pose = cv2.recoverPose(E,q1,q2,self.K)

        t = t.flatten()

        # Get transformation matrix
        transformation_matrix = self._form_transf(R, np.squeeze(t))
        return transformation_matrix


def main():
    data_dir = "KITTI_sequence_1"  # Try KITTI_sequence_2 too
    vo = VisualOdometry(data_dir)

    gt_path = []
    estimated_path = []
    #for i, gt_pose in enumerate(tqdm(vo.gt_poses, unit="pose")):
    for i, gt_pose in enumerate(vo.gt_poses):
        if i == 0:
            cur_pose = gt_pose
        else:
            q1, q2 = vo.get_matches(i)
            transf = vo.get_pose(q1, q2)
            cur_pose = np.matmul(cur_pose, np.linalg.inv(transf))
        gt_path.append((gt_pose[0, 3], gt_pose[2, 3]))
        estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))
    plotting.visualize_paths(gt_path, estimated_path, "Visual Odometry", file_out=os.path.basename(data_dir) + ".html")


if __name__ == "__main__":
    main()
