import cv2
import numpy as np
import glob

chemins_photos = glob.glob("photos/*.jpg")
points_3d = []
points_2d= []

coins_damier = np.zeros((48,3), dtype=np.float32)
grille_indices = np.mgrid[0:8, 0:6]
points_grille= grille_indices.T.reshape(48, 2)
coins_damier[:,:2] = points_grille
taille_image = ()


for indice, chemin_photo in enumerate(chemins_photos):
    image_couleur = cv2.imread(chemin_photo)
    if image_couleur is None:
        raise FileNotFoundError("error")

    image_grise = cv2.cvtColor(image_couleur, cv2.COLOR_BGR2GRAY)

    copie_gris= image_grise.copy()
    copie_couleur = image_couleur.copy()

    trouve, coins = cv2.findChessboardCorners(image_grise, (8,6))

    if trouve == True:
        criteres = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        cv2.cornerSubPix(image_grise, coins, (11,11), (-1,-1), criteres)

        points_3d.append(coins_damier.copy())
        points_2d.append(coins.copy())
        print(len(points_3d), len(points_2d))

        hauteur_image, largeur_image = image_grise.shape
        taille_image = (largeur_image, hauteur_image)
        print("taille_image:", taille_image)
        # dessin les coins sur image grise
        cv2.drawChessboardCorners(copie_gris, (8,6), coins, trouve)
        cv2.imwrite(f"resultats/coins_gris_{indice}.jpg", copie_gris)

        # dessin les coins sur image couleur
        cv2.drawChessboardCorners(copie_couleur, (8,6), coins, trouve)

        # agrandit les cercles des coins
        for coin in coins:
            coin_x, coin_y = coin.ravel()
            cv2.circle(copie_couleur, (int(coin_x), int(coin_y)), 6, (0, 0, 255), -1)

        cv2.imwrite(f"resultats/coins_couleur_{indice}.jpg", copie_couleur)




    else:
        print("Chessboard not found")

    print(chemin_photo, ":", trouve)

calibration = cv2.calibrateCamera(points_3d, points_2d, taille_image, None, None)

photo_8_corrigee = cv2.undistort(
    image_couleur,
    cameraMatrix=calibration[1],
    distCoeffs=calibration[2]
)
cv2.imwrite("resultats/corrigee_photo_8.jpg", photo_8_corrigee)

photo_5 = cv2.imread("photos/photo_5.jpg")
photo_5_corrigee = cv2.undistort(
    photo_5,
    cameraMatrix=calibration[1],
    distCoeffs=calibration[2]
)
cv2.imwrite("resultats/corrigee_photo_5.jpg", photo_5_corrigee)

valeurs_camera = cv2.calibrationMatrixValues(
    cameraMatrix=calibration[1],
    imageSize=taille_image,
    apertureWidth=9.80,
    apertureHeight=7.30
)

pose = cv2.solvePnP(
    objectPoints=points_3d[7],
    imagePoints=points_2d[7],
    cameraMatrix=calibration[1],
    distCoeffs=calibration[2]
)

image_axes = cv2.drawFrameAxes(
    image= image_couleur,
    cameraMatrix=calibration[1],
    distCoeffs=calibration[2],
    rvec=pose[1],
    tvec=pose[2],
    length=3,
)
cv2.imwrite("resultats/axes.jpg", image_axes)

print(pose[1], pose[2])
sommets_cube = np.array([
    [0,0,0],
    [0,1,0],
    [1,0,0],
    [1,1,0],

    [0,0,-1],
    [0,1,-1],
    [1,0,-1],
    [1,1,-1],
    ],dtype=np.float32)

projection_cube = cv2.projectPoints(
    objectPoints=sommets_cube,
    rvec=pose[1],
    tvec=pose[2],
    cameraMatrix=calibration[1],
    distCoeffs=calibration[2]
)

sommets_2d = projection_cube[0].astype(np.int32).reshape(8,2)

arete_AB = cv2.line(image_axes, (sommets_2d[0]), (sommets_2d[1]), (255,0,0), 2)
arete_AC = cv2.line(image_axes, (sommets_2d[0]), (sommets_2d[2]), (255,0,0), 2)
arete_CD = cv2.line(image_axes, (sommets_2d[2]), (sommets_2d[3]), (255,0,0), 2)
arete_BD = cv2.line(image_axes, (sommets_2d[1]), (sommets_2d[3]), (255,0,0), 2)

arete_EF = cv2.line(image_axes, (sommets_2d[4]), (sommets_2d[5]), (0,255,0), 2)
arete_EG = cv2.line(image_axes, (sommets_2d[4]), (sommets_2d[6]), (0,255,0), 2)
arete_GH = cv2.line(image_axes, (sommets_2d[6]), (sommets_2d[7]), (0,255,0), 2)
arete_FH = cv2.line(image_axes, (sommets_2d[5]), (sommets_2d[7]), (0,255,0), 2)

arete_AE = cv2.line(image_axes, (sommets_2d[0]), (sommets_2d[4]), (0,0,255), 2)
arete_BF = cv2.line(image_axes, (sommets_2d[1]), (sommets_2d[5]), (0,0,255), 2)
arete_CG = cv2.line(image_axes, (sommets_2d[2]), (sommets_2d[6]), (0,0,255), 2)
arete_DH = cv2.line(image_axes, (sommets_2d[3]), (sommets_2d[7]), (0,0,255), 2)

cv2.imwrite("resultats/axes_cube.jpg", arete_DH)


# cv2.waitKey(0)
# cv2.destroyAllWindows()
