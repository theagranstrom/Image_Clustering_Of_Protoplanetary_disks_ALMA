""" Det här är en samanställning av bildbehandlingen som används i slutrapporten. För tidigare varianter av
funktionerna, se projektets github.

Här följer stegen i bildbehandlingsprocessen:
1. crop_circle: Bilden beskärs till cirkeln med halva radien för att utesluta det värsta bruset.
2. gaussian_fit: En normalfördelning anpassas till kvarvarande pixelvärden.
4. zoom: Ett område kring objektet lokaliseras.
    - Pixlar med värde högre än medelvärde + k std-avvikelser anses kunna tillhöra objektet.
5. slice_: Originalbilden beskärs till området som hittats av zoom.
6. find_rotation_factor: Hittar rotationsfaktorn som avgör hur roterat objektet är.
7. stretch: Skaländrar och roterar mha rotationsfaktorn.
8. save_data: Sparar data om bilderna i en json-fil.
9. normalize: Normaliserar bilden.
10. convert_to_jpg: Gör om bilden (array) till en jpg-fil och sparar.

*. process_images: Master-funktion som genomför hela bildbehandlingen ovan.
"""

import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from math import floor, ceil
from pathlib import Path
import json
from PIL import Image
from scipy.ndimage import gaussian_filter


# --- Beskärning av bilderna ---
def crop_circle(data, cx, cy, r):
    """
    Beskär bilden till en cirkel med mindre radie
    :param data: Bilden som beskärs (numpy-array).
    :param cx: x-värde för mittpunkten.
    :param cy: y-värde för mittpunkten.
    :param r: Radie på cirkeln.
    :return: Beskärd bild (numpy-array).
    """
    x = np.arange(data.shape[0])
    y = np.arange(data.shape[1])
    mask = (x[:, None]-cx)**2 + (y[None, :]-cy)**2 <= r**2
    new_data = np.where(mask, data, 0)

    return new_data


# --------- Anpassande av normalfördelning ---------
def gaussian_fit(data):
    """
    Anpassar en endimensionell normalfördelning till bilden.
    :param data: Bilden (numpy-array).
    :return: Medelvärde och standardavvikelse
    """
    # Tar bort alla nan-värden
    data = np.nan_to_num(data)

    # Plattar ut matrisen
    data_flat = data.flatten()

    # Tar bort peaken av nollor som skapas av alla NaN värden.
    data_flat = data_flat[data_flat != 0]

    # Beräknar medel och std
    mu = np.mean(data_flat)
    std = np.std(data_flat)

    return mu, std


# --------- Lokalisering och förstoring av objekt ---------
def zoom(img, factor=0.0, border=0.2, exact_factor=False):
    """
    Lokaliserar det starkast lysnade objektet och returnerar till var bilden bör beskäras.

    :param img: Bilden som genomsökes (numpy-array).
    :param factor: Avgör vilka värden som anses tillhöra objektet. Max 1 (endast maxpunkten),
                    min 0 (allting större än medelvärdet).
    :param border: Storlek på ramen i andel av den förstorade bilden.
    :param exact_factor: True om factor ges som ett exakt värde.
    :return: Koordinater för övre vänstra och nedre högra hörnet av området.
    """

    # -- Kontroller --
    if not isinstance(img, np.ndarray):
        print("Error: Img must be a numpy array.")
        raise ValueError("Img not a numpy array.")

    if (factor < 0 or factor > 1) and not exact_factor:
        print("Error: factor must be between 0 and 1 (if not exact_factor).")
        raise ValueError("Factor not between 0 and 1.")

    if border < 0:
        print("Error: border must be non-negative.")
        raise ValueError("Border < 0.")

    # -- Beräkning --
    # x och y för maxpunkten
    ma_x, ma_y = np.where(img == max(img[~np.isnan(img)]))

    # Värde i maxpunkten
    ma = max(img[~np.isnan(img)])

    # Kontrollerar att bilden har värdefulla värden
    if ma < factor and exact_factor or ma == 0:
        print("Error: No source found.")
        raise ValueError("No source found.")

    # Startar gränserna längst ut
    x_mi, x_ma, y_mi, y_ma = 0, img.shape[0], 0, img.shape[1]

    # Startar föregående gränser längst ut
    x_mi_old, x_ma_old, y_mi_old, y_ma_old = 0, img.shape[0], 0, img.shape[1]

    if exact_factor:
        # Hittar alla x resp. y där värdet är större än factor
        core_x, core_y = np.where(img >= factor)
    else:
        # Hittar alla x resp. y där värdet är större än factor*(max-mean) + mean
        core_x, core_y = np.where(img >= factor * max(img[~np.isnan(img)]) + (1 - factor) * np.mean(img[~np.isnan(img)]))

    # Loopar tills ett nytt steg inte ger någon extra inzoomning
    while True:
        # Tar bort värden ur core som befinner sig utanför det nuvarande området
        core_x, core_y = [x for (x, y) in zip(core_x, core_y) if x_ma_old >= x >= x_mi_old and y_ma_old >= y >= y_mi_old], \
                         [y for (x, y) in zip(core_x, core_y) if x_ma_old >= x >= x_mi_old and y_ma_old >= y >= y_mi_old]

        # Hittar första kolumnerna resp. raderna från mitten som inte har några pixlar i core
        # - Dessa innesluter vårt "område" som innehåller objektet.
        for i in range(*ma_x, x_mi_old-1, -1):
            if i not in core_x:
                x_mi = i
                break
        for i in range(*ma_x, x_ma_old):
            if i not in core_x:
                x_ma = i + 1
                break
        for i in range(*ma_y, y_mi_old-1, -1):
            if i not in core_y:
                y_mi = i
                break
        for i in range(*ma_y, y_ma_old):
            if i not in core_y:
                y_ma = i + 1
                break

        # Avbryter om detta steg inte zoomade in något extra.
        if x_mi == x_mi_old and x_ma == x_ma_old and y_mi == y_mi_old and y_ma == y_ma_old:
            # Lägger till border och returnar två hörn
            return [(max(x_mi - round((x_ma - x_mi)*border), 0), max(y_mi - round((y_ma - y_mi)*border), 0)),
                    (min(x_ma + round((x_ma - x_mi)*border), img.shape[0]), min(y_ma + round((y_ma - y_mi)*border), img.shape[1]))]

        # Uppdaterar föregående gränser till området
        x_mi_old, x_ma_old, y_mi_old, y_ma_old = x_mi, x_ma, y_mi, y_ma


# ------ Beskärning av originalbilden ------
def slice_(img, corners):
    """
    Beskär bilden till den som innesluts av "corners".
    :param img: Bild (numpy-array)
    :param corners: Koordinater för övre vänstra och nedre högra hörnet.
    :return: Beskärd bild (numpy-array)
    """
    return img[corners[0][0]: corners[1][0], corners[0][1]: corners[1][1]]


# --------- Beräkning av rotationsfaktor ---------
def _rot_from_diag(max_val, d1, d2, factor=0.0, exact_factor=False):
    """
    Hjälpfunktion för att hitta rotationsfaktorn.

    :param max_val: Maxvärde i bilden.
    :param d1: Första diagonalen
    :param d2: Andra diagonalen.
    :param factor: Avgör vilka värden som anses tillhöra objektet. Max 1 (endast maxpunkten),
                    min 0 (alla pixlar).
    :param exact_factor: True om factor ges som ett exakt värde.
    :return: Rotationsfaktorn (float)
    """
    # --- Kontroller ---
    if (factor < 0 or factor > 1) and not exact_factor:
        print("Error: factor must be between 0 and 1 (if not exact_factor).")
        raise ValueError("Factor not between 0 and 1.")

    # Kontrollerar att diagonalerna är lika långa. Funktionen kräver en kvadratisk bild.
    size = len(d1)
    if size != len(d2):
        print("Error: diagonals not equal length")
        raise ValueError("Diagonals not equal length.")

    # Stegar diagonalt inåt från hörnen tills dess att medelvärdet av motstående pixlar är större än factor*max_val.
    # - Detta ger ett mått på hur många pixlar lång objektet är längs med respektive diagonal.
    len1 = 0
    len2 = 0
    for i in range((size+1)//2):
        if d1[i] + d1[-i-1] > 2*factor*max_val and not exact_factor or exact_factor and d1[i] + d1[-i-1] > 2*factor:
            len1 = size - 2*i
            break
    for i in range((size+1)//2):
        if d2[i] + d2[-i-1] > 2*factor*max_val and not exact_factor or exact_factor and d2[i] + d2[-i-1] > 2*factor:
            len2 = size - 2*i
            break

    # Om någon av längderna inte hittades
    if len1 == 0 or len2 == 0:
        print("Error: can't find rotation")
        return 1

    # Returnerar rotationsfaktorn
    return len1/len2


def find_rotation_factor(img, goal_size, factor, exact_factor=False):
    """
    Hittar rotationsfaktorn, det vill säga kvoten mellan längden på diagonalerna hos objektet.
    :param img: Bilden (numpy-array).
    :param goal_size: Sidan på den slutgiltiga bilden i antal pixlar. Bilden måste vara kvadratisk
    :param factor: Avgör vilka värden som anses tillhöra objektet. Max 1 (endast maxpunkten),
                    min 0 (hela bilden).
    :param exact_factor: True om factor ges som ett exakt värde.
    :return: Rotationsfaktorn (float)
    """

    # -- Kontroller --
    if not isinstance(img, np.ndarray):
        print("Error: Img must be a numpy array.")
        raise ValueError("Img not a numpy array.")

    if (factor < 0 or factor > 1) and not exact_factor:
        print("Error: factor must be between 0 and 1 (if not exact_factor).")
        raise ValueError("Factor not between 0 and 1.")

    if goal_size <= 0:
        print("Error: goal_size must be positive.")
        raise ValueError("Negative goal_size.")

    if img.shape[0] == 0 or img.shape[1] == 0:
        print("Error: No image.")
        raise ValueError("Image has no size.")

    # -- Beräkning --
    # Storlek på gamla bilden
    old_xsize = img.shape[0]
    old_ysize = img.shape[1]

    # Räknar ut färger på diagonalerna i den nya bilden om vi *inte* roterar.
    # - Se "ändrar storlek" i stretch-funktionen för förklaringar.
    diag1 = [0]*goal_size
    diag2 = [0]*goal_size

    for i, j in [(t, t) for t in range(goal_size)] + [(t, goal_size-1-t) for t in range(goal_size)]:
        x_new = i + 0.5
        y_new = j + 0.5

        x_old = x_new / goal_size * old_xsize
        y_old = y_new / goal_size * old_ysize

        i_old = max(x_old - 0.5, 0)
        j_old = max(y_old - 0.5, 0)

        inv_dist = [(1 - i_old % 1) * (1 - j_old % 1), (i_old % 1) * (1 - j_old % 1),
                    (1 - i_old % 1) * (j_old % 1), (i_old % 1) * (j_old % 1)]

        nearest_values = [img[floor(i_old), floor(j_old)], img[min(ceil(i_old), old_xsize - 1), floor(j_old)],
                          img[floor(i_old), min(ceil(j_old), old_ysize - 1)],
                          img[min(ceil(i_old), old_xsize - 1), min(ceil(j_old), old_ysize - 1)]]

        if i == j:
            diag1[i] = sum(np.multiply(inv_dist, nearest_values)) / sum(inv_dist)
        else:
            diag2[i] = sum(np.multiply(inv_dist, nearest_values)) / sum(inv_dist)

    # Hittar rotationsfaktor
    return _rot_from_diag(max(img[~np.isnan(img)]), diag1, diag2, factor=factor, exact_factor=exact_factor)


# --------- Skaländring och rotation ---------
def stretch(img, xsize, ysize, rotation_factor=1):
    """
    Ändrar skala på bilden och fixar rotation på objektet.
    :param img: Bilden (numpy-array).
    :param xsize: Storlek i antal pixlar i x-led på slutbilden.
    :param ysize: Storlek i antal pixlar i y-led på slutbilden.
    :param rotation_factor: Rotationsfaktorn. 1 om ingen rotation ska genomföras.
    :return: Den skaländrade bilden (numpy-array).
    """

    # --- Kontroller ---
    if not isinstance(img, np.ndarray):
        print("Error: Img must be a numpy array.")
        raise ValueError("Img not a numpy array.")

    if xsize <= 0 or ysize <= 0:
        print("Error: Side-lengths must be positive.")
        raise ValueError("Side-lengths must be positive.")

    if xsize != ysize and rotation_factor != 1:
        print("Error: Rotation requires xsize=ysize.")
        raise ValueError("Rotation requires xsize=ysize.")

    if img.shape[0] == 0 or img.shape[1] == 0:
        raise ValueError("No image.")

    # --- Byter storlek på bilden ---
    # Skapar en ny (tom) bild med rätt fromat
    new_img = np.zeros((xsize, ysize))

    # Storlek på gamla bilden
    old_xsize = img.shape[0]
    old_ysize = img.shape[1]

    # ---- Ändrar storlek ----
    # Loopar över varje pixel
    for i in range(xsize):
        for j in range(ysize):
            # x och y är mitten av pixeln
            x_new = i + 0.5
            y_new = j + 0.5

            # Beräknar vart dessa x och y hamnar i gamla bilden
            x_old = x_new / xsize * old_xsize
            y_old = y_new / ysize * old_ysize

            # Fixar problemet med rotation över diagonalen
            if rotation_factor > 1:
                # Rotation över y=x
                x_old = (x_old + y_old * old_xsize / old_ysize + 1 / rotation_factor * (
                            x_old - y_old * old_xsize / old_ysize)) / 2
                y_old = (y_old + x_old * old_ysize / old_xsize + 1 / rotation_factor * (
                            y_old - x_old * old_ysize / old_xsize)) / 2

            elif rotation_factor < 1:
                # Rotation över y=-x
                x_old = (x_old + old_xsize - y_old * old_xsize / old_ysize + rotation_factor * (
                            x_old - (old_xsize - y_old * old_xsize / old_ysize))) / 2
                y_old = (y_old + old_ysize - x_old * old_ysize / old_xsize + rotation_factor * (
                            y_old - (old_ysize - x_old * old_ysize / old_xsize))) / 2

            # Detta är för att kunna jobba med heltal istället för .5 på allt.
            # - max så att kanterna inte försöker hitta pixlar utanför bilden.
            i_old = max(x_old - 0.5, 0)
            j_old = max(y_old - 0.5, 0)

            # Invers distans till närmaste 4 pixlarna. Nära = stort, långt ifrån = litet
            # Separata normaliseringar för x- resp. y-led. Linjärt.
            inv_dist = [(1 - i_old % 1) * (1 - j_old % 1), (i_old % 1) * (1 - j_old % 1),
                            (1 - i_old % 1) * (j_old % 1), (i_old % 1) * (j_old % 1)]

            # Värdet ("färgen") på de 4 närmaste pixlarna.
            # - min är (precis som innan) för att resten av kanterna inte ska kolla utanför bilden.
            nearest_values = [img[floor(i_old), floor(j_old)], img[min(ceil(i_old), old_xsize - 1), floor(j_old)],
                              img[floor(i_old), min(ceil(j_old), old_ysize - 1)],
                              img[min(ceil(i_old), old_xsize - 1), min(ceil(j_old), old_ysize - 1)]]

            # Multiplicerar grannarnas värden med inversa distansen, summerar och normerar.
            new_img[i, j] = sum(np.multiply(inv_dist, nearest_values)) / sum(inv_dist)
    return new_img


# --------- Sparande av bilddata ---------
def save_data(fits_files, fits_dir, json_dir, border, sigma_level, size=32, rotate=True):
    """
    Sparar data om fits-filerna i en json-fil.
    :param fits_files: FITS-filer som ska behandlas.
    :param fits_dir: Mapp med FITS-filerna. Måste sluta med "/".
    :param json_dir: Mapp där json-filen placeras. Måste sluta med "/".
    :param border: Storlek på ramen i andel av den förstorade bilden.
    :param sigma_level: Antal sigma över medelvärdet som används i zoom och rotation.
    :param size: Storlek på den slutgiltiga bilden.
    :param rotate: Boolean. True om bilden roteras, False annars.
    :return: None
    """
    # --- Kontroller ---
    if fits_dir[-1] != "/":
        print("Error: fits_dir must end with '/'.")
        ValueError("fits_dir must end with '/'")
    if json_dir[-1] != "/":
        print("Error: json_dir must end with '/'.")
        ValueError("json_dir must end with '/'")

    # --- Variabler ---
    # Innehåll i FITS-filerna
    data = [fits.getdata(Path(fits_dir + x))[0][0] for x in fits_files]

    # Dict som samlar all data
    to_save = dict()

    for i, im in enumerate(data):
        # -- Genomför bildbehandlingen --
        # Beskär bilden till cirkel med halva radien.
        cropped = crop_circle(im, im.shape[0] // 2, im.shape[1] // 2, im.shape[0] // 4)

        # -- Anpassar normalfördelning --
        mean_, standard_ = gaussian_fit(cropped)

        # -- Zoomar in på objektet --
        corners = zoom(cropped, factor=mean_ + sigma_level * standard_, border=border, exact_factor=True)
        core = slice_(im, corners)

        # -- Hittar rotationsfaktor --
        if rotate:
            rot_factor = find_rotation_factor(core, size, mean_ + sigma_level * standard_, exact_factor=True)
        else:
            rot_factor = 1

        # -- Sparar bilddata --
        # Hämtar data från fits-headern
        with fits.open(Path(fits_dir + fits_files[i])) as f:
            hdr = f[0].header
            object_ = hdr["OBJECT"]

        to_save[files[i]] = {"shape_before_resize": core.shape, "rotation_factor": rot_factor, "object": object_}

        # -- Counter --
        print(f'{i + 1}/{len(data)}')

    # -- Sparar i json --
    if save_data:
        with open(Path(json_dir + "image_data.json"), "w") as f:
            json.dump(to_save, f)


def read_data(path="./data/image_data.json"):
    """
    Läser sparad data om fits-filerna.
    :param path: Sökväg för json-fil (string).
    :return: Dictionary med datan.
    """
    with open(Path(path), "r") as f:
        data = json.load(f)
    return data


# --------- Normalisering ---------
def normalize(data, low=0, high=1):
    """
    Normaliserar data mellan low och high.
    :param data: Bilden som ska normaliseras (numpy-array).
    :param low: Lägsta värdet.
    :param high: Högsta värdet.
    :return: Den normaliserade bilden (numpy-array).
    """
    data_min = np.nanmin(data)
    data_max = np.nanmax(data)

    data = np.nan_to_num(data, nan=np.nanmin(data))
    data = ((data - data_min) / (data_max - data_min)) * (high - low) + low
    return data


# --------- Gör till .jpg --------
def convert_to_jpg(img, path_):
    """
    Omvandlar en numpy-array till jpg. Bilden bör normaliseras mellan 0 och 255 först.
    :param img: Bilden (numpy-array).
    :param path_: Mapp för output.
    :return: None
    """

    im_done = Image.fromarray(img)  # Konverterar matrisdata till gråskaliga pixlar
    if im_done.mode != 'RGB':
        im_done = im_done.convert('RGB')

    with open(Path(path_), "w") as f:
        im_done.save(f)


# --------- Master-funktion ----------
def process_images(fits_files, fits_dir, out_dir, border, sigma_level, size=32, rotate=True, save_data=True,
                   plot_=False, make_jpg=True, overwrite_jpg=-1, blur=False, blur_range=2, blur_exp=1.5, blur_sigma=7):
    """
    Master-funktion för bildbehandling. Genomför hela bildbehandlingen.
    :param fits_files: FITS-filer som ska behandlas.
    :param fits_dir: Mapp som innehåller FITS-filerna.
    :param out_dir: Output-mapp för färdiga jpg-bilder och json med data.
    :param border: Storlek på ramen i andel av den förstorade bilden.
    :param sigma_level: Antal sigma över medelvärdet som används i zoom och rotation.
    :param size: Storlek på den slutgiltiga bilden.
    :param rotate: Boolean. True om bilden ska roteras.
    :param save_data: Boolean. True om data om bilderna ska sparas.
    :param plot_: Boolean. True om bilderna ska plottas.
    :param make_jpg: Boolean. True om bilderna ska göras till jpg.
    :param overwrite_jpg: True om gamla jpg-filer ska skrivas över. False om inte. Vid annat värde frågas innan.
    :param blur: Boolean. True om en blur av bilden ska användas som mask på slutresultatet.
    :param blur_range: Övre gräns på normaliseringen för blur-funktionen. Alla pixlar med värde < 1 påverkas.
    :param blur_exp: Exponent på blur-funktionen.
    :param blur_sigma: Sigma för gaussian-blur.
    :return: None
    """

    # --- Kontroller ---
    if fits_dir[-1] != "/":
        print("Error: fits_dir must end with '/'.")
        ValueError("fits_dir must end with '/'")
    if out_dir[-1] != "/":
        print("Error: out_dir must end with '/'.")
        ValueError("out_dir must end with '/'")

    # --- Variabler ---
    # Innehåll i FITS-filerna
    data = [fits.getdata(Path(fits_dir + x))[0][0] for x in fits_files]

    # Dict som samlar all data
    to_save = dict()

    # Räknar antalet bilden där ingen source hittas
    noSourceCount = 0

    for i, im in enumerate(data):
        # -- Genomför bildbehandlingen --
        # Beskär bilden till cirkel med halva radien.
        cropped = crop_circle(im, im.shape[0] // 2, im.shape[1] // 2, im.shape[0] // 4)

        # -- Anpassar normalfördelning --
        mean_, standard_ = gaussian_fit(cropped)

        # -- Zoomar in på objektet --
        try:
            corners = zoom(cropped, factor=mean_ + sigma_level * standard_, border=border, exact_factor=True)
        except ValueError:  # Kontrollerar att bilden innehåller någonting
            noSourceCount += 1
            continue
        core = slice_(im, corners)

        # -- Blur som används som mask till bilden --
        if blur:
            norm_core = normalize(core, 0, blur_range)
            norm_core[norm_core > 1] = 1
            norm_core = norm_core ** blur_exp
            blurred = gaussian_filter(norm_core, sigma=blur_sigma)
            core = np.multiply(blurred, core)

        # -- Hittar rotationsfaktor --
        if rotate:
            rot_factor = find_rotation_factor(core, size, mean_ + sigma_level * standard_, exact_factor=True)
        else:
            rot_factor = 1

        # -- Skaländrar och roterar --
        stretched = stretch(core, size, size, rot_factor)

        # -- Normaliserar --
        normal = normalize(stretched, high=255)

        # -- Sparar bilddata --
        # Hämtar data från fits-headern
        with fits.open(Path(fits_dir + fits_files[i])) as f:
            hdr = f[0].header
            object_ = hdr["OBJECT"]

        to_save[fits_files[i]] = {"shape_before_resize": core.shape, "rotation_factor": rot_factor, "object": object_}

        # -- Gör om till jpg --
        if make_jpg:
            # Kontrollerar om jpg redan finns och agerar utifrån overwrite_jpg.
            if overwrite_jpg != 1 and os.path.isfile(out_dir + files[i] + ".jpg"):
                if not overwrite_jpg:
                    print(f"{i + 1}/{len(data)}: Overwriting of jpg blocked.")
                    continue
                else:
                    overwrite_jpg = input("WARNING: Old jpg-files are about to be overwritten. "
                                          "1: Continue. 0: Continue without overwriting. Enter: Break.")
                    if overwrite_jpg == "1":
                        convert_to_jpg(normal, out_dir + files[i] + ".jpg")
                    elif overwrite_jpg != "0":
                        return 0
                    overwrite_jpg = int(overwrite_jpg)
            else:
                # Gör om till jpg
                convert_to_jpg(normal, out_dir + files[i] + ".jpg")

        # -- Counter --
        print(f'{i + 1}/{len(data)}')

        # -- Plottar --
        if plot_:
            fig, ax = plt.subplots()
            ax.axis("off")

            fig.add_subplot(2, 2, 1)
            plt.axis("off")
            plt.title("Originalbild")
            plt.imshow(im)

            fig.add_subplot(2, 2, 2)
            plt.axis("off")
            plt.title("Beskärd")
            plt.imshow(cropped)

            fig.add_subplot(2, 2, 3)
            plt.axis("off")
            plt.title("Objekt hittat")
            plt.imshow(core)

            fig.add_subplot(2, 2, 4)
            plt.axis("off")
            if rotate:
                plt.title("Skaländrad och roterad")
            else:
                plt.title("Skaländrad")
            plt.imshow(stretched)
            plt.show()

    # -- Sparar i json --
    if save_data:
        with open(Path(out_dir + "image_data.json"), "w") as f:
            json.dump(to_save, f)

    print(f"Images where no source was found: {noSourceCount}/{len(data)}")


if __name__ == "__main__":
    files = [x for x in os.listdir("data/fits") if x[-4:] == "fits"
             and x + ".jpg" in os.listdir("data/datasets/sigma7_rotated_border=0.2")]
    process_images(files, "./data/fits/", "./data/datasets/sigma5_rotated_border=0.2_only-sigma7/", 0.2, 5,
                   save_data=True, make_jpg=True, plot_=False, blur=False, rotate=True)

