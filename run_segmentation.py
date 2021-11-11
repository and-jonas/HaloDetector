
# ======================================================================================================================
# Author: Jonas Anderegg, jonas.anderegg@usys.ethz.ch
# Project: LesionZoo
# Date: 23.06.2021
# ======================================================================================================================

# ======================================================================================================================
# For parallel processing
# ======================================================================================================================

# from image_segmentor_mult import ImageSegmentor
#
#
# def run():
#     dir_to_process = f"{workdir}/sample_images"
#     dir_model = f"{workdir}/Output/Models/rf_segmentation.pkl"
#     dir_output = f"{workdir}/Output"
#     image_segmentor = ImageSegmentor(dir_to_process=dir_to_process,
#                                      dir_output=dir_output,
#                                      dir_model=dir_model)
#     image_segmentor.run()
#
#
# if __name__ == '__main__':
#     run()

# ======================================================================================================================
# For sequential processing
# ======================================================================================================================

from image_segmentor2 import ImageSegmentor


workdir = "P:/Public/Jonas/001_LesionZoo/MRE"


def run():
    dir_positives = f"{workdir}/sample_images"
    dir_negatives = ""
    dir_model = f"{workdir}/Output/Models/rf_segmentation.pkl"
    file_index = [0, 200]
    image_segmentor = ImageSegmentor(dir_positives=dir_positives,
                                     dir_negatives=dir_negatives,
                                     dir_model=dir_model,
                                     file_index=file_index,
                                     save_output=True)
    image_segmentor.iterate_images(img_type='prediction')


if __name__ == '__main__':
    run()

