
# ======================================================================================================================
# Author: Jonas Anderegg, jonas.anderegg@usys.ethz.ch
# Sample training data for lesion classification
# Last edited: 2021-11-10
# ======================================================================================================================


from image_segmentor_tr import ImageSegmentor


workdir = "Z:/Public/Jonas/001_LesionZoo/MRE"


def run():
    dir_positives = f'{workdir}/train_data_lesions/Positives/'
    dir_negatives = f'{workdir}/train_data_lesions/Negatives/'
    dir_model = f'{workdir}/Output/Models/rf_segmentation.pkl'
    image_segmentor = ImageSegmentor(dir_positives=dir_positives,
                                     dir_negatives=dir_negatives,
                                     dir_model=dir_model,
                                     save_output=True)
    image_segmentor.iterate_images()


if __name__ == '__main__':
    run()