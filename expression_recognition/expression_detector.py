import argparse
import os
import numpy as np

import expression_recognition.api as exp_recog


def get_files_list(folder, ftype="jpg"):
    frm = ".{}".format(ftype)
    return [os.path.join(folder, f) for f in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, f)) and f.endswith(frm)]

def filter_selected(items, indexes, th=1):
    return [f for (f,i) in zip(items, indexes) if i>=th]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="folder with images to detect")
    parser.add_argument("--landmarks", help="use precalculated landmark files instead of images",
                        action="store_true")
    
    args = parser.parse_args()
    folder = args.folder
    use_landmarks = args.landmarks

    if use_landmarks:
        files = get_files_list(folder, "txt")
        if len(files) < 1:
            print("No txt file found, nothing to detect")
            quit()

        landmarks = exp_recog.read_landmarks(files)
        files_with_faces = files
    else:
        files = get_files_list(folder)
        if len(files) < 1:
            print("No jpg file found, nothing to detect")
            quit()

        landmarks, indexes = exp_recog.compute_landmarks(files, True)

        files_with_faces = filter_selected(files, indexes)
    
    smiles_predictions = exp_recog.detect_smile(landmarks)
    open_mouthes_predictions = exp_recog.detect_open_mouth(landmarks)
    
    smiles =  filter_selected(files_with_faces, smiles_predictions)
    open_mouthes = filter_selected(files_with_faces, open_mouthes_predictions)
    
    print("SMILES")
    print('\t'.join(smiles))
    print('\n')
    
    print("OPEN MOUTHES")
    print('\t'.join(open_mouthes))


if __name__ == '__main__':
    main()