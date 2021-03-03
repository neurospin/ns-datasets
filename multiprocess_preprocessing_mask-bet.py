"""
pynet data preprocessing : n processes script
========================
Script to launch a multi-processing pre-processing pipeline

Pipeline details:
    reorient to standard
    scale : 1 iso
    skullstripped with fsl functions
    Correction of biasfield with ants
    Linear registration to MNI template
    Noises around the brain corrections (caused by linear regitration)


"""

import os
import sys

import re
import subprocess
import nibabel
import argparse

from pynet.preprocessing import reorient2std
from pynet.preprocessing import scale
from pynet.preprocessing import bet2
from pynet.preprocessing import super_bet2
from pynet.preprocessing import register
from pynet.preprocessing import EraseNoise
from pynet.preprocessing import biasfield
from pynet.preprocessing import Processor

from threading import Thread, RLock

verrou = RLock()


class Preprocessing_dl(Thread):

    """ Thread in charge of the preprocessing of a set of images """

    def __init__(self, list_images, list_already_done, dest_path):
        Thread.__init__(self)
        self.list_images = list_images
        self.list_already_done = list_already_done
        self.dest_path = dest_path

    def run(self):
        """ Code to execute during thread execution.
            Apply pipeline and write results in bids format
        """
        

        target = nibabel.load('/i2bm/local/fsl/data/standard'
                              '/MNI152_T1_1mm_brain.nii.gz')
        for file, root in self.list_images:
            # execute pipeline
            print("\nthe file processed is : ", file)
            path_image = os.path.join(root, file)
            image = nibabel.load(path_image)
            pipeline = Processor()
            pipeline.register(reorient2std, check_pkg_version=False,
                              apply_to="image")
            pipeline.register(scale, scale=1, check_pkg_version=False,
                              apply_to="image")
            pipeline.register(super_bet2, target=target,
                              check_pkg_version=False,
                              apply_to="image")
            pipeline.register(biasfield, check_pkg_version=False,
                              apply_to="image")
            pipeline.register(register, target=target,
                              check_pkg_version=False,
                              apply_to="image")
            pipeline.register(EraseNoise, check_pkg_version=False,
                              apply_to="image")
            normalized = pipeline(image)

            # write the results in bids format
            path_tmp = path_image.split(os.sep)
            filename = path_tmp[-1]
            newfilename = filename.split("_")
            newfilename.insert(2, "preproc-linear")
            newfilename = "_".join(newfilename)
            ses = path_tmp[-3]
            sub = path_tmp[-4]
            sub_dest = os.path.join(self.dest_path, sub)
            ses_dest = os.path.join(sub_dest, ses)
            anatdest = os.path.join(ses_dest, "anat")
            # create filetree
            subprocess.check_call(['mkdir', '-p', anatdest])
            if re.search(".gz", file):
                end_path = "/{0}/{1}/anat/{2}"\
                           .format(sub, ses, newfilename)
            else:
                end_path = "/{0}/{1}/anat/{2}.gz"\
                           .format(sub, ses, newfilename)
            dest_file = self.dest_path+end_path
            # save results
            nibabel.save(normalized, dest_file)
            with verrou:
                # write already preprocessed images
                already_done = os.path.join(self.dest_path, "already_done.txt")
                with open(already_done, "a") as file1:
                    ligne = file+"\n"
                    file1.write(ligne)


def read_alreadydone(dest_path):
    """Read already preprocessed images file.

    Parameters
    ----------
    dest_path: string
        path to the output.

    Returns
    -------
    list_already_done: list
        list of the already preprocessed image.
    """
    list_already_done = []
    already_done = os.path.join(dest_path, "already_done.txt")
    if os.path.exists(already_done):
        with open(already_done, "r") as file1:
            for line in file1.readlines():
                list_already_done.append(line[0:-1])
    else:
        file_object = open(already_done, "x")
        file_object.close()
    return list_already_done


def divise_namelist(path_rawdata, number_process, list_already_done):
    """Divides images to preprocessed into n batches.

    Parameters
    ----------
    path_rawdata: string
        path to the rawdata folder.
    number_process: int
        number of processes to launch
    list_already_done: list
        list of the already preprocessed images.

    Returns
    -------
    out: list of list
        list of batches to launch.
    """
    list1 = []
    for root, dirs, files in os.walk(path_rawdata):
        for file in files:
            if re.search("T1w[_]*[a-zA-Z0-9]*.nii", file)\
               and file not in list_already_done:
                list1.append([file, root])
    avg = len(list1)/float(number_process)
    out = []
    last = 0.0
    while last < len(list1):
        out.append(list1[int(last):int(last+avg)])
        last += avg
    return out


def main():

    # terminal command options
    parser = argparse.ArgumentParser()
    parser.add_argument('--rawdata', help='path to rawdata', nargs='+', required=True, type=str)
    parser.add_argument('-o', '--output', help='path to quasi-raw output', nargs='+', required=True, type=str)
    parser.add_argument('-j', help='number of threads', nargs='+', required=True, type=int)
    options = parser.parse_args()

    if options.rawdata is None:
        parser.print_help()
        raise SystemExit("Error: Rawdata is missing.")

    if options.output is None:
        parser.print_help()
        raise SystemExit("Error: Output is missing.")

    if options.j is None:
        parser.print_help()
        raise SystemExit("Error: Number of threads is missing.")

    # initilization
    path_rawdata = options.rawdata[0]
    dest_path = options.output[0]
    number_process = options.j[0]
    list_already_done = read_alreadydone(dest_path)

    ## launch n processes

    # split rawdata liste
    biglist = divise_namelist(path_rawdata, number_process, list_already_done)

    # check if there are any unprocessed images left
    if len(biglist) > 0:
        # threads name list
        processes_list = []
        processes_obj_list = []
        for i in range(number_process):
            processes_list.append("thread_"+str(i))
        # print(processes_list)
        # threads creation
        for c, j in enumerate(processes_list):
            # create processes
            j = Preprocessing_dl(biglist[c], list_already_done, dest_path)
            j.name = processes_list[c]
            processes_obj_list.append(j)
        # threads launch
        for j in processes_obj_list:
            j.start()
        # waiting for threads to finish
        for j in processes_obj_list:
            j.join()
    else:
        print("no more T1w to processed")


if __name__ == "__main__":
    main()
