import sys
sys.path.append("/neurospin/psy_sbox/git/ns-datasets")
import xml.etree.ElementTree as ET
import argparse
import glob
import ast
import re
import csv
import traceback
import sys
from cat12vbm_qc_utils import get_keys

def parse_xml_files(xml_filenames, output_file=None):
    # organized as /participant_id/sess_id/[TIV, GM, WM, CSF, ROIs]
    output = dict()
    ROI_names = None
    for xml_file in xml_filenames:

        xml_file_keys = get_keys(xml_file)
        participant_id = xml_file_keys['participant_id']
        session = xml_file_keys['session'] or 'V1'
        run = xml_file_keys['run'] or '1'

        # Parse the CAT12 report to find the TIV and CGW volumes
        if re.match('.*report/cat_.*\.xml', xml_file):
            tree = ET.parse(xml_file)
            try:
                tiv = float(tree.find('subjectmeasures').find('vol_TIV').text)
                vol_abs_CGW = list(ast.literal_eval(tree.find('subjectmeasures').find('vol_abs_CGW').text.
                                                    replace(' ', ',')))

            except Exception as e:
                print('Parsing error for %s:\n%s' % (xml_file, traceback.format_exc()))
            else:
                if participant_id not in output:
                    output[participant_id] = {session: dict()}
                elif session not in output[participant_id]:
                    output[participant_id][session] = {run: dict()}
                elif run not in output[participant_id][session]:
                    output[participant_id][session][run] = dict()
                output[participant_id][session][run]['TIV'] = float(tiv)
                output[participant_id][session][run]['CSF_Vol'] = float(vol_abs_CGW[0])
                output[participant_id][session][run]['GM_Vol'] = float(vol_abs_CGW[1])
                output[participant_id][session][run]['WM_Vol'] = float(vol_abs_CGW[2])

        elif re.match('.*label/catROI_.*\.xml', xml_file):
            tree = ET.parse(xml_file)
            try:
                _ROI_names = [item.text for item in tree.find('neuromorphometrics').find('names').findall('item')]
                if ROI_names is None:
                    ROI_names = _ROI_names
                elif set(ROI_names) != set(_ROI_names):
                    raise ValueError('Inconsistent ROI names from %s (expected %s, got %s) ' % (xml_file, ROI_names,
                                                                                                _ROI_names))
                else:
                    ROI_names = _ROI_names
                V_GM = list(ast.literal_eval(tree.find('neuromorphometrics').find('data').find('Vgm').text.
                                             replace(';', ',')))
                V_CSF = list(ast.literal_eval(tree.find('neuromorphometrics').find('data').find('Vcsf').text.
                                              replace(';', ',')))
                assert len(ROI_names) == len(V_GM) == len(V_CSF)

            except Exception as e:
                print('Parsing error for %s: \n%s' % (xml_file, traceback.format_exc()))
            else:
                for i, ROI_name in enumerate(ROI_names):
                    if participant_id not in output:
                        output[participant_id] = {session: {run: {ROI_name+'_GM_Vol': float(V_GM[i]), ROI_name+'_CSF_Vol': float(V_CSF[i])}}}
                    elif session not in output[participant_id]:
                        output[participant_id][session] = {run: {ROI_name+'_GM_Vol': float(V_GM[i]), ROI_name+'_CSF_Vol': float(V_CSF[i])}}
                    elif run not in output[participant_id][session]:
                        output[participant_id][session][run] = {ROI_name+'_GM_Vol': float(V_GM[i]), ROI_name+'_CSF_Vol': float(V_CSF[i])}
                    else:
                        output[participant_id][session][run][ROI_name+'_GM_Vol'] = float(V_GM[i])
                        output[participant_id][session][run][ROI_name+'_CSF_Vol'] = float(V_CSF[i])

    ROI_names = ROI_names or []
    fieldnames = ['participant_id', 'session', 'run', 'TIV', 'CSF_Vol', 'GM_Vol', 'WM_Vol'] + \
                 [roi + '_GM_Vol' for roi in ROI_names] + \
                 [roi + '_CSF_Vol' for roi in ROI_names]
    if output_file is not None:
        with open(output_file, 'w') as tsvfile:
            writer = csv.DictWriter(tsvfile, fieldnames=fieldnames, dialect="excel-tab")
            writer.writeheader()
            for participant_id in output:
                for session in output[participant_id].keys():
                    for (run, measures) in output[participant_id][session].items():
                        writer.writerow(dict(participant_id=participant_id, session=session, run=run, **measures))

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='A list of .xml files', required=True, nargs='+')
    parser.add_argument('--output', help='Name or path to the output TSV file', default='output_measure.tsv')

    options = parser.parse_args()
    xml_filenames = options.input

    parse_xml_files(xml_filenames, options.output)

# COMMAND TERMINAL
# cd BIDS_roi
# python3 /neurospin/psy_sbox/git/ns-datasets/adni/cat12vbm_roi.py --input /neurospin/cati/ADNI/adni/BIDS/sub-*/ses-*/anat/label/catROI_sub-*_ses-*_T1w.xml /neurospin/cati/ADNI/adni/BIDS/sub-*/ses-*/anat/report/cat_sub-*_ses-*_T1w.xml --output cat12-12.6_vbm_roi.tsv
