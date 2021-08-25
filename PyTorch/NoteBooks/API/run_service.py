#!/usr/bin/python3
import os
import yaml, json, subprocess, shutil, pydicom, cv2, time
from collections import OrderedDict
from utils import os_makedirs
import numpy as np
import boto3
import requests
# from flask import Flask, request
# from flask import render_template
# from flask import jsonify
from sanic import Sanic
from sanic import response

# app = Flask(__name__)
app = Sanic(__name__)

from botocore.utils import is_valid_endpoint_url
from botocore.client import Config
from botocore.exceptions import ClientError, EndpointConnectionError

imagesTr_dir = '/claraDevDay/Data/customData/imagesTr'
labelsTr_dir = '/claraDevDay/Data/customData/labelsTr'
mmar_dir = '/claraDevDay/MMAR'
config_file = '/tmp/data/config/config.yaml'

organ2mmar = {
    "liver": "clara_pt_liver_and_tumor_ct_segmentation",
    "spleen": "clara_pt_spleen_ct_segmentation",
    "pancreas": "clara_pt_pancreas_and_tumor_ct_segmentation" 
}

def download_s3_file(endpoint, access_key, secret_key, bucket, dicom_filepath, dicom_file):
    config = Config(signature_version='s3')
    is_verify = False
    connection = boto3.client(
        's3',
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        endpoint_url=endpoint,
        config=config,
        verify=is_verify
    )
    docker_dicom_filepath = os.path.join(docker_dicom_dir, dicom_file)

    try:
        connection.download_file(bucket, dicom_filepath, docker_dicom_filepath)
    except ClientError as e:
        pass

    if os.path.isfile(docker_dicom_filepath):
        return True
    else:
        return False

empty_response = {  "version": "",
                    "flags": {},
                    "shapes": [],
                    "lineColor": [],
                    "fillColor": [],
                    "imagePath": "",
                    "imageData": "",
                    "imageHeight": 0,
                    "imageWidth": 0,
                    "message": ""
                 }

def get_contour_point_num(contour, coeff):
    epsilon = coeff*cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    return approx.shape[0]

def binary_search_coeff(contour, coeff_l = 0.01, coeff_h = 0.001, target_point_num = 100):
    num_l = get_contour_point_num(contour, coeff_l)
    num_h = get_contour_point_num(contour, coeff_h)
    # print("binary_search_coeff")
    while (num_l>=target_point_num):
        coeff_l = coeff_l * 2
        num_l = get_contour_point_num(contour, coeff_l)

    while (num_h<=target_point_num):
        coeff_h = coeff_h / 2
        num_h = get_contour_point_num(contour, coeff_h)

    # print("target_point_num = {}".format(target_point_num))
    # print("Initial coeff_l = {} ; num_l = {}".format(coeff_l, num_l))
    # print("Initial coeff_h = {} ; num_h = {}".format(coeff_h, num_h))

    num_l_old = num_l
    num_h_old = num_h
    count = 0
    while (num_l < num_h):
        coeff_m = (coeff_l+coeff_h)/2

        # log10_coeff_m = (math.log10(coeff_l) + math.log10(coeff_h)) / 2
        # coeff_m = math.pow(10, log10_coeff_m)

        # loge_coeff_m = (math.log(coeff_l) + math.log(coeff_h)) / 2
        # coeff_m = math.exp(loge_coeff_m)

        num_m = get_contour_point_num(contour, coeff_m)

        # print("coeff_l = {} ; num_l = {}".format(coeff_l, num_l))
        # print("coeff_m = {} ; num_m = {}".format(coeff_m, num_m))
        # print("coeff_h = {} ; num_h = {}".format(coeff_h, num_h))
        # print("= "*10)
        # print("")
        if num_m >= target_point_num-3 and num_m <= target_point_num+3:
            return coeff_m

        if num_m > target_point_num:
            coeff_h = coeff_m
            num_h = num_m
        else:
            coeff_l = coeff_m
            num_l = num_m

        if num_l_old == num_l and num_h_old == num_h:
            count = count + 1
        else:
            count = 0
        num_l_old = num_l
        num_h_old = num_h
        if count > 3:
            return coeff_h

def merge_list(L, R):
    result = list()

    i = j = 0
    while i < len(L) and j < len(R):
        if L[i] < R[j]:
            result.append(L[i])
            i += 1
        else:
            result.append(R[j])
            j += 1

    if i < len(L):
        result.extend(L[i:])

    if j < len(R):
        result.extend(R[j:])
    return result

def insert_list(org_list, element):
    if len(org_list)==0:
        org_list.append(element)
    elif element > org_list[-1]:
        org_list.append(element)
    else:
        l = 0
        r = len(org_list)-1

        while l<=r:
            m = (l+r)//2
            if org_list[m] >= element and (m-1 < 0 or org_list[m-1] < element):
                org_list.insert(m, element)
                return m
            else:
                if element >= org_list[m]:
                    l = m + 1
                else:
                    r = m - 1

def sample_from_xy_list(xy_list, AIAA_points):
    points = len(xy_list)
    if points <= AIAA_points:
        return xy_list
    else:
        rest = list(range(points))
        select = list()

        while len(select) < AIAA_points:
            interval = len(rest) / (AIAA_points-len(select))
            interval = int(round(interval, 0))
            if interval==1:
                interval=2

            if interval>=len(rest):
                element = random.choice(rest)
                insert_list(select, element)
                rest.remove(element)
            else:
                select_from_rest = rest[0::interval]
                # print("len(select) = {}".format(len(select)))
                # print("len(select_from_rest) = {}".format(len(select_from_rest)))
                select = merge_list(select, select_from_rest)
                rest = sorted(list(set(rest) - set(select_from_rest)))
            # print("interval = {} ; select = {} ; rest = {}".format(interval, len(select), len(rest)))
            # print("")

        # # print(select)
        # print(len(xy_list), AIAA_points, len(select))
        # print(select)
        # print(len(list(np.array(xy_list)[select])))
        # print(list(np.array(xy_list)[select])[0])
        return [list(ele) for ele in np.array(xy_list)[select]]

def check_dcm_equivalent(dcm_file_path, nii_dcm_file_path):
    ds = pydicom.read_file(dcm_file_path)
    dcm_img = ds.pixel_array
    intercept = ds.RescaleIntercept
    slope = ds.RescaleSlope
    dcm_img = slope * dcm_img + intercept
    dcm_img = dcm_img.astype('int16')

    ds = pydicom.read_file(nii_dcm_file_path)
    nii_dcm_img = ds.pixel_array

    return np.array_equal(dcm_img, nii_dcm_img)

def rename_seg_nii_dcm(dcm_dir, nii_dcm_dir, seg_nii_dcm_dir): # seg_nii_dcm_dir
    dcm_list = sorted(os.listdir(dcm_dir))
    nii_dcm_list = sorted(os.listdir(nii_dcm_dir), reverse=True)
    assert len(dcm_list)==len(nii_dcm_list), "len(dcm_list)!=len(nii_dcm_list)"

    dcm_file_path = os.path.join(dcm_dir, dcm_list[0])
    nii_dcm_file_path = os.path.join(nii_dcm_dir, nii_dcm_list[0])
    if not check_dcm_equivalent(dcm_file_path, nii_dcm_file_path):
        nii_dcm_list.reverse()

    for idx, (dcm_file, nii_dcm_file) in enumerate(zip(dcm_list, nii_dcm_list)):
        dcm_file_path = os.path.join(dcm_dir, dcm_file)
        nii_dcm_file_path = os.path.join(nii_dcm_dir, nii_dcm_file)
        assert check_dcm_equivalent(dcm_file_path, nii_dcm_file_path), "check_dcm_equivalent fails"
        seg_nii_dcm_file_path = os.path.join(seg_nii_dcm_dir, nii_dcm_file)
        seg_dcm_file_path = os.path.join(seg_nii_dcm_dir, dcm_file)
        if os.path.exists(seg_nii_dcm_file_path):
            os.rename(seg_nii_dcm_file_path, seg_dcm_file_path)

def dcm_to_png(dcm_file_path):
    ds = pydicom.read_file(dcm_file_path)

    img = ds.pixel_array
    img = img.astype('float64')
    intercept = ds.RescaleIntercept
    wc = ds.WindowCenter[0]
    ww = ds.WindowWidth[0]
    UL = wc + ww/2
    LL = wc - ww/2
    img -= (-intercept+LL)
    img[img<0] = 0
    img[img>(UL-LL)] = (UL-LL)
    img *= 255.0/img.max()
    img = img.astype('uint8')
    return img

def seg_dcm_to_png_and_json(seg_dcm_dir, png_dir, json_dir, organ):
    seg_dcm_list = sorted(os.listdir(seg_dcm_dir))
    for seg_dcm_file in seg_dcm_list:
        seg_dcm_file_path = os.path.join(seg_dcm_dir, seg_dcm_file)
        ds = pydicom.read_file(seg_dcm_file_path)
        img = ds.pixel_array
        img[img>0] = 255
        img = img.astype('uint8')

        if img.any():
            png_file_name = os.path.splitext(seg_dcm_file)[0] + '.png'
            png_file_path = os.path.join(png_dir, png_file_name)
            cv2.imwrite(png_file_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])


            
            contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            json_dict = OrderedDict()
            for idx in range(len(contours)):
                if idx==0:
                    json_dict["version"] = "4.5.6"
                    json_dict["flags"] = dict()
                    json_dict["shapes"] = list()
                    json_dict["imagePath"] = png_file_name
                    json_dict["imageData"] = None
                    json_dict["imageHeight"] = img.shape[0]
                    json_dict["imageWidth"] = img.shape[1]

                downsample = False
                downsample_algo = 0
                samples = 20
                # Douglas-Peucker algorithm (cv2.approxPolyDP)
                if downsample_algo==0 and contours[idx].shape[0]>samples:
                    downsample = True
                    t_start = time.time()
                    print("Before DP algo = {}".format(contours[idx].shape[0]))
                    coeff = binary_search_coeff(contours[idx], target_point_num = samples)
                    epsilon = coeff*cv2.arcLength(contours[idx], True)
                    approx = cv2.approxPolyDP(contours[idx], epsilon, True)
                    contours[idx] = approx
                    print("After  DP algo = {}; time = {}".format(contours[idx].shape[0], time.time()-t_start))

                xy_list = np.squeeze(contours[idx], axis=1).tolist()
                if (len(xy_list)<3):
                    continue
                else:
                    shapes = OrderedDict()
                    shapes["label"] = organ
                    # shapes["points"] = xy_list
                    shapes["group_id"] = None
                    shapes["shape_type"] = "polygon"
                    shapes["flags"] = dict()

                    # equal sample
                    if downsample_algo==1 and len(xy_list)>samples:
                        downsample = True
                        t_start = time.time()
                        print("Before equal sample = {}".format(len(xy_list)))
                        shapes["points"] = sample_from_xy_list(xy_list, samples)
                        print("After  equal sample = {}; time = {}".format(len(shapes["points"]), time.time()-t_start))
                    else:
                        shapes["points"] = xy_list

                    json_dict["shapes"].append(shapes)

            json_file_name = os.path.splitext(seg_dcm_file)[0] + '.json'
            json_file_path = os.path.join(json_dir, json_file_name)
            with open(json_file_path, 'w') as json_file:
                json.dump(json_dict, json_file, sort_keys=False, indent=2, separators=(',', ': '))

# dcm2niix -f 01160825 -o ./ 01160825
# liver, spleen, pancreas
# curl -X POST http://localhost:1234/pneu --header "Content-Type: application/json" --data '{"dcm_dir_name": "01160825", "organ": "liver"}'

@app.route("/pneu", methods=['POST'])
async def pneu_inference(request):
    request_dict = request.json
    dcm_dir_name = request_dict.get('dcm_dir_name', None)
    organ = request_dict.get('organ', None)

    dcm_dir = os.path.join(imagesTr_dir, dcm_dir_name)

    organ_mmar_dir = os.path.join(mmar_dir, organ2mmar[organ])

    nii_file_name = '{}.nii.gz'.format(dcm_dir_name)
    nii_file_path = os.path.join(imagesTr_dir, nii_file_name)
    nii_dcm_dir = os.path.join(imagesTr_dir, '{}_nii_dcm'.format(dcm_dir_name))
    '''
    if os.path.exists(nii_file_path):
        os.remove(nii_file_path)
    if os.path.exists(nii_dcm_dir):
         shutil.rmtree(nii_dcm_dir)

    json_file_name = '{}.json'.format(dcm_dir_name)
    json_file_path = os.path.join(imagesTr_dir, json_file_name)

    cmd = "dcm2niix -z y -f {} -o {} {}".format(dcm_dir_name, imagesTr_dir, dcm_dir)
    proc = subprocess.Popen(cmd, shell=True)
    proc.wait()
    if os.path.exists(json_file_path):
        os.remove(json_file_path)

    cmd = "nifti2dicom -i {} -o {} -a 123456".format(nii_file_path, nii_dcm_dir)
    proc = subprocess.Popen(cmd, shell=True)
    proc.wait()



    dataset_json_file_path = os.path.join(organ_mmar_dir, "config/dataset.json")
    dataset_json_dict = json.load(open(dataset_json_file_path))
    dataset_json_dict["test"] = ["imagesTr/{}".format(nii_file_name)]
    json.dump(dataset_json_dict, open(dataset_json_file_path, "w"), indent=4)

    cmd = "bash {}".format(os.path.join(organ_mmar_dir, "commands/infer.sh"))
    proc = subprocess.Popen(cmd, shell=True)
    proc.wait()
    '''
    seg_nii_file_name = '{}_seg.nii.gz'.format(dcm_dir_name)
    seg_nii_file_path = os.path.join(organ_mmar_dir, "eval/{}/{}".format(dcm_dir_name, seg_nii_file_name))
    seg_nii_dcm_dir = os.path.join(organ_mmar_dir, "eval/{0}/{0}_seg".format(dcm_dir_name))
    seg_nii_png_dir = os.path.join(organ_mmar_dir, "eval/{0}/{0}_png".format(dcm_dir_name))
    seg_nii_json_dir = os.path.join(organ_mmar_dir, "eval/{0}/{0}_json".format(dcm_dir_name))
    os_makedirs(seg_nii_png_dir)
    os_makedirs(seg_nii_json_dir)
    
    '''
    if os.path.exists(seg_nii_dcm_dir):
         shutil.rmtree(seg_nii_dcm_dir)
    cmd = "nifti2dicom -i {} -o {} -a 123456".format(seg_nii_file_path, seg_nii_dcm_dir)
    proc = subprocess.Popen(cmd, shell=True)
    proc.wait()
    '''

    rename_seg_nii_dcm(dcm_dir, nii_dcm_dir, seg_nii_dcm_dir)
    seg_dcm_to_png_and_json(seg_nii_dcm_dir, seg_nii_png_dir, seg_nii_json_dir, organ)

    return response.json({})


# 
'''
# curl -X POST http://localhost:81/pneu --header "Content-Type: application/json" --data '{"endpoint" : "http://61.219.26.12:8080", "access_key" : "327abedcf2d64324b9a82fc65b4cf265", "secret_key" : "DCWHm9t2Vak3eXPCw0ZPtM0cgfVioyIK5", "bucket" : "pneu-dicom", "file" : "0001-PNEUMO2013021401.dcm"}'
@app.route("/pneu", methods=['POST'])
async def pneu_inference(request):
    request_dict = request.json
    endpoint = request_dict.get('endpoint', None)
    access_key = request_dict.get('access_key', None)
    secret_key = request_dict.get('secret_key', None)
    bucket = request_dict.get('bucket', None)
    dicom_filepath = request_dict.get('file', None)
    downsample_algo = request_dict.get('downsample_algo', None)
    samples = request_dict.get('samples', None)

    # Download file from S3 blob
    if endpoint and access_key and secret_key and bucket and dicom_filepath:
        dicom_file = dicom_filepath.split('/')[-1]
        if not download_s3_file(endpoint, access_key, secret_key, bucket, dicom_filepath, dicom_file):
            empty_response["message"] = "The file does not download from S3 blob"
            return response.json(empty_response)
    else:
        empty_response["message"] = "The API request information is not complete"
        return response.json(empty_response)

    # Query pneumothorax inference API
    pneu_headers = {'Content-Type': 'application/json'}
    pneu_data = {"dcm": dicom_file}
    if downsample_algo is not None:
        pneu_data["downsample_algo"] = downsample_algo
    if samples is not None:
        pneu_data["samples"] = samples

    try:
        r = requests.post('http://localhost:5050/ADV_pneu', headers = pneu_headers, json = pneu_data)
    except requests.exceptions.RequestException as e:
        empty_response["message"] = "Pneumothorax inference API does not exist"
        return response.json(empty_response)

    if r.status_code == requests.codes.ok:
        response_dict = r.json()
        signal = response_dict.get('signal', None)
        errcode = response_dict.get('errcode', None)
        errmessage = response_dict.get('errmessage', None)
        host_heatmap_json_path = response_dict.get('heatmap_json_path', None)

        if host_heatmap_json_path:
            if os.path.isfile(config_file):
                with open(config_file) as file:
                    config_dict = yaml.load(file, Loader=yaml.FullLoader)
                    if config_dict==None:
                        config_dict = {}
            else:
                config_dict = {}

            path = config_dict.get('PATH', None)
            docker_heatmap_json_path = host_heatmap_json_path.replace(path, docker_data_dir)
            json_dict = json.load(open(docker_heatmap_json_path))
            for i in range(len(json_dict["shapes"])):
                json_dict["shapes"][i]["label"] = "pneumothorax_seg"
            return response.json(json_dict)
        else:
            empty_response["message"] = "No AIAA json file is created"
            if signal == "0":
                empty_response["message"] = empty_response["message"] + " due to no pneumothorax"
            elif errmessage:
                empty_response["message"] = empty_response["message"] + " due to " + errmessage
            return response.json(empty_response)
    else:
        empty_response["message"] = "Pneumothorax inference error ({})".format(r.status_code)
        return response.json(empty_response)

@app.route("/label", methods=['POST', 'GET'])
async def get_label(request):
    return response.text("pneumothorax_seg")
'''

if __name__ == "__main__":
    app.run(host = '0.0.0.0', port = 1234)
