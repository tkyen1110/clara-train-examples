#!/usr/bin/python3
import os
import time
import yaml, json
import pydicom, cv2
import subprocess
from collections import OrderedDict
from utils import *
import numpy as np
import base64, io
import PIL.Image
from multiprocessing import Process

import requests
from sanic import Sanic
from sanic import response

import boto3
from botocore.utils import is_valid_endpoint_url
from botocore.client import Config
from botocore.exceptions import ClientError, EndpointConnectionError

from point_reduction import get_contour_point_num, binary_search_coeff, merge_list, insert_list, sample_from_xy_list

app = Sanic(__name__)

imagesTr_dir = '/claraDevDay/Data/customData/imagesTr'
labelsTr_dir = '/claraDevDay/Data/customData/labelsTr'
mmar_dir = '/claraDevDay/MMAR'
config_file = '/tmp/data/config/config.yaml'

organ2mmar = {
    "liver": "clara_pt_liver_and_tumor_ct_segmentation",
    "spleen": "clara_pt_spleen_ct_segmentation",
    "pancreas": "clara_pt_pancreas_and_tumor_ct_segmentation" 
}

def connect_to_s3(endpoint, access_key, secret_key):
    config = Config(signature_version='s3')
    is_verify = True
    connection = boto3.client(
        's3',
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        endpoint_url=endpoint,
        config=config,
        verify=is_verify
    )
    return connection

def download_s3_file(connection, bucket, s3_file_path, docker_file_path):
    try:
        connection.download_file(bucket, s3_file_path, docker_file_path)
    except ClientError as e:
        pass

    if os.path.isfile(docker_file_path):
        return True
    else:
        return False

def upload_s3_file(connection, bucket, docker_file_path, s3_file_path):
    try:
        connection.upload_file(docker_file_path, bucket, s3_file_path)
        return True
    except ClientError as e:
        return False

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

def rename_seg_nii_dcm(dcm_dir, nii_dcm_dir, seg_nii_dcm_dir):
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

def org_dcm_to_png(dcm_dir, png_dir):
    dcm_list = sorted(os.listdir(dcm_dir))

    for dcm_file in dcm_list:
        dcm_file_path = os.path.join(dcm_dir, dcm_file)
        img = dcm_to_png(dcm_file_path)

        png_file_name = os.path.splitext(dcm_file)[0] + '.png'
        png_file_path = os.path.join(png_dir, png_file_name)
        cv2.imwrite(png_file_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])


def img_b64_to_arr(img_b64):
    f = io.BytesIO()
    f.write(base64.b64decode(img_b64))
    img_arr = np.array(PIL.Image.open(f))
    return img_arr


def img_arr_to_b64(img_arr):
    img_pil = PIL.Image.fromarray(img_arr)
    f = io.BytesIO()
    img_pil.save(f, format='PNG')
    img_bin = f.getvalue()
    if hasattr(base64, 'encodebytes'):
        img_b64 = base64.encodebytes(img_bin)
    else:
        img_b64 = base64.encodestring(img_bin)
    return img_b64

def seg_dcm_to_png_and_json(org_png_dir, seg_dcm_dir, seg_png_dir, seg_json_dir, organ):
    seg_dcm_list = sorted(os.listdir(seg_dcm_dir))
    # kernel_1 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    # kernel_2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

    for seg_dcm_file in seg_dcm_list:
        seg_dcm_file_path = os.path.join(seg_dcm_dir, seg_dcm_file)
        ds = pydicom.read_file(seg_dcm_file_path)
        img = ds.pixel_array
        img[img>0] = 255
        img = img.astype('uint8')

        save_empty_json = False
        png_file_name = os.path.splitext(seg_dcm_file)[0] + '.png'

        org_png_file_path = os.path.join(org_png_dir, png_file_name)
        img_pil = PIL.Image.open(org_png_file_path)
        img_arr = np.array(img_pil)
        imageData = img_arr_to_b64(img_arr).decode('ascii')

        if img.any():
            png_file_path = os.path.join(seg_png_dir, png_file_name)
            cv2.imwrite(png_file_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])

            # dilation = cv2.dilate(img, kernel_1, iterations = 1)
            # erosion = cv2.erode(dilation, kernel_2, iterations = 1)
            contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            new_contours = []
            for k in range(len(contours)):
                area = cv2.contourArea(contours[k])
                if area < 25:
                    continue
                else:
                    new_contours.append(contours[k])
            contours = new_contours

            if len(contours)>0:
                json_dict = OrderedDict()
                for idx in range(len(contours)):
                    if idx==0:
                        json_dict["version"] = "4.5.6"
                        json_dict["flags"] = dict()
                        json_dict["shapes"] = list()
                        json_dict["imagePath"] = png_file_name
                        json_dict["imageData"] = imageData
                        json_dict["imageHeight"] = img.shape[0]
                        json_dict["imageWidth"] = img.shape[1]

                    downsample = False
                    downsample_algo = 0
                    samples = 50
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
            else:
                save_empty_json = True
        else:
            save_empty_json = True

        if save_empty_json:
            json_dict = OrderedDict()
            json_dict["version"] = "4.5.6"
            json_dict["flags"] = dict()
            json_dict["shapes"] = list()
            json_dict["imagePath"] = png_file_name
            json_dict["imageData"] = imageData
            json_dict["imageHeight"] = img.shape[0]
            json_dict["imageWidth"] = img.shape[1]

        json_file_name = os.path.splitext(seg_dcm_file)[0] + '.json'
        json_file_path = os.path.join(seg_json_dir, json_file_name)
        with open(json_file_path, 'w') as json_file:
            json.dump(json_dict, json_file, sort_keys=False, indent=2, separators=(',', ': '))

def clara_inference(request_dict):
    t_start_all = time.time()
    dcm_dir_name = request_dict.get('dcm_dir_name', None)
    organ = request_dict.get('organ', None)

    dcm_dir = os.path.join(imagesTr_dir, dcm_dir_name)

    # Convert dicom to png for labelme
    # /claraDevDay/Data/customData/imagesTr/{dcm_dir_name} -->
    # /claraDevDay/Data/customData/imagesTr/{dcm_dir_name}_png
    png_dir = os.path.join(imagesTr_dir, dcm_dir_name + "_png")
    os_makedirs(png_dir)
    org_dcm_to_png(dcm_dir, png_dir)

    # Remove redundant files
    # files:   /claraDevDay/Data/customData/imagesTr/{dcm_dir_name}.nii.gz
    # folders: /claraDevDay/Data/customData/imagesTr/{dcm_dir_name}_nii_dcm
    nii_file_name = '{}.nii.gz'.format(dcm_dir_name)
    nii_file_path = os.path.join(imagesTr_dir, nii_file_name)
    nii_dcm_dir = os.path.join(imagesTr_dir, '{}_nii_dcm'.format(dcm_dir_name))
    os_remove(nii_file_path)
    shutil_rmtree(nii_dcm_dir)

    # Convert dicom to nii and remove redundant files
    # /claraDevDay/Data/customData/imagesTr/{dcm_dir_name} -->
    # /claraDevDay/Data/customData/imagesTr/{dcm_dir_name}.nii.gz
    t_start = time.time()
    cmd = "dcm2niix -z y -f {} -o {} {}".format(dcm_dir_name, imagesTr_dir, dcm_dir)
    proc = subprocess.Popen(cmd, shell=True)
    proc.wait()
    json_file_name = '{}.json'.format(dcm_dir_name)
    json_file_path = os.path.join(imagesTr_dir, json_file_name)
    os_remove(json_file_path)
    t_dicom2nifti = time.time() - t_start

    # Convert nii back to dicom in order to check the ordering
    # /claraDevDay/Data/customData/imagesTr/{dcm_dir_name}.nii.gz -->
    # /claraDevDay/Data/customData/imagesTr/{dcm_dir_name}_nii_dcm
    t_start = time.time()
    cmd = "nifti2dicom -i {} -o {} -a 123456".format(nii_file_path, nii_dcm_dir)
    proc = subprocess.Popen(cmd, shell=True)
    proc.wait()
    t_nifti2dicom_1 = time.time() - t_start

    '''
    # Prepare dataset.json in MMAR folder of specific organ
    t_start = time.time()
    organ_mmar_dir = os.path.join(mmar_dir, organ2mmar[organ])
    dataset_json_file_path = os.path.join(organ_mmar_dir, "config/dataset.json")
    dataset_json_dict = json.load(open(dataset_json_file_path))
    dataset_json_dict["test"] = ["imagesTr/{}".format(nii_file_name)]
    json.dump(dataset_json_dict, open(dataset_json_file_path, "w"), indent=4)
    t_json = time.time() - t_start

    # Inference by infer.sh
    # output: /claraDevDay/MMAR/clara_pt_liver_and_tumor_ct_segmentation/eval/{dcm_dir_name}/{dcm_dir_name}_seg.nii.gz
    t_start = time.time()
    cmd = "bash {}".format(os.path.join(organ_mmar_dir, "commands/infer.sh"))
    proc = subprocess.Popen(cmd, shell=True)
    proc.wait()
    t_infer = time.time() - t_start
    '''

    # Remove redundant files
    organ_mmar_dir = os.path.join(mmar_dir, organ2mmar[organ])
    seg_nii_file_name = '{}_seg.nii.gz'.format(dcm_dir_name)
    seg_nii_file_path = os.path.join(organ_mmar_dir, "eval/{}/{}".format(dcm_dir_name, seg_nii_file_name))
    seg_nii_dcm_dir = os.path.join(organ_mmar_dir, "eval/{0}/{0}_seg".format(dcm_dir_name))
    seg_nii_png_dir = os.path.join(organ_mmar_dir, "eval/{0}/{0}_png".format(dcm_dir_name))
    seg_nii_json_dir = os.path.join(organ_mmar_dir, "eval/{0}/{0}_json".format(dcm_dir_name))
    os_remove(seg_nii_file_path)
    shutil_rmtree(seg_nii_dcm_dir)
    os_makedirs(seg_nii_png_dir)
    os_makedirs(seg_nii_json_dir)

    # Inference by AIAA server and triton server
    t_start = time.time()
    cmd = "curl -X POST \"http://127.0.0.1:5000/v1/segmentation?model={}&output=image\" \
           -H \"accept: multipart/form-data\" -H \"Content-Type: multipart/form-data\" -F \"params={}\" \
           -F \"image=@{};type=application/x-gzip\" -o {} -f".format(organ2mmar[organ], "{}", nii_file_path, seg_nii_file_path)
    proc = subprocess.Popen(cmd, shell=True)
    proc.wait()
    t_infer = time.time() - t_start

    # Convert nii back to dicom in order to check the ordering
    # /claraDevDay/MMAR/clara_pt_liver_and_tumor_ct_segmentation/eval/{dcm_dir_name}/{dcm_dir_name}_seg.nii.gz -->
    # /claraDevDay/MMAR/clara_pt_liver_and_tumor_ct_segmentation/eval/{dcm_dir_name}/{dcm_dir_name}_seg
    t_start = time.time()
    cmd = "nifti2dicom -i {} -o {} -a 123456".format(seg_nii_file_path, seg_nii_dcm_dir)
    proc = subprocess.Popen(cmd, shell=True)
    proc.wait()
    t_nifti2dicom_2 = time.time() - t_start

    # Check the ordering before dicom2nifti and after nifti2dicom
    # (According to the files in dcm_dir and nii_dcm_dir to rename the files in seg_nii_dcm_dir)
    # dcm_dir (before dicom2nifti) / nii_dcm_dir (after nifti2dicom)
    t_start = time.time()
    rename_seg_nii_dcm(dcm_dir, nii_dcm_dir, seg_nii_dcm_dir)
    t_rename = time.time() - t_start

    # Convert dicom to png and json
    t_start = time.time()
    seg_dcm_to_png_and_json(png_dir, seg_nii_dcm_dir, seg_nii_png_dir, seg_nii_json_dir, organ)
    t_convert = time.time() - t_start

    # Remove intermediate files
    os_remove(nii_file_path)
    shutil_rmtree(nii_dcm_dir)

    os_remove(seg_nii_file_path)
    shutil_rmtree(seg_nii_dcm_dir)
    # shutil_rmtree(seg_nii_png_dir)
    # shutil_rmtree(seg_nii_json_dir)

    print("t_dicom2nifti   = {}".format(t_dicom2nifti))
    print("t_nifti2dicom_1 = {}".format(t_nifti2dicom_1))
    # print("t_json          = {}".format(t_json))
    print("t_infer         = {}".format(t_infer))
    print("t_nifti2dicom_2 = {}".format(t_nifti2dicom_2))
    print("t_rename        = {}".format(t_rename))
    print("t_convert       = {}".format(t_convert))
    print("t_all           = {}".format(time.time()-t_start_all))

    return seg_nii_json_dir


def update_progress(progress_dict, progress_json_path):
    with open(progress_json_path, 'w') as json_file:
        json.dump(progress_dict, json_file, sort_keys=False, indent=2, separators=(',', ': '))


def clara_func(request_dict, organ):
    endpoint = request_dict.get('endpoint', None)
    access_key = request_dict.get('access_key', None)
    secret_key = request_dict.get('secret_key', None)
    bucket = request_dict.get('bucket', None)
    asset_group = request_dict.get('asset_group', None)
    output_s3_folder = request_dict.get('output_s3_folder', None)

    connection = connect_to_s3(endpoint, access_key, secret_key)

    result_json = []
    progress_dict = {"total": "", "finished": "", "interrupted": 0, "asset_group": ""}
    for asset in asset_group:
        category_name = asset.get("category_name", None)
        result_json.append({"category_name": category_name, "status": "", "message": ""})
    progress_dict["total"] = len(asset_group)
    progress_dict["finished"] = 0
    progress_dict["asset_group"] = result_json
    organ_mmar_dir = os.path.join(mmar_dir, organ2mmar[organ])
    organ_eval_dir = os.path.join(organ_mmar_dir, "eval")
    progress_json_path = os.path.join(organ_eval_dir, '{}_log.json'.format(output_s3_folder.split('/')[-1]))
    progress_s3_path = os.path.join(output_s3_folder, '{}_log.json'.format(output_s3_folder.split('/')[-1]))
    update_progress(progress_dict, progress_json_path)
    upload_s3_file(connection, bucket, progress_json_path, progress_s3_path)

    for i, asset in enumerate(asset_group):
        category_name = asset.get("category_name", None)
        files = asset.get("files", None)

        dcm_dir_name = files[0].split('/')[-2]
        dcm_dir = os.path.join(imagesTr_dir, dcm_dir_name)

        # Download file from S3 blob
        print("Downloading {:<15s} folder from s3 blob to {}".format(dcm_dir_name, dcm_dir))
        result_json[i]["status"] = "processing"
        result_json[i]["message"] = "downloading sth from s3"
        progress_dict["asset_group"] = result_json
        update_progress(progress_dict, progress_json_path)

        os_makedirs(dcm_dir)
        j = 0
        for s3_file_path in files:
            dcm_file_name = os.path.basename(s3_file_path)
            dcm_file_path = os.path.join(dcm_dir, dcm_file_name)
            if not download_s3_file(connection, bucket, s3_file_path, dcm_file_path):
                result_json[i]["status"] = "fail"
                result_json[i]["message"] = "Cannot download {} from s3 blob".format(s3_file_path)
                progress_dict["finished"] = i + 1
                progress_dict["asset_group"] = result_json
                update_progress(progress_dict, progress_json_path)
                upload_s3_file(connection, bucket, progress_json_path, progress_s3_path)
                break
            j = j + 1
        if j != len(files):
            continue

        # Inference
        print("Inferring {}".format(dcm_dir_name))
        result_json[i]["status"] = "processing"
        result_json[i]["message"] = "inferring"
        progress_dict["asset_group"] = result_json
        update_progress(progress_dict, progress_json_path)
        clara_dict = {"dcm_dir_name": dcm_dir_name, "organ": organ}
        seg_nii_json_dir = clara_inference(clara_dict)

        # Upload file to S3 blob
        s3_file_dir = os.path.join(output_s3_folder, category_name, "json")
        print("Uploading jsons from {:<15s} folder to s3 blob {}".format(os.path.basename(seg_nii_json_dir), s3_file_dir))
        result_json[i]["status"] = "processing"
        result_json[i]["message"] = "uploading sth to s3"
        progress_dict["asset_group"] = result_json
        update_progress(progress_dict, progress_json_path)
        for seg_nii_json_file in os.listdir(seg_nii_json_dir):
            docker_file_path = os.path.join(seg_nii_json_dir, seg_nii_json_file)
            s3_file_path = os.path.join(s3_file_dir, seg_nii_json_file)
            if not upload_s3_file(connection, bucket, docker_file_path, s3_file_path):
                result_json[i]["status"] = "fail"
                result_json[i]["message"] = "Cannot upload {} to s3 blob".format(docker_file_path)
                progress_dict["finished"] = i + 1
                progress_dict["asset_group"] = result_json
                update_progress(progress_dict, progress_json_path)

        result_json[i]["status"] = "finished"
        result_json[i]["message"] = "clara inference is finished"
        progress_dict["finished"] = i + 1
        progress_dict["asset_group"] = result_json
        update_progress(progress_dict, progress_json_path)

empty_response = {
                    "message": ""
                 }

@app.route("/clara/<organ>", methods=['POST'])
async def clara(request, organ):
    proc = None
    if organ not in organ2mmar:
        empty_response["message"] = "Does not support {} inference. Support only {}"\
                                    .format(organ, ', '.join(organ2mmar.keys()))
        return response.json(empty_response)

    request_dict = request.json
    endpoint = request_dict.get('endpoint', None)
    access_key = request_dict.get('access_key', None)
    secret_key = request_dict.get('secret_key', None)
    bucket = request_dict.get('bucket', None)
    asset_group = request_dict.get('asset_group', None)
    output_s3_folder = request_dict.get('output_s3_folder', None)

    if endpoint and access_key and secret_key and bucket and asset_group:
        if isinstance(asset_group, list):
            proc = Process(target=clara_func, args=(request_dict, organ))
            proc.start()
            empty_response["message"] = "Total {} patiens will be dealt with by batch mode".format(len(asset_group))
            return response.json(empty_response)
        else:
            empty_response["message"] = "asset_group is not list"
            return response.json(empty_response)
    else:
        des_msg_list = []
        if not endpoint:
            des_msg_list.append("endpoint")
        if not access_key:
            des_msg_list.append("access_key")
        if not secret_key:
            des_msg_list.append("secret_key")
        if not bucket:
            des_msg_list.append("bucket")
        if not asset_group:
            des_msg_list.append("asset_group")

        empty_response["message"] = "Lack of {}".format(', '.join(des_msg_list))
        return response.json(empty_response)

    empty_response["message"] = "success"
    return response.json(empty_response)


@app.route("/clara_infer", methods=['POST'])
async def clara_infer(request):
    t_start_all = time.time()
    request_dict = request.json
    clara_inference(request_dict)
    return response.json({})


@app.route("/label", methods=['POST', 'GET'])
async def get_label(request):
    return response.text("clara_seg")


@app.route("/stop", methods=['POST'])
async def stop_infer(request):
    print("Stop inference")
    if proc:
        if proc.is_alive():
            proc.terminate()

            with open(progress_json_path, 'r') as json_file:
                progress_dict = json.load(json_file)

            progress_dict["interrupted"] = 1
            update_progress(progress_dict, progress_json_path)
            upload_s3_file(connection, bucket, progress_json_path, progress_s3_path)
            return response.json({"message": "stopped"})

    return response.json({"message": "no process running"})

if __name__ == "__main__":
    app.run(host = '0.0.0.0', port = 1234)
