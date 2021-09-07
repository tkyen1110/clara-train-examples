#!/usr/bin/env python3
import os, sys, shutil

def os_makedirs(dst_dir, keep_exists=False):
    if keep_exists:
        if not os.path.isdir(dst_dir):
            os.makedirs(dst_dir)
    else:
        if os.path.isdir(dst_dir):
            shutil.rmtree(dst_dir)
        os.makedirs(dst_dir)

def os_remove(dst_file_path):
    if os.path.isfile(dst_file_path):
        os.remove(dst_file_path)

def shutil_rmtree(dst_dir):
    if os.path.isdir(dst_dir):
        shutil.rmtree(dst_dir)

def shutil_copytree(src_dir, dst_dir):
    shutil.copytree(src_dir, dst_dir)

def shutil_copyfile(src_file_path, dst_file_path):
    shutil.copyfile(src_file_path, dst_file_path)

def shutil_move(src_file_path, dst_dir):
    src_file_name = os.path.basename(src_file_path)
    dst_file_path = os.path.join(dst_dir, src_file_name)
    if (os.path.exists(dst_file_path)):
        os.remove(dst_file_path)
    shutil.move(src_file_path, dst_dir)