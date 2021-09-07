#!/usr/bin/python3
import cv2
import numpy as np

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
