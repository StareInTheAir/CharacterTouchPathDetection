import math
import os
import plistlib
import shutil

import numpy


def transform_plist(plist_path):
    plist_dict = plistlib.load(open(plist_path, 'rb'))
    cleaned_data = list(map(
        lambda sample:
        {
            'class': chr(ord('A') + sample['label']),
            'points': list(map(lambda point: (point['x'], point['y']), sample['points']))
        },
        plist_dict['samples']
    ))
    return cleaned_data


def add_neighboring_pairs(data):
    for sample in data:
        sample['pairs'] = []
        for current, nxt in zip(sample['points'][:], sample['points'][1:]):
            sample['pairs'].append((current, nxt))


def add_angles(data):
    for sample in data:
        sample['angles'] = []
        for pair in sample['pairs']:
            sample['angles'].append(
                math.degrees(math.atan2(pair[1][1] - pair[0][1], pair[1][0] - pair[0][0])))


def get_offset_angle_histogram(angles, bins, offset, norm=True):
    hist = numpy.histogram(list(map(lambda angle: loop_number(angle + offset, -180, 180), angles)),
                           bins=bins,
                           range=(-180, 180))[0]
    if norm:
        hist = hist / numpy.linalg.norm(hist)
    return hist


def write_bin_header(file, dataset_name, bins):
    file.write('@RELATION {}{}'.format(dataset_name, os.linesep))
    for bin in range(1, bins + 1):
        file.write('@ATTRIBUTE bin{} NUMERIC{}'.format(bin, os.linesep))

    file.write(
        '@ATTRIBUTE class {{{}}}{}@DATA{}'.format(','.join(map(chr, range(ord('A'), ord('Z') + 1))),
                                                  os.linesep,
                                                  os.linesep))


def main():
    data = transform_plist('samples.plist')
    add_neighboring_pairs(data)
    add_angles(data)
    output_directory = 'weka-files'

    if os.path.exists(output_directory):
        if os.path.isdir:
            shutil.rmtree(output_directory)
        else:
            raise Exception(output_directory + ' is not a directory')

    os.mkdir(output_directory)
    # for sample in data:
    #     print(sample['class'])
    #     print(get_offset_angle_histogram(sample['angles'], 8, 0))
    #     print()
    for bins in [3, 4, 5, 6, 8, 10, 12, 16, 20]:
        for offset in [(0, 'no'), (360 / bins / 2, 'half'), (360 / bins / 4, 'quarter')]:
            dataset_name = 'charReg-vectorHistogram-{:02d}bins-{}Offset'.format(bins, offset[1])
            file = open((os.path.join(output_directory, dataset_name) + '.arff'), 'w')
            write_bin_header(file, dataset_name, bins)
            for sample in data:
                histogram = get_offset_angle_histogram(sample['angles'], bins, offset[0])
                numpy_str_arr = numpy.char.mod('%f', histogram)
                file.write(','.join(numpy_str_arr))
                file.write(',' + sample['class'] + os.linesep)
            print('wrote ' + file.name)
            file.close()


def loop_number(value, min, max):
    range = abs(min - max)
    while value > max:
        value -= range
    while value < min:
        value += range
    return value


if __name__ == '__main__':
    main()
